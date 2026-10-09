"""Optional shared token accounting for the lead and all delegated model calls."""
import sys
import threading
import asyncio
import os
import re
import time
import math

import httpx

from langchain.agents.middleware import AgentMiddleware
from tools import RetryableError, with_retry, redact


def _transient(exc):
    if isinstance(exc, (httpx.TransportError, TimeoutError, ConnectionError)):
        return True
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if (code == 429 or type(exc).__name__ == "GoogleRateLimitError") and re.search(
            r"PerDay|per_day|per day|retry in \d+h", str(exc), re.I):
        return False
    if code in {429, 500, 502, 503, 504}:
        return True
    return type(exc).__name__ in {"GoogleRateLimitError", "GoogleServerError"}


def _model_retry_after(exc):
    text = str(exc)
    match = re.search(r"(?:retry in |retryDelay['\"]?\s*:\s*['\"]?)([\d.]+)s", text, re.I)
    if match:
        return float(match.group(1)) + 1
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            return float(response.headers.get("Retry-After"))
        except (ValueError, TypeError):
            pass
    return None


class TokenBudgetExceeded(RuntimeError):
    """The run exhausted its shared token allowance."""


class SharedTokenBudget:
    """Thread-safe usage ledger; reserves capacity before parallel calls start.

    Provider-reported usage replaces each reservation. Unknown usage keeps the
    conservative estimate. A provider may exceed an estimate, so this is a guard
    against runaway runs, not a billing guarantee.
    """
    def __init__(self, limit=1_500_000, reserve_output=8192, model_interval=None):
        if limit < 1 or reserve_output < 1:
            raise ValueError("token limits must be positive")
        self.limit = limit
        self.reserve_output = reserve_output
        self._lock = threading.Lock()
        self.used = self.reserved = self.input = self.output = self.calls = 0
        self.estimated_calls = 0
        self._warned = False
        default_interval = "5" if os.getenv("LAB_MODEL", "").startswith("google_genai:") else "0"
        self.model_interval = float(model_interval if model_interval is not None
                                    else os.getenv("LAB_MODEL_INTERVAL", default_interval))
        if self.model_interval < 0:
            raise ValueError("model interval must be nonnegative")
        self._next_model_slot = 0
        default_tpm = "200000" if os.getenv("LAB_MODEL", "").startswith("google_genai:") else "0"
        self.model_tpm = int(os.getenv("LAB_MODEL_TPM", default_tpm))
        if self.model_tpm < 0:
            raise ValueError("model token throughput must be nonnegative")
        self._recent_models = []

    def schedule_model(self, input_estimate=0):
        """Reserve a spaced start time while allowing network/tools to overlap."""
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_model_slot)
            if self.model_tpm and input_estimate > self.model_tpm:
                raise TokenBudgetExceeded("a model request exceeds the configured per-minute token allowance")
            self._recent_models = [(t, n) for t, n in self._recent_models if t > start - 60]
            while self.model_tpm and sum(n for _, n in self._recent_models) + input_estimate > self.model_tpm:
                start = max(start, self._recent_models[0][0] + 60.1)
                self._recent_models = [(t, n) for t, n in self._recent_models if t > start - 60]
            self._recent_models.append((start, input_estimate))
            self._next_model_slot = start + self.model_interval
            return max(0, start - now)

    def reserve(self, amount):
        with self._lock:
            if self.used + self.reserved + amount > self.limit:
                raise TokenBudgetExceeded(f"Shared token budget exhausted (limit={self.limit})")
            self.reserved += amount

    def finish(self, reservation, usage=None, *, failed=False):
        with self._lock:
            self.reserved -= reservation
            # On an error the provider might still have billed the request.
            actual = (usage or {}).get("total_tokens")
            if actual is None and usage:
                actual = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            self.used += actual if actual is not None else reservation
            self.input += (usage or {}).get("input_tokens", 0)
            self.output += (usage or {}).get("output_tokens", 0)
            self.calls += 1
            self.estimated_calls += int(actual is None)
            if self.used >= 0.8 * self.limit and not self._warned:
                print("[budget] shared token allowance is at least 80% consumed", file=sys.stderr)
                self._warned = True

    def snapshot(self):
        with self._lock:
            return {"limit": self.limit, "used_or_estimated": self.used,
                    "input": self.input, "output": self.output, "model_calls": self.calls,
                    "estimated_calls": self.estimated_calls,
                    "scope": "lead/subagent middleware calls; excludes internal context summarization"}


class SharedTokenBudgetMiddleware(AgentMiddleware):
    def __init__(self, budget):
        self.budget = budget

    def _estimate(self, request):
        messages = ([request.system_message] if request.system_message else []) + request.messages
        # Include tool definitions and message envelopes, not only content.
        payload = repr([m.model_dump() for m in messages]) + repr(request.tools)
        amount = len(payload.encode("utf-8")) + self.budget.reserve_output
        return amount

    def wrap_model_call(self, request, handler):
        def call():
            reservation = self._estimate(request)
            input_estimate = math.ceil((reservation - self.budget.reserve_output) / 3)
            time.sleep(self.budget.schedule_model(input_estimate))
            self.budget.reserve(reservation)
            try:
                response = handler(request)
            except BaseException as exc:
                self.budget.finish(reservation, failed=True)
                if _transient(exc):
                    raise RetryableError(f"transient model transport/service error: {type(exc).__name__}: {redact(exc)[:1200]}",
                                         _model_retry_after(exc)) from None
                raise
            usage = None
            for message in response.result:
                metadata = getattr(message, "usage_metadata", None)
                if metadata:
                    if usage is None:
                        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                    for field in usage:
                        usage[field] += metadata.get(field, 0)
            self.budget.finish(reservation, usage)
            return response
        return with_retry(call, attempts=5, base=2, cap=60)

    async def awrap_model_call(self, request, handler):
        for attempt in range(5):
            reservation = self._estimate(request)
            input_estimate = math.ceil((reservation - self.budget.reserve_output) / 3)
            await asyncio.sleep(self.budget.schedule_model(input_estimate))
            self.budget.reserve(reservation)
            try:
                response = await handler(request)
            except BaseException as exc:
                self.budget.finish(reservation, failed=True)
                if not _transient(exc) or attempt == 4:
                    raise
                await asyncio.sleep(min(60, _model_retry_after(exc) or 2 * 2**attempt))
                continue
            known = [m.usage_metadata for m in response.result if getattr(m, "usage_metadata", None)]
            usage = {key: sum(m.get(key, 0) for m in known)
                     for key in ("input_tokens", "output_tokens", "total_tokens")} if known else None
            self.budget.finish(reservation, usage)
            return response
