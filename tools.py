"""Host-side source tools. No credentials or network clients enter the sandbox."""
import json
import os
import random
import re
import threading
import time
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote, quote_plus, urlparse

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()
ARXIV_URL = "https://export.arxiv.org/api/query"
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"
_arxiv_lock = threading.Lock()
_arxiv_last = None


class RetryableError(Exception):
    """Transient failure with optional Retry-After in seconds."""
    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def redact(text):
    """Mask configured secrets and URL-encoded variants in diagnostics."""
    text = str(text)
    for name, value in os.environ.items():
        if value and name.upper().endswith(("KEY", "TOKEN", "SECRET", "PASSWORD")):
            for encoded in {value, quote(value, safe=""), quote_plus(value)}:
                text = text.replace(encoded, "[REDACTED]")
    return re.sub(r"(?i)(exaApiKey=)[^&\s\"']+", r"\1[REDACTED]", text)


def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Retry only RetryableError; cap waits, add jitter, never sleep after final failure."""
    if attempts < 1 or base < 0 or cap < 0:
        raise ValueError("invalid retry settings")
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            delay = (float(exc.retry_after) if exc.retry_after is not None
                     else base * 2**attempt + random.uniform(0, base))
            time.sleep(max(0, min(cap, delay)))


def _retry_after(value):
    if not value:
        return None
    try:
        return max(0, float(value))
    except (TypeError, ValueError):
        try:
            return max(0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _request(method, url, **kwargs):
    try:
        response = httpx.request(method, url, timeout=45, follow_redirects=True, **kwargs)
    except httpx.TransportError as exc:
        raise RetryableError(redact(exc)) from None
    if response.status_code in {429, 500, 502, 503, 504}:
        raise RetryableError(f"HTTP {response.status_code}", _retry_after(response.headers.get("Retry-After")))
    response.raise_for_status()
    return response


def _error(exc):
    return f"ERROR: {type(exc).__name__}: {redact(exc)}"


def _clean(value, limit=None):
    text = " ".join(str(value or "").split())
    return text[:limit] if limit else text


def _records(records):
    return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv by a few keywords, newest first. Returns JSON {id,url,published,title,summary}."""
    try:
        terms = [t for t in re.findall(r"[^\W_]+(?:-[^\W_]+)*", query)
                 if t.upper() not in {"AND", "OR", "NOT", "ALL", "TI", "AU", "ABS"}]
        if not terms:
            return "NO RESULTS"

        def call():
            global _arxiv_last
            with _arxiv_lock:
                if _arxiv_last is not None:
                    time.sleep(max(0, 3 - (time.monotonic() - _arxiv_last)))
                _arxiv_last = time.monotonic()
            # Serialize request starts, not slow network responses. A stalled
            # request must not hold every independent researcher behind it.
            return _request("GET", ARXIV_URL, params={
                "search_query": " AND ".join(f"all:{t}" for t in terms),
                "sortBy": "submittedDate", "sortOrder": "descending",
                "max_results": max(1, min(30, max_results)), "start": 0})

        root = ET.fromstring(with_retry(call, attempts=7, cap=60).text)
        records = []
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            def field(name):
                return entry.findtext("{http://www.w3.org/2005/Atom}" + name, "")
            identifier = re.sub(r"v\d+$", "", field("id").split("/abs/")[-1])
            if not identifier or identifier.endswith("errors"):
                continue
            records.append({"id": identifier, "url": f"https://arxiv.org/abs/{identifier}",
                            "published": field("published")[:10], "title": _clean(field("title")),
                            "summary": _clean(field("summary"), 600)})
        return _records(records)
    except Exception as exc:
        return _error(exc)


def _hf_records(items, prefer_ai=False):
    if not isinstance(items, list):
        raise ValueError("expected Hugging Face result list")
    records = []
    for item in items:
        paper = item.get("paper") or {}
        if not paper.get("id"):
            continue
        summary = (paper.get("ai_summary") or item.get("ai_summary")) if prefer_ai else None
        records.append({"id": paper["id"], "url": f"https://huggingface.co/papers/{paper['id']}",
                        "published": str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
                        "title": _clean(paper.get("title") or item.get("title")),
                        "summary": _clean(summary or paper.get("summary") or item.get("summary"), 600),
                        "upvotes": paper.get("upvotes") or item.get("upvotes") or 0,
                        "github": paper.get("githubRepo") or item.get("githubRepo") or "",
                        "stars": paper.get("githubStars") or item.get("githubStars") or 0})
    return records


@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Get trending Hugging Face papers sorted by upvotes. Optional YYYY-MM-DD date and client-side keyword filter.
    Returns JSON {id,url,published,title,summary,upvotes,github,stars}. Use hf_search_papers for topic search."""
    try:
        params = {"limit": max(1, min(100, limit))}
        if date:
            datetime.strptime(date, "%Y-%m-%d")
            params["date"] = date
        items = with_retry(lambda: _request("GET", HF_DAILY_URL, params=params)).json()
        records = _hf_records(items)
        if keyword.strip():
            records = [r for r in records if keyword.casefold() in (r["title"] + " " + r["summary"]).casefold()]
        return _records(sorted(records, key=lambda r: r["upvotes"], reverse=True))
    except Exception as exc:
        return _error(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns JSON {id,url,published,title,summary,upvotes,github,stars}."""
    try:
        if not query.strip():
            return "NO RESULTS"
        items = with_retry(lambda: _request("GET", HF_SEARCH_URL,
                                         params={"q": query, "limit": max(1, min(50, limit))})).json()
        return _records(_hf_records(items, prefer_ai=True))
    except Exception as exc:
        return _error(exc)


def _rpc_response(response):
    """Decode JSON or multiline server-sent JSON-RPC events."""
    if response.text.lstrip().startswith("{"):
        return response.json()
    chunks, data = [], []
    for line in response.text.splitlines() + [""]:
        if line.startswith("data:"):
            data.append(line[5:].lstrip())
        elif not line and data:
            chunks.append(json.loads("\n".join(data)))
            data = []
    for chunk in reversed(chunks):
        if chunk.get("id") == 1 and ("result" in chunk or "error" in chunk):
            return chunk
    raise ValueError("no JSON-RPC response in Exa stream")


def _rate_limit_meta(metadata):
    if not isinstance(metadata, dict):
        return False
    for key, value in metadata.items():
        flag = re.sub(r"[^a-z]", "", str(key).lower())
        if flag.endswith(("ratelimited", "ratelimitexceeded", "ratelimitreached", "toomanyrequests")):
            if value is True or str(value).lower() in {"true", "1", "yes"}:
                return True
        if isinstance(value, dict) and _rate_limit_meta(value):
            return True
        if isinstance(value, str) and re.search(r"rate.?limit(?:ed|_exceeded| exceeded| reached)|too many requests", value, re.I):
            return True
    return False


def _exa_call(name, arguments):
    def call():
        key = os.getenv("EXA_API_KEY", "").strip()
        response = _request("POST", EXA_URL, params={"exaApiKey": key} if key else {},
                            headers={"Accept": "application/json, text/event-stream"},
                            json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                  "params": {"name": name, "arguments": arguments}})
        payload = _rpc_response(response)
        if "error" in payload:
            raise ValueError(redact(json.dumps(payload["error"])))
        result = payload.get("result", {})
        text = "\n".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
        metadata = result.get("_meta") or {}
        limited = (_rate_limit_meta(metadata)
                   or ((len(text) < 1500 or result.get("isError"))
                       and bool(re.search(r"rate.?limit(?:ed| exceeded| reached)|too many requests", text, re.I))))
        if limited:
            raise RetryableError("Exa rate limit", _retry_after(metadata.get("retryAfter") or metadata.get("retry_after")))
        if result.get("isError"):
            raise ValueError(redact(text or "Exa tool failed"))
        return redact(text.strip()) or "NO RESULTS"
    return with_retry(call, attempts=7, cap=60)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search Exa for web pages and papers. Returns retrieved text with URLs; objective describes desired evidence."""
    try:
        if not query.strip():
            return "NO RESULTS"
        return _exa_call("web_search_exa", {"query": query,
                         "objective": objective or f"Find reliable sources about {query}",
                         "numResults": max(1, min(10, num_results))})[:18000]
    except Exception as exc:
        return _error(exc)


@tool
def web_fetch(url: str) -> str:
    """Fetch one public HTTP(S) page through Exa. Returns retrieved text, truncated to 12000 characters."""
    try:
        if urlparse(url).scheme not in {"http", "https"} or not urlparse(url).netloc:
            raise ValueError("a complete HTTP(S) URL is required")
        return _exa_call("web_fetch_exa", {"urls": [url], "maxCharacters": 12000})[:12000]
    except Exception as exc:
        return _error(exc)


SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    failed = False
    for fn, args in [(arxiv_search, {"query": "world model", "max_results": 3}),
                     (hf_daily_papers, {"limit": 20}),
                     (hf_search_papers, {"query": "world model", "limit": 3}),
                     (web_search, {"query": "survey paper on world models", "num_results": 2}),
                     (web_fetch, {"url": "https://arxiv.org/abs/1803.10122"})]:
        result = fn.invoke(args)
        print(f"== {fn.name}\n{result[:500]}\n")
        failed |= result.startswith("ERROR:")
    raise SystemExit(1 if failed else 0)
