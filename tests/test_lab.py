"""Offline regression tests for retry, malformed citations, budgets and persistence."""
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace

import httpx
from langchain_core.messages import AIMessage

import research
import tools
from budget import SharedTokenBudget, SharedTokenBudgetMiddleware, TokenBudgetExceeded, _model_retry_after, _transient
from check_citations import check
from finalize_citations import finalize
from normalize_headings import normalize


class RetryTests(unittest.TestCase):
    @patch("tools.time.sleep")
    @patch("tools.random.uniform", return_value=0.5)
    def test_backoff_cap_and_no_final_sleep(self, jitter, sleep):
        fn = Mock(side_effect=tools.RetryableError("busy"))
        with self.assertRaises(tools.RetryableError):
            tools.with_retry(fn, attempts=4, base=1, cap=3)
        self.assertEqual(fn.call_count, 4)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [1.5, 2.5, 3])

    @patch("tools.time.sleep")
    def test_retry_after_and_nonretryable(self, sleep):
        fn = Mock(side_effect=[tools.RetryableError("busy", 7), "ok"])
        self.assertEqual(tools.with_retry(fn), "ok")
        sleep.assert_called_once_with(7)
        bad = Mock(side_effect=ValueError("bad input"))
        with self.assertRaises(ValueError):
            tools.with_retry(bad)
        self.assertEqual(bad.call_count, 1)

    @patch("tools.httpx.request")
    def test_http_classification(self, request):
        request.return_value = httpx.Response(429, headers={"Retry-After": "4"}, request=httpx.Request("GET", "https://x.org"))
        with self.assertRaises(tools.RetryableError) as caught:
            tools._request("GET", "https://x.org")
        self.assertEqual(caught.exception.retry_after, 4)
        request.return_value = httpx.Response(401, request=httpx.Request("GET", "https://x.org"))
        with self.assertRaises(httpx.HTTPStatusError):
            tools._request("GET", "https://x.org")
        request.side_effect = httpx.ConnectError("offline")
        with self.assertRaises(tools.RetryableError):
            tools._request("GET", "https://x.org")


class ToolTests(unittest.TestCase):
    @patch("tools._request")
    def test_empty_arxiv_skips_network(self, request):
        self.assertEqual(tools.arxiv_search.invoke({"query": ':" AND OR'}), "NO RESULTS")
        request.assert_not_called()

    @patch("tools.time.sleep")
    @patch("tools.time.monotonic", side_effect=[10, 11, 13])
    @patch("tools._request")
    def test_arxiv_spacing_and_normalization(self, request, monotonic, sleep):
        request.return_value = Mock(text='<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2501.12345v2</id><title>A\n B</title><summary>Some   text</summary><published>2025-01-02T00:00:00Z</published></entry></feed>')
        with patch.object(tools, "_arxiv_last", None):
            a = json.loads(tools.arxiv_search.invoke({"query": 'all:"world" AND model'}))
            tools.arxiv_search.invoke({"query": "world model"})
        self.assertEqual(a[0]["id"], "2501.12345")
        self.assertEqual(a[0]["title"], "A B")
        self.assertEqual(request.call_args.kwargs["params"]["search_query"], "all:world AND all:model")
        sleep.assert_called_once_with(2)

    def test_hf_prefers_ai_and_skips_missing_id(self):
        records = tools._hf_records([{"paper": {"id": "x", "summary": "raw", "ai_summary": "ai"}}, {"paper": {}}], True)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["summary"], "ai")

    @patch("tools.time.sleep")
    @patch("tools._request")
    def test_exa_http200_rate_limit_and_sse(self, request, sleep):
        def response(result):
            return Mock(text="data: " + json.dumps({"jsonrpc": "2.0", "id": 1, "result": result}) + "\n\n")
        request.side_effect = [response({"_meta": {"rateLimited": True}, "content": []}),
                               response({"content": [{"type": "text", "text": "evidence"}]})]
        self.assertEqual(tools._exa_call("web_fetch_exa", {"urls": ["https://x.org"]}), "evidence")
        self.assertEqual(request.call_count, 2)
        sleep.assert_called_once()

    def test_redact_raw_and_encoded_keys(self):
        secret = "example/private+credential"
        with patch.dict(os.environ, {"EXA_API_KEY": secret}):
            masked = tools.redact(secret + " " + tools.quote(secret, safe="") + " https://x.org?exaApiKey=hidden")
        self.assertNotIn(secret, masked)
        self.assertNotIn("exaApiKey=hidden", masked)

    def test_rate_limit_metadata_false_and_nested_flags(self):
        self.assertFalse(tools._rate_limit_meta({"rate_limit_exceeded": False}))
        self.assertFalse(tools._rate_limit_meta({"rateLimited": "false", "rateLimit": {"limit": 100}}))
        self.assertTrue(tools._rate_limit_meta({"exa": {"rateLimited": True}}))
        self.assertTrue(tools._rate_limit_meta({"errorCode": "RATE_LIMIT_EXCEEDED"}))


class CitationTests(unittest.TestCase):
    def setUp(self):
        self.sources = [{"n": n, "url": f"https://example.org/{n}"} for n in range(1, 4)]
        self.refs = "\n## References\n" + "\n".join(f"[{n}] Title. https://example.org/{n}" for n in range(1, 4))

    def test_grouped_and_adjacent_citations(self):
        for body in ("Claim [1][2][3].", "Claim [1, 2, 3].", "Claim [1-3]."):
            self.assertEqual(check(body + self.refs, self.sources), [])

    def test_code_and_markdown_links_are_not_citations(self):
        problems = check("Claim [1]. `[2]`\n```\n[3]\n```\n[2](https://example.org/2)" + self.refs, self.sources)
        self.assertIn("source [2] never cited", problems)
        self.assertIn("source [3] never cited", problems)

    def test_invalid_sources_and_references(self):
        for sources in ([], {}, [None], [{"n": True, "url": "bad"}], self.sources + [self.sources[0]]):
            self.assertTrue(check("[1][2][3]" + self.refs, sources))
        self.assertTrue(check("[9]" + self.refs, self.sources))
        self.assertTrue(check("[1][2][3]" + self.refs + "\n[1] extra https://example.org/1", self.sources))
        self.assertTrue(check("[1][2][3]" + self.refs.replace("https://example.org/1", "https://wrong.org/1"), self.sources))
        self.assertTrue(check("[1][2][3]" + self.refs.replace("Title.", "Title. https://extra.org", 1), self.sources))

    def test_finalizer_roundtrip(self):
        report, sources, problems = finalize("Claim [3, 1].", self.sources)
        self.assertFalse(problems)
        self.assertEqual(check(report, sources), [])
        self.assertEqual(finalize(report, sources)[:2], (report, sources))

    def test_source_family_mismatch_and_parenthesized_url(self):
        sources = [{"n": 1, "id": "2501.12345", "url": "https://arxiv.org/abs/2501.12345", "source": "hf-search"}]
        report = "Claim [1].\n## References\n[1] Paper. https://arxiv.org/abs/2501.12345"
        self.assertTrue(any("family" in p for p in check(report, sources)))
        sources = [{"n": 1, "url": "https://example.org/a_(b)"}]
        self.assertEqual(check("Claim [1].\n## References\n[1] Title. https://example.org/a_(b)", sources), [])

    def test_heading_normalization_is_idempotent_and_preserves_code(self):
        text = "## Trends, Open Problems, and Limitations\nOriginal prose.\n```md\n## Trends and Open Problems\n```\n"
        new = normalize(text)
        self.assertTrue(new.startswith("## Trends and open problems\nOriginal prose."))
        self.assertIn("```md\n## Trends and Open Problems\n```", new)
        self.assertEqual(normalize(new), new)


class PersistenceTests(unittest.TestCase):
    @patch("research.upload")
    def test_final_gate_restores_trusted_validator(self, upload):
        backend = Mock()
        backend.execute.return_value = SimpleNamespace(exit_code=0, output="OK")
        research.finalize_in_sandbox(backend)
        restored = upload.call_args.args[1]
        self.assertEqual(restored[research.VALIDATOR_PATH], research.VALIDATOR_SOURCE.read_bytes())
        self.assertEqual(restored[research.FINALIZER_PATH], research.FINALIZER_SOURCE.read_bytes())

    @patch("research.download")
    def test_provenance_rejects_invented_or_changed_source_records(self, download):
        progress = research.Progress()
        record = {"id": "2401.00001", "url": "https://huggingface.co/papers/2401.00001",
                  "title": "Retrieved paper", "published": "2024-01-02"}
        with patch("builtins.print"):
            progress.on_tool_start({"name": "hf_search_papers"}, "{}", run_id="source")
        progress.on_tool_end(json.dumps([record]), run_id="source")
        source = {"n": 1, "id": record["id"], "url": record["url"], "title": record["title"],
                  "date": record["published"], "source": "hf-search"}
        download.return_value = {research.SOURCES_PATH: json.dumps([source]).encode()}
        self.assertFalse(research.provenance_issues(None, progress))
        source["title"] = "Invented title"
        download.return_value = {research.SOURCES_PATH: json.dumps([source]).encode()}
        self.assertTrue(research.provenance_issues(None, progress))
        source["url"] = "https://huggingface.co/papers/2502.56789"
        download.return_value = {research.SOURCES_PATH: json.dumps([source]).encode()}
        self.assertTrue(research.provenance_issues(None, progress))

    @patch("research.make_model")
    def test_google_network_wait_is_bounded_without_nested_retries(self, factory):
        model = SimpleNamespace(timeout=None, max_retries=6)
        factory.return_value = model
        with patch.dict(os.environ, {"LAB_MODEL": "google_genai:test"}):
            self.assertIs(research.make_lab_model(), model)
        self.assertEqual(model.timeout, 120)
        self.assertEqual(model.max_retries, 1)
        other = SimpleNamespace(timeout=10, max_retries=2)
        factory.return_value = other
        with patch.dict(os.environ, {"LAB_MODEL": "other:test"}):
            research.make_lab_model()
        self.assertEqual((other.timeout, other.max_retries), (10, 2))

    @patch("research.download")
    def test_audit_requires_fetches_not_just_claimed_verdicts(self, download):
        progress = research.Progress()
        urls = [f"https://example.org/{n}" for n in range(3)]
        audit = "SUPPORTED\n" * 5 + "\n".join(urls)
        download.return_value = {research.AUDIT_PATH: audit.encode(),
                                research.SOURCES_PATH: json.dumps([{"url": u} for u in urls]).encode()}
        self.assertTrue(research.audit_fetch_issues(None, progress))
        for url in urls:
            with patch("builtins.print"), patch("tools._exa_call", return_value="Retrieved evidence"):
                tools.web_fetch.invoke({"url": url}, config={"callbacks": [progress]})
        self.assertFalse(research.audit_fetch_issues(None, progress))

    def test_slug_and_accounting(self):
        for topic in ("../../x", "", "a" * 200, "../\\..//"):
            slug = research.slugify(topic)
            self.assertTrue(slug)
            self.assertLessEqual(len(slug), 60)
            self.assertNotIn("/", slug)
            self.assertNotIn("\\", slug)
        message = AIMessage(content="", tool_calls=[{"name": "task", "args": {}, "id": "1"}],
                            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        summary = research.summarize([message], 1.26, "model")
        self.assertEqual(summary["subagent_calls"], 1)
        self.assertEqual(summary["tokens"], {"input": 10, "output": 5})

    @patch("research.download")
    def test_invalid_download_writes_nothing(self, download):
        for files in ({}, {research.REPORT_PATH: b"report", research.SOURCES_PATH: b"bad"},
                      {research.REPORT_PATH: b"", research.SOURCES_PATH: b"[]"}):
            download.return_value = files
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(RuntimeError):
                    research.save_outputs(None, "topic", [], 0, "model", reports_dir=tmp)
                self.assertEqual(list(Path(tmp).iterdir()), [])

    @patch("research.download")
    def test_preserve_original_bytes_and_rollback(self, download):
        sources = [{"n": 1, "id": "1", "url": "https://arxiv.org/abs/1", "source": "arxiv", "title": "A", "date": "n.d."},
                   {"n": 2, "id": "2", "url": "https://huggingface.co/papers/2", "source": "hf-search", "title": "B", "date": "n.d."},
                   {"n": 3, "id": "3", "url": "https://example.org/3", "source": "web", "title": "C", "date": "n.d."}]
        report = ("# Survey\n## TL;DR\nClaim [1][2][3].\n## Background\nIntro.\n"
                  "## Theme A\nA.\n## Theme B\nB.\n## Theme C\nC.\n## Trends and open problems\nGaps.\n\n## References\n"
                  + "\n".join(f"[{s['n']}] {s['title']}. {s['url']}" for s in sources)).encode()
        source_bytes = json.dumps(sources, indent=4).encode()
        download.return_value = {research.REPORT_PATH: report, research.SOURCES_PATH: source_bytes, research.AUDIT_PATH: b"audit"}
        calls = [AIMessage(content="", tool_calls=[{"name": "task", "args": {}, "id": str(n)} for n in range(3)])]
        with tempfile.TemporaryDirectory() as tmp:
            path = research.save_outputs(None, "topic", calls, 1, "model", reports_dir=tmp)
            self.assertEqual(path.read_bytes(), report)
            self.assertEqual(path.with_suffix(".sources.json").read_bytes(), source_bytes)
            original_replace = os.replace
            def replace(src, dst):
                if str(dst).endswith(".meta.json"):
                    raise OSError("disk failure")
                return original_replace(src, dst)
            download.return_value[research.REPORT_PATH] = report.replace(b"Claim", b"Updated claim")
            with patch("research.os.replace", side_effect=replace):
                with self.assertRaises(OSError):
                    research.save_outputs(None, "topic", calls, 1, "model", reports_dir=tmp)
            self.assertEqual(path.read_bytes(), report)
            self.assertFalse(any(p.is_dir() for p in Path(tmp).iterdir()))

    def test_report_structure_requires_template_and_themes(self):
        with self.assertRaises(RuntimeError):
            research._validate_report_structure("## TL;DR\n## Background\n## References")
        with self.assertRaises(RuntimeError):
            research._validate_report_structure("## TL;DR\n## Background\n## Trends and open problems\n## References")
        research._validate_report_structure("## TL;DR\n## Background\n## A\n## B\n## C\n## Trends and open problems\n## References")


class BudgetTests(unittest.TestCase):
    def test_parallel_reservations_and_accounting(self):
        budget = SharedTokenBudget(100)
        def reserve(_):
            try:
                budget.reserve(30)
                return True
            except TokenBudgetExceeded:
                return False
        with ThreadPoolExecutor(max_workers=8) as pool:
            accepted = list(pool.map(reserve, range(8)))
        self.assertEqual(sum(accepted), 3)
        budget.finish(30, {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10})
        self.assertEqual(budget.snapshot()["used_or_estimated"], 10)
        self.assertEqual(budget.reserved, 60)

    @patch("budget.time.monotonic", return_value=10)
    def test_model_slots_are_shared(self, clock):
        budget = SharedTokenBudget(100, model_interval=5)
        self.assertEqual([budget.schedule_model() for _ in range(3)], [0, 5, 10])

    @patch("budget.time.monotonic", return_value=10)
    def test_model_token_throughput_waits_for_window(self, clock):
        budget = SharedTokenBudget(1000, model_interval=5)
        budget.model_tpm = 100
        self.assertEqual(budget.schedule_model(45), 0)
        self.assertEqual(budget.schedule_model(45), 5)
        self.assertAlmostEqual(budget.schedule_model(45), 60.1)

    @patch("tools.time.sleep")
    @patch("budget.time.sleep")
    def test_transient_model_retry_and_failure_accounting(self, throttle_sleep, retry_sleep):
        budget = SharedTokenBudget(100_000, model_interval=0)
        middleware = SharedTokenBudgetMiddleware(budget)
        request = SimpleNamespace(system_message=None, messages=[], tools=[])
        response = SimpleNamespace(result=[AIMessage(content="ok", usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7})])
        handler = Mock(side_effect=[httpx.ReadError("connection aborted"), response])
        self.assertIs(middleware.wrap_model_call(request, handler), response)
        self.assertEqual(handler.call_count, 2)
        self.assertEqual(budget.reserved, 0)
        self.assertEqual(budget.snapshot()["estimated_calls"], 1)
        bad = Mock(side_effect=ValueError("bad request"))
        with self.assertRaises(ValueError):
            middleware.wrap_model_call(request, bad)
        self.assertEqual(bad.call_count, 1)

    def test_google_rate_limit_and_retry_delay(self):
        from langchain_google_genai.chat_models import GoogleRateLimitError, GoogleAuthenticationError
        error = GoogleRateLimitError("429 RESOURCE_EXHAUSTED. Please retry in 32.75s.")
        self.assertTrue(_transient(error))
        self.assertEqual(_model_retry_after(error), 33.75)
        self.assertFalse(_transient(GoogleAuthenticationError("401")))
        daily = GoogleRateLimitError("429 quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier")
        self.assertFalse(_transient(daily))


if __name__ == "__main__":
    unittest.main()
