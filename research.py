"""Run one survey or all preset topics; persist only verified sandbox outputs."""
import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
import threading
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from unittest.mock import patch

from langchain_core.callbacks import BaseCallbackHandler

from agents import (FINALIZER_PATH, NOTES_DIR, REPORT_PATH, SOURCES_PATH,
                    VALIDATOR_PATH, WORKDIR, build_lead_agent, build_research_planner)
from budget import SharedTokenBudget
from check_citations import check
from model import make_model
from sandbox import download, open_sandbox, sandbox_kind, upload
from sandbox import DockerSandbox
from tools import redact

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"
NORMALIZER_SOURCE = ROOT / "normalize_headings.py"
NORMALIZER_PATH = f"{WORKDIR}/research/normalize_headings.py"
FAMILIES = {"arxiv", "hf-daily", "hf-search", "web"}
AUDIT_PATH = f"{WORKDIR}/research/citation-audit.md"
_DOCKER_PREFIX = ["docker"]


def make_lab_model():
    """Configure finite Google waits without modifying the provided factory."""
    model = make_model()
    if os.getenv("LAB_MODEL", "").startswith("google_genai:"):
        model.timeout = 120
        # Google uses 1 for a single attempt. Shared middleware owns retries.
        model.max_retries = 1
    return model


def slugify(topic):
    return re.sub(r"[^\w]+", "-", str(topic).lower(), flags=re.UNICODE).strip("-_")[:60].rstrip("-_") or "topic"


def build_prompt(topic):
    today = datetime.now(timezone(timedelta(hours=7))).date().isoformat()
    return (f"Create a survey of: {topic.strip()}\nResearch date: {today}. Write in English. "
            "Follow the complete research, synthesis, finalization and citation-audit workflow. "
            "At least three independent researcher delegations and three source families are required. "
            f"Final files: {REPORT_PATH} and {SOURCES_PATH}; audit: {AUDIT_PATH}.")


def summarize(messages, elapsed, model_name):
    counts = Counter()
    tokens = {"input": 0, "output": 0}
    for message in messages:
        calls = message.get("tool_calls", []) if isinstance(message, dict) else getattr(message, "tool_calls", [])
        for call in calls or []:
            counts[call.get("name", "unknown")] += 1
        usage = message.get("usage_metadata") if isinstance(message, dict) else getattr(message, "usage_metadata", None)
        if usage:
            tokens["input"] += usage.get("input_tokens", 0)
            tokens["output"] += usage.get("output_tokens", 0)
    return {"model": model_name, "elapsed_s": round(elapsed, 1), "subagent_calls": counts["task"],
            "tool_calls": dict(sorted(counts.items())), "tokens": tokens}


def _validate_sources(sources):
    families = set()
    for entry in sources:
        if not all(key in entry for key in ("n", "id", "url", "title", "date", "source")):
            raise RuntimeError("source metadata missing required fields")
        family = entry["source"]
        if family not in FAMILIES:
            raise RuntimeError("unknown source family")
        prefix = "https://arxiv.org/abs/" if family == "arxiv" else "https://huggingface.co/papers/"
        if family != "web" and entry["url"] != prefix + entry["id"]:
            raise RuntimeError("source URL does not match its family and id")
        families.add(family)
    if len(families) < 3:
        raise RuntimeError("report needs at least three cited source families")
    return sorted(families)


def _validate_report_structure(report):
    headings = re.findall(r"^##[ \t]+(.+?)\s*$", report, re.M)
    required = {"TL;DR", "Background", "Trends and open problems", "References"}
    missing = required - set(headings)
    if missing:
        raise RuntimeError("required report headings missing: " + ", ".join(sorted(missing)))
    themes = [h for h in headings if h not in required]
    if not 3 <= len(themes) <= 6:
        raise RuntimeError("report needs 3-6 thematic sections")


def finalize_in_sandbox(backend):
    # Agents may edit workspace files. Restore trusted checking scripts before
    # the independent final gate instead of trusting agent-modified validators.
    upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                     FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
                     NORMALIZER_PATH: NORMALIZER_SOURCE.read_bytes()})
    for script in (NORMALIZER_PATH, FINALIZER_PATH):
        result = backend.execute(f"python3 {script}")
        if result.exit_code:
            raise RuntimeError("sandbox postprocessing failed: " + result.output[:500])


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS, budget=None, prior_meta=None,
                 audit_fetches=None, worker_model_name=None, audit_reuse=None):
    files = download(backend, [REPORT_PATH, SOURCES_PATH, AUDIT_PATH])
    report, source_bytes = files.get(REPORT_PATH), files.get(SOURCES_PATH)
    if not report or not report.strip() or not source_bytes:
        raise RuntimeError("agent did not produce a nonempty report and sources.json")
    try:
        sources = json.loads(source_bytes)
        decoded = report.decode("utf-8")
        problems = check(decoded, sources)
    except (ValueError, UnicodeError) as exc:
        raise RuntimeError("invalid downloaded report or sources.json") from exc
    if problems:
        raise RuntimeError("invalid report: " + "; ".join(problems[:5]))
    _validate_report_structure(decoded)
    families = _validate_sources(sources)
    meta = {"topic": topic, **summarize(messages, elapsed, model_name),
            "n_sources": len(sources), "source_families": families,
            "sandbox": sandbox_kind(), "tokens_scope": "lead messages only"}
    if worker_model_name:
        meta["subagent_model"] = worker_model_name
    if audit_reuse and prior_meta:
        prior_audit = prior_meta.get("citation_audit_fetches")
        if prior_audit:
            meta["citation_audit_fetches"] = prior_audit
        meta["citation_audit_reuse"] = audit_reuse
    if prior_meta:
        current_summary = summarize(messages, elapsed, model_name)
        if worker_model_name:
            current_summary["subagent_model"] = worker_model_name
        meta["subagent_calls"] += prior_meta.get("subagent_calls", 0)
        meta["tool_calls"] = dict(Counter(meta["tool_calls"]) + Counter(prior_meta.get("tool_calls", {})))
        meta["tokens"] = {key: meta["tokens"][key] + prior_meta.get("tokens", {}).get(key, 0)
                          for key in ("input", "output")}
        meta["elapsed_s"] = round(meta["elapsed_s"] + prior_meta.get("elapsed_s", 0), 1)
        meta["generation_metadata"] = prior_meta.get("generation_metadata", prior_meta)
        meta["repair_runs"] = [*prior_meta.get("repair_runs", []), current_summary]
    if meta["subagent_calls"] < 3:
        raise RuntimeError("fewer than three subagent delegations")
    if not files.get(AUDIT_PATH):
        raise RuntimeError("citation-checker audit is missing")
    if budget:
        meta["shared_token_budget"] = budget.snapshot()
    if audit_fetches is not None:
        meta["citation_audit_fetches"] = {"attempted_urls": sorted(audit_fetches),
            "failed_urls": sorted(url for url, text in audit_fetches.items()
                                  if text.startswith(("ERROR:", "NO RESULTS"))),
            "scope": "web_fetch calls across this run, including researchers"}
    slug = slugify(topic)
    payloads = {f"{slug}.md": report, f"{slug}.sources.json": source_bytes,
                f"{slug}.meta.json": (json.dumps(meta, indent=2, ensure_ascii=False) + "\n").encode(),
                f"{slug}.audit.md": files[AUDIT_PATH]}
    # Validate EVERYTHING before touching reports/. Keep original sandbox bytes.
    destination = Path(reports_dir)
    destination.mkdir(parents=True, exist_ok=True)
    originals = {name: (destination / name).read_bytes() if (destination / name).exists() else None
                 for name in payloads}
    committed = []
    with _staging_directory(destination) as staging:
        for name, content in payloads.items():
            (Path(staging) / name).write_bytes(content)
        try:
            for name in payloads:
                os.replace(Path(staging) / name, destination / name)
                committed.append(name)
        except OSError:
            for name in committed:
                if originals[name] is None:
                    (destination / name).unlink(missing_ok=True)
                else:
                    (destination / name).write_bytes(originals[name])
            raise
    return destination / f"{slug}.md"


@contextmanager
def _staging_directory(destination):
    # Python 3.13+ creates mode=700 directories with private Windows ACLs.
    # Inherit workspace access so outputs remain readable by the shared workspace.
    staging = destination / (".staging-" + uuid.uuid4().hex)
    staging.mkdir(mode=0o755)
    try:
        yield staging
    finally:
        for path in staging.iterdir():
            path.unlink()
        staging.rmdir()


class Progress(BaseCallbackHandler):
    """Log tool names only: never print arguments, URLs containing keys or prompts."""
    def __init__(self):
        self._lock = threading.Lock()
        self._fetch_args = {}
        self._source_calls = {}
        self.source_records = {family: {} for family in ("arxiv", "hf-search", "hf-daily")}
        self.web_urls = set()
        self.web_evidence = []
        self.web_fetch_results = {}

    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"[tool] {serialized.get('name', 'unknown')}", flush=True)
        name = serialized.get("name")
        if name in {"arxiv_search", "hf_search_papers", "hf_daily_papers", "web_search"}:
            with self._lock:
                self._source_calls[kwargs.get("run_id")] = name
        if serialized.get("name") == "web_fetch":
            inputs = kwargs.get("inputs")
            if not isinstance(inputs, dict):
                try:
                    inputs = ast.literal_eval(input_str)
                except (ValueError, SyntaxError):
                    inputs = {}
            if isinstance(inputs, dict) and inputs.get("url"):
                with self._lock:
                    self._fetch_args[kwargs.get("run_id")] = inputs["url"]

    def on_tool_end(self, output, **kwargs):
        with self._lock:
            text = str(getattr(output, "content", output))
            name = self._source_calls.pop(kwargs.get("run_id"), None)
            family = {"arxiv_search": "arxiv", "hf_search_papers": "hf-search",
                      "hf_daily_papers": "hf-daily"}.get(name)
            if family:
                try:
                    records = json.loads(text)
                    for record in records if isinstance(records, list) else []:
                        if isinstance(record, dict) and record.get("url"):
                            self.source_records[family][record["url"]] = record
                except ValueError:
                    pass
            elif name == "web_search" and not text.startswith(("ERROR:", "NO RESULTS")):
                self.web_urls.update(u.rstrip(".,;") for u in re.findall(r'https?://[^\s<>"\)\]]+', text))
                self.web_evidence.append(text)
            url = self._fetch_args.pop(kwargs.get("run_id"), None)
            if url:
                self.web_fetch_results[url] = text
                if not text.startswith(("ERROR:", "NO RESULTS")):
                    self.web_urls.add(url)


def provenance_issues(backend, progress):
    """Require structured source metadata to match actual source-tool results."""
    files = download(backend, [SOURCES_PATH])
    sources = json.loads(files.get(SOURCES_PATH) or b"[]")
    issues = []
    for source in sources:
        family, url = source.get("source"), source.get("url")
        if family == "web":
            if url not in progress.web_urls:
                issues.append(f"web source URL was never retrieved successfully: {url}")
            continue
        original = progress.source_records.get(family, {}).get(url)
        if original is None:
            issues.append(f"{family} source URL absent from actual tool results: {url}")
            continue
        for field, original_field in (("id", "id"), ("title", "title"), ("date", "published")):
            expected = original.get(original_field) or ("n.d." if field == "date" else "")
            if source.get(field) != expected:
                issues.append(f"metadata {field} for {url} must match retrieved value: {expected}")
    return issues


def audit_fetch_issues(backend, progress):
    """Do not accept an audit that names URLs never fetched during this run."""
    files = download(backend, [AUDIT_PATH, SOURCES_PATH])
    if not files.get(AUDIT_PATH):
        return ["citation audit is missing"]
    audit = files[AUDIT_PATH].decode("utf-8")
    sources = json.loads(files.get(SOURCES_PATH) or b"[]")
    urls = {s["url"] for s in sources if isinstance(s, dict) and s.get("url") and s["url"] in audit}
    issues = []
    statuses = re.findall(r"\b(?:SUPPORTED|PARTIAL|UNSUPPORTED|UNVERIFIABLE)\b", audit)
    if len(statuses) < 5 or len(urls) < 3:
        issues.append(f"audit needs at least five explicit claim verdicts spanning at least three source URLs; "
                       f"found {len(statuses)} verdict labels, {len(urls)} cited URLs, and "
                       f"{len(progress.web_fetch_results)} successful/failed web_fetch URLs this run")
    missing = urls - progress.web_fetch_results.keys()
    if missing:
        issues.append("audit URLs not actually fetched with web_fetch this run: " + ", ".join(sorted(missing)))
    return issues


def prepare_docker():
    """Use native Docker or the WSL CLI shim without altering provided sandbox.py."""
    if sandbox_kind() != "docker":
        return
    global _DOCKER_PREFIX
    native = shutil.which("docker")
    if native and not native.lower().endswith((".cmd", ".bat")):
        _DOCKER_PREFIX = [native]
    elif sys.platform == "win32" and shutil.which("wsl"):
        _DOCKER_PREFIX = [shutil.which("wsl"), "--exec", "docker"]
    else:
        raise RuntimeError("Docker CLI not found; install/start Docker or configure its PATH")
    result = subprocess.run([*_DOCKER_PREFIX, "info", "--format", "{{.ServerVersion}}"],
                            capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError("Docker engine unavailable: " + result.stderr.decode(errors="replace")[:300])
    print("[sandbox] Docker engine " + result.stdout.decode(errors="replace").strip(), flush=True)


def _docker_run(command, **kwargs):
    if command and command[0] == "docker":
        command = [*_DOCKER_PREFIX, *command[1:]]
    return subprocess.run(command, **kwargs)


@contextmanager
def docker_bridge():
    """Inject a CLI runner into the provided backend without changing its source."""
    keeper = None
    try:
        if sandbox_kind() == "docker" and "--exec" in _DOCKER_PREFIX:
            # A dockerd service alone does not keep an idle WSL distro alive.
            # Hold a Windows WSL client open while the API/TPM limiter waits.
            keeper = subprocess.Popen([_DOCKER_PREFIX[0], "--exec", "cat"], stdin=subprocess.PIPE,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if sandbox_kind() == "docker":
            with patch("sandbox.DockerSandbox", partial(DockerSandbox, runner=_docker_run)):
                yield
        else:
            yield
    finally:
        if keeper is not None:
            keeper.stdin.close()
            try:
                keeper.wait(timeout=10)
            except subprocess.TimeoutExpired:
                keeper.kill()
                keeper.wait(timeout=10)


def main(topic):
    if not topic.strip():
        print('Usage: python research.py "<topic>" or --all', file=sys.stderr)
        return 2
    started = time.monotonic()
    try:
        prepare_docker()
        model = make_lab_model()
        budget = SharedTokenBudget(int(os.getenv("LAB_TOKEN_BUDGET", "1500000")))
        with docker_bridge(), open_sandbox() as backend:
            response = backend.execute(f"mkdir -p {NOTES_DIR} {WORKDIR}/report")
            if response.exit_code != 0:
                raise RuntimeError("cannot initialize sandbox workspace")
            upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                             FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
                             NORMALIZER_PATH: NORMALIZER_SOURCE.read_bytes()})
            # Provided upload() does not inspect errors; check the seeded files explicitly.
            seeded = backend.execute(f"test -s {VALIDATOR_PATH} && test -s {FINALIZER_PATH}")
            if seeded.exit_code != 0:
                raise RuntimeError("sandbox script upload failed")
            planner = build_research_planner(backend, model, budget)
            progress = Progress()
            result = planner.invoke({"messages": [{"role": "user", "content": build_prompt(topic) +
                " FIRST PHASE ONLY: plan and delegate at least three independent researchers, "
                "obtain notes and actual sources across at least three families, read their notes, "
                "then STOP and return the notes paths. Do not draft the report, sources.json or audit yet. "
                "Never modify the uploaded validator/finalizer scripts."}]},
                                  config={"recursion_limit": 1000, "callbacks": [progress]})
            ledger_path = f"{WORKDIR}/research/retrieved-source-ledger.json"
            evidence_path = f"{WORKDIR}/research/retrieved-web-evidence.txt"
            upload(backend, {ledger_path: json.dumps(progress.source_records, ensure_ascii=False).encode(),
                             evidence_path: "\n\n".join(progress.web_evidence).encode()})
            agent = build_lead_agent(backend, model, budget)
            result = agent.invoke({"messages": [*result.get("messages", []), {"role": "user", "content":
                f"SECOND PHASE: complete the survey for {topic}. Read ALL researcher notes, "
                f"{ledger_path} and {evidence_path}. The ledger contains ACTUAL retrieved records; "
                "copy exact IDs, URLs, titles and published dates, never invent or paraphrase metadata. "
                "If missing evidence, delegate targeted replacement research; never fabricate sources. "
                "Write a 1200-1800 word report with 3-6 themes, source metadata, finalize, validate and "
                "delegate citation-checker to actually fetch five claims spanning three distinct URLs. "
                "Audit entries must include current citation number, verbatim claim, exact URL and verdict. "
                "Never edit the validator or finalizer. Complete the full remaining workflow."}]},
                config={"recursion_limit": 1000, "callbacks": [progress]})
            # Return failed gates to the same lead, with bounded repair attempts.
            # Repairs/finalization still happen inside the sandbox, before download.
            for repair in range(3):
                issues = []
                try:
                    finalize_in_sandbox(backend)
                except RuntimeError as exc:
                    issues.append(str(exc))
                validated = backend.execute(f"python3 {VALIDATOR_PATH}")
                if validated.exit_code != 0 or not validated.output.startswith("OK:"):
                    issues.append("sandbox validator: " + validated.output[:1500])
                try:
                    candidates = download(backend, [SOURCES_PATH, REPORT_PATH])
                    _validate_sources(json.loads(candidates.get(SOURCES_PATH) or b"null"))
                    _validate_report_structure((candidates.get(REPORT_PATH) or b"").decode("utf-8"))
                except (ValueError, TypeError, RuntimeError, KeyError) as exc:
                    issues.append(str(exc))
                if not issues:
                    issues.extend(provenance_issues(backend, progress))
                if not issues:
                    issues.extend(audit_fetch_issues(backend, progress))
                if not issues:
                    break
                if repair == 2:
                    raise RuntimeError("final quality gate failed: " + "; ".join(issues))
                print("[quality] requesting corrections inside sandbox", flush=True)
                ledger_path = f"{WORKDIR}/research/retrieved-source-ledger.json"
                upload(backend, {ledger_path: json.dumps(progress.source_records, ensure_ascii=False).encode()})
                messages = [*result.get("messages", []), {"role": "user", "content":
                    "Final quality gate found: " + "; ".join(issues) +
                    f". Read {ledger_path}: it contains ACTUAL tool results keyed by family and URL. "
                    "Use only those exact IDs, URLs, titles and published dates for arxiv/hf sources; "
                    "replace invented sources and their claims with grounded evidence. "
                    ". Fix the metadata/report in the sandbox using the retrieved notes. "
                    "A source retrieved by hf-search/hf-daily must retain its Hugging Face URL. "
                    "Do not relabel provenance to hide an error. Rerun finalizer and validator, "
                    "then delegate citation-checker to actually fetch every audit URL and refresh the audit. "
                    "Report failed fetches as UNVERIFIABLE rather than claiming support."}]
                result = agent.invoke({"messages": messages}, config={"recursion_limit": 1000, "callbacks": [progress]})
            path = save_outputs(backend, topic, result.get("messages", []),
                                time.monotonic() - started, os.getenv("LAB_MODEL") or os.getenv("OPENAI_DEPLOYMENT_MODEL"),
                                budget=budget, audit_fetches=progress.web_fetch_results)
        print(f"Saved {path}", flush=True)
        return 0
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {redact(exc)}", file=sys.stderr)
        if "budget" in locals():
            print("[budget] " + json.dumps(budget.snapshot()), file=sys.stderr)
        return 1


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("topic", nargs="*")
    parser.add_argument("--all", action="store_true", help="run all five preset topics sequentially")
    parser.add_argument("--skip-existing", action="store_true", help="skip already validated preset outputs")
    args = parser.parse_args(argv)
    if args.all and args.topic:
        parser.error("choose a topic or --all")
    topics = re.findall(r"^\d+\. (.+)$", (ROOT / "topics.md").read_text(encoding="utf-8"), re.M) if args.all else [" ".join(args.topic)]
    status = 0
    for topic in topics:
        if args.skip_existing and _existing_valid(topic):
            print(f"Skip validated report: {topic}")
            continue
        print(f"Researching: {topic}", flush=True)
        code = main(topic)
        status = max(status, code)
        # Stop on a failure: do not repeatedly spend tokens on a broken setup.
        if code:
            break
    return status


def _existing_valid(topic):
    slug = slugify(topic)
    try:
        report = (REPORTS / f"{slug}.md").read_text(encoding="utf-8")
        sources = json.loads((REPORTS / f"{slug}.sources.json").read_bytes())
        meta = json.loads((REPORTS / f"{slug}.meta.json").read_bytes())
        _validate_report_structure(report)
        return (not check(report, sources) and bool(_validate_sources(sources))
                and meta.get("topic") == topic and meta.get("subagent_calls", 0) >= 3
                and (REPORTS / f"{slug}.audit.md").is_file())
    except (OSError, ValueError, RuntimeError, TypeError):
        return False


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(cli())
