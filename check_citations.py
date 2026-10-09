"""Standard-library citation validator; uploaded and executed inside the sandbox."""
import json
import re
import sys
from collections import Counter
from urllib.parse import urlparse

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"
_CODE = re.compile(r"```.*?```|~~~.*?~~~|`+[^`\n]*`+", re.S)
_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\s*\()")
_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*\r?$")


def _numbers(group):
    for part in re.split(r"\s*,\s*", group):
        ends = re.split(r"\s*[–-]\s*", part)
        if len(ends) == 1:
            yield int(ends[0])
        else:
            a, b = map(int, ends)
            if not 0 <= b - a <= 200:
                raise ValueError("invalid or excessively large citation range")
            yield from range(a, b + 1)


def check(report_text, sources):
    """Validate identities, body citations, and exactly one matching URL per reference."""
    problems = []
    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json (expected a nonempty list)"]
    by_n, urls = {}, set()
    for index, entry in enumerate(sources):
        if not isinstance(entry, dict):
            problems.append(f"source {index}: expected an object")
            continue
        n, url = entry.get("n"), entry.get("url")
        if type(n) is not int or n < 1:
            problems.append(f"source {index}: n must be a positive integer")
        elif n in by_n:
            problems.append(f"duplicate source number [{n}]")
        else:
            by_n[n] = entry
        if not isinstance(url, str) or urlparse(url).scheme not in {"http", "https"} or not urlparse(url).netloc:
            problems.append(f"source {index}: invalid HTTP(S) url")
        elif url in urls:
            problems.append(f"duplicate source URL: {url}")
        else:
            urls.add(url)
        family = entry.get("source")
        if family is not None:
            if family not in {"arxiv", "hf-daily", "hf-search", "web"}:
                problems.append(f"source [{n}] has unknown source family")
            elif family != "web":
                prefix = "https://arxiv.org/abs/" if family == "arxiv" else "https://huggingface.co/papers/"
                if not entry.get("id") or url != prefix + str(entry.get("id")):
                    problems.append(f"source [{n}] URL does not match {family} family and id")
    visible = _CODE.sub("", report_text)
    headings = list(_HEADING.finditer(visible))
    if not headings:
        return problems + ["missing ## References"]
    if len(headings) != 1:
        problems.append("expected exactly one ## References heading")
    heading = headings[0]
    body, references = visible[:heading.start()], visible[heading.end():]
    cited = set()
    for match in _GROUP.finditer(body):
        try:
            cited.update(_numbers(match.group(1)))
        except ValueError as exc:
            problems.append(str(exc))
    for n in sorted(cited - by_n.keys()):
        problems.append(f"[{n}] cited but missing from sources.json")
    for n in sorted(by_n.keys() - cited):
        problems.append(f"source [{n}] never cited")
    counts = Counter()
    for line in references.splitlines():
        match = re.match(r"^\[(\d+)\]\s*(.*)$", line.strip())
        if not match:
            if line.strip():
                problems.append("reference line must start with [n]: " + line[:100])
            continue
        n = int(match.group(1))
        counts[n] += 1
        if n not in by_n:
            problems.append(f"reference [{n}] missing from sources.json")
        found = re.findall(r"https?://[^\s<>]+", match.group(2))
        # The finalizer writes plain URLs; preserve parentheses and punctuation
        # that can legitimately belong to a URL instead of silently changing it.
        if len(found) != 1:
            problems.append(f"reference [{n}] must contain exactly one URL")
        elif n in by_n and found[0] != by_n[n].get("url"):
            problems.append(f"reference [{n}] URL differs from sources.json")
    for n in sorted(by_n):
        if counts[n] != 1:
            problems.append(f"source [{n}] needs exactly one reference line (found {counts[n]})")
    return problems


def main(argv):
    try:
        with open(argv[1] if len(argv) > 1 else REPORT, encoding="utf-8") as f:
            report = f.read()
        with open(argv[2] if len(argv) > 2 else SOURCES, encoding="utf-8") as f:
            sources = json.load(f)
        problems = check(report, sources)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
