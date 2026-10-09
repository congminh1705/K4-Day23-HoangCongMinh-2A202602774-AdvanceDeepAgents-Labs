"""Canonicalize template headings INSIDE the sandbox; leave all prose unchanged."""
import re
import sys
from pathlib import Path

REPORT = "/tmp/work/report/report.md"


def normalize(text):
    canonical = {h.casefold(): h for h in ("TL;DR", "Background", "Trends and open problems", "References")}
    lines = []
    fence = None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            symbol = marker.group(1)[0]
            fence = None if fence == symbol else symbol if fence is None else fence
        heading = re.match(r"^(##[ \t]+)([^\r\n]+)(\r?\n)?$", line) if fence is None else None
        if heading:
            title = heading.group(2).strip()
            replacement = canonical.get(title.casefold())
            if re.search(r"\btrends?\b.*\bopen\s+problems?\b", title, re.I):
                replacement = "Trends and open problems"
            if replacement:
                line = "## " + replacement + (heading.group(3) or "")
        lines.append(line)
    return "".join(lines)


def main(argv):
    path = Path(argv[1] if len(argv) > 1 else REPORT)
    try:
        text = path.read_text(encoding="utf-8")
        new = normalize(text)
        if new != text:
            path.write_text(new, encoding="utf-8")
        print("OK: report headings canonicalized")
        return 0
    except (OSError, UnicodeError) as exc:
        print(f"cannot normalize report: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
