#!/usr/bin/env python3
"""Detects a report or slide PDF that is older than its sources.

    python scripts/docs_stamp.py write   # after rebuilding the PDFs: record the source hashes
    python scripts/docs_stamp.py check   # CI: fail when a source changed but the PDF was not rebuilt

The stamp (docs/pdf-stamps.json) holds a SHA-256 over the LaTeX sources, the generated fragments, the
bibliography and the figures each PDF is built from, plus the PDF's own page count. Standard library only.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAMP = ROOT / "docs" / "pdf-stamps.json"
FIGURES = sorted((ROOT / "results" / "figures").glob("*"))
GENERATED = sorted((ROOT / "report" / "generated").glob("*.tex"))

TARGETS = {
    "report": {
        "pdf": ROOT / "report" / "main.pdf",
        "sources": [
            ROOT / "report" / "main.tex",
            *sorted((ROOT / "report" / "sections").glob("*.tex")),
            ROOT / "report" / "references.bib",
            *GENERATED,
            *FIGURES,
        ],
    },
    "slides": {
        "pdf": ROOT / "slides" / "WriteRHand-slides.pdf",
        "sources": [
            ROOT / "slides" / "beamer" / "main.tex",
            *sorted((ROOT / "slides" / "beamer").glob("*.png")),
            *GENERATED,
            *FIGURES,
        ],
    },
}


def digest(files):
    h = hashlib.sha256()
    for f in files:
        h.update(f.relative_to(ROOT).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def pages(pdf):
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if line.startswith("Pages:"):
            return int(line.split()[1])
    raise SystemExit(f"pdfinfo gave no page count for {pdf}")


def current():
    for name, t in TARGETS.items():
        if not t["pdf"].is_file():
            raise SystemExit(f"{t['pdf'].relative_to(ROOT)} is missing; build it first")
    return {n: {"sources_sha256": digest(t["sources"]), "pages": pages(t["pdf"])} for n, t in TARGETS.items()}


def main(argv):
    mode = argv[1] if len(argv) > 1 else "check"
    now = current()
    if mode == "write":
        STAMP.parent.mkdir(exist_ok=True)
        STAMP.write_text(json.dumps(now, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {STAMP.relative_to(ROOT)}: " + ", ".join(f"{n} {v['pages']} pages" for n, v in now.items()))
        return 0
    if not STAMP.is_file():
        print("docs/pdf-stamps.json is missing; run: python scripts/docs_stamp.py write")
        return 1
    saved = json.loads(STAMP.read_text(encoding="utf-8"))
    bad = [n for n in now if saved.get(n) != now[n]]
    if bad:
        print(
            "stale PDF(s): " + ", ".join(bad) + ". The sources changed (or the PDF was replaced) since the last "
            "recorded build. Rebuild the PDF and run: python scripts/docs_stamp.py write"
        )
        return 1
    print("PDF stamps match: " + ", ".join(f"{n} ({v['pages']} pages)" for n, v in now.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
