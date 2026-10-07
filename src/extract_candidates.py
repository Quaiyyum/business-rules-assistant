"""Step 1 of rule authoring: mine the source PDFs for candidate rule sentences.

    PDF -> text page by page -> sentences -> keep the ones that look like a
    requirement -> CSV for a person to review.

This script does NOT write rules. It only shortlists sentences, each with its
file and page, so a reviewer can accept, reject or merge them. Accepted
candidates are then written by hand into rules/<domain>/*.yaml.

Usage:
    python src/extract_candidates.py
    python src/extract_candidates.py --only nzta_truck_loading_code_2026.pdf
"""

import csv
import re
import sys
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "docs" / "raw"
OUT_FILE = ROOT / "docs" / "candidates" / "candidates.csv"

# A sentence is a candidate if it sounds like an obligation or states a limit.
OBLIGATION = re.compile(
    r"\b(must( not)?|shall( not)?|required?|requires|not permitted|prohibited|"
    r"do not|don't|never|only (if|when)|provided that|unless)\b",
    re.IGNORECASE,
)
LIMIT = re.compile(
    r"\b(at least|no more than|not more than|not less than|no less than|"
    r"maximum|minimum|exceed(s|ing)?|within|up to|or more|or less)\b",
    re.IGNORECASE,
)
QUANTITY = re.compile(
    r"\d[\d,.]*\s?(mm|cm|m|metres?|km/h|km|kg|tonnes?|t|g|%|degrees?|°|kPa|N)(?!\w)"
)

MIN_WORDS, MAX_WORDS = 6, 80


def sentences(page_text: str) -> list[str]:
    """Join wrapped lines, then split on sentence ends and bullet markers."""
    text = re.sub(r"\s+", " ", page_text)
    parts = re.split(r"(?<=[.;:])\s+(?=[A-Z0-9(])|\s[•▪›]\s", text)
    return [p.strip() for p in parts if p and p.strip()]


def triggers(sentence: str) -> list[str]:
    found = []
    if OBLIGATION.search(sentence):
        found.append("obligation")
    if LIMIT.search(sentence):
        found.append("limit")
    if QUANTITY.search(sentence):
        found.append("quantity")
    return found


def score(found: list[str]) -> int:
    """A number plus an obligation or limit word is the strongest signal."""
    return len(found) + (1 if "quantity" in found and len(found) > 1 else 0)


def mine(pdf_path: Path) -> list[dict]:
    rows = []
    reader = PdfReader(pdf_path)
    for page_number, page in enumerate(reader.pages, start=1):
        for sentence in sentences(page.extract_text() or ""):
            words = len(sentence.split())
            if not MIN_WORDS <= words <= MAX_WORDS:
                continue
            found = triggers(sentence)
            if not found:
                continue
            rows.append(
                {
                    "file": pdf_path.name,
                    "pdf_page": page_number,
                    "score": score(found),
                    "triggers": "+".join(found),
                    "sentence": sentence,
                    "decision": "",   # reviewer fills: accept / reject / merge
                    "rule_id": "",    # reviewer fills when accepted
                    "note": "",
                }
            )
    return rows


def main() -> int:
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    pdfs = sorted(p for p in RAW_DIR.glob("*.pdf") if only in (None, p.name))
    if not pdfs:
        print(f"No PDFs found in {RAW_DIR}. Run src/fetch_docs.py first.")
        return 1

    all_rows = []
    for pdf in pdfs:
        rows = mine(pdf)
        strong = sum(r["score"] >= 3 for r in rows)
        print(f"{pdf.name}: {len(rows)} candidates ({strong} strong)")
        all_rows.extend(rows)

    all_rows.sort(key=lambda r: (r["file"], -r["score"], r["pdf_page"]))
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n{len(all_rows)} candidates written to {OUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
