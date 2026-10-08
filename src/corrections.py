"""Read and append staff feedback in corrections/corrections.yaml.

Two kinds of entry are written by the app:

    confirmed  staff marked an answer as correct. Kept as a log of good answers;
               not used for retrieval, because it adds no new knowledge.
    corrected  staff marked an answer as wrong and wrote the right one. Ingest
               loads these into the `corrections` collection, and retrieval shows
               one first when a new question closely matches the one corrected.

The file is plain YAML in the repo, so corrections are version-controlled and can
be reviewed, edited or deleted by hand like the rules.
"""

from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CORRECTIONS_FILE = ROOT / "corrections" / "corrections.yaml"

HEADER = """\
# Staff feedback written by the app (src/app.py). Edit or delete entries by hand
# if needed, then run `python src/ingest.py` so corrections take effect.
#
# status: confirmed  the answer shown was right (a log; not used for retrieval)
# status: corrected  the answer shown was wrong; `answer` is the staff-written fix
"""


def load_corrections(domain: str | None = None, status: str | None = None) -> list[dict]:
    if not CORRECTIONS_FILE.exists():
        return []
    entries = yaml.safe_load(CORRECTIONS_FILE.read_text(encoding="utf-8")) or []
    return [e for e in entries
            if (domain is None or e.get("domain") == domain)
            and (status is None or e.get("status") == status)]


def _next_id() -> str:
    numbers = [int(e["id"][1:]) for e in load_corrections() if str(e.get("id", "")).startswith("C")]
    return f"C{max(numbers, default=0) + 1:03d}"


def append_correction(status: str, domain: str, question: str,
                      cited_source: str, answer: str) -> str:
    """Add one entry to the end of the file, keeping anything already in it. Returns its id."""
    if status not in ("confirmed", "corrected"):
        raise ValueError(f"unknown status {status!r}")
    entry = {
        "id": _next_id(),
        "date": date.today().isoformat(),
        "status": status,
        "domain": domain,
        "question": question.strip(),
        "cited_source": cited_source,
        "answer": answer.strip(),
    }
    existing = CORRECTIONS_FILE.read_text(encoding="utf-8") if CORRECTIONS_FILE.exists() else ""
    if not existing.strip():
        existing = HEADER
    block = yaml.safe_dump([entry], sort_keys=False, allow_unicode=True, width=88)
    CORRECTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CORRECTIONS_FILE.write_text(existing.rstrip("\n") + "\n\n" + block, encoding="utf-8")
    return entry["id"]
