"""Load the knowledge base into Chroma.

Two collections are built:
    rules  one chunk per hand-written rule (never split)
    docs   the source PDFs, page by page, in overlapping chunks

Every chunk carries its domain, so the same code serves any domain folder:

    python src/ingest.py                         # all domains
    python src/ingest.py --domain light_vehicle
    python src/ingest.py --domain heavy_vehicle

Each run deletes everything stored for the chosen domain and rebuilds it, so an
edited rule can never leave its old version behind.

Embeddings are computed locally (sentence-transformers). No API key is needed.
"""

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

import yaml
from pypdf import PdfReader

from fetch_docs import SOURCES

ROOT = Path(__file__).resolve().parent.parent
RULES_DIR = ROOT / "rules"
RAW_DIR = ROOT / "docs" / "raw"
CHROMA_DIR = ROOT / "chroma_store"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
# Chunk sizes are in CHARACTERS, not tokens. 500 characters is roughly 100-125
# tokens, which fits inside the embedding model's input limit (all-MiniLM-L6-v2
# reads at most 256 tokens and silently ignores the rest). A 500-token chunk
# would be cut roughly in half before embedding, so its second half could never
# be found by search.
CHUNK_CHARS = 500     # target chunk length, in characters
OVERLAP_CHARS = 50    # characters shared between neighbouring chunks
MIN_CHUNK_CHARS = 40  # ignore fragments shorter than this
BATCH = 500

# fetch_docs.py labels each PDF "light" or "heavy"; the rule folders are
# rules/light_vehicle and rules/heavy_vehicle.
PDF_DOMAIN = {name: f"{label}_vehicle" for name, label, _url in SOURCES}

REQUIRED = ("id", "vehicle_class", "topic", "rule_type", "rule", "source", "last_reviewed")
REQUIRED_SOURCE = ("document", "file", "page", "section")


# ---------------------------------------------------------------- rules

def load_rules(domain: str) -> list[dict]:
    """Read every YAML file in rules/<domain>/ and check each rule is complete."""
    rules = []
    for path in sorted((RULES_DIR / domain).glob("*.yaml")):
        for rule in yaml.safe_load(path.read_text(encoding="utf-8")) or []:
            where = f"{path.relative_to(ROOT)} ({rule.get('id', 'no id')})"
            missing = [k for k in REQUIRED if k not in rule]
            missing += [f"source.{k}" for k in REQUIRED_SOURCE if k not in rule.get("source", {})]
            if missing:
                raise ValueError(f"{where}: missing {', '.join(missing)}")
            rules.append(rule)
    ids = [r["id"] for r in rules]
    duplicates = [i for i, n in Counter(ids).items() if n > 1]
    if duplicates:
        raise ValueError(f"Duplicate rule ids in {domain}: {duplicates}")
    return rules


def rule_record(rule: dict, domain: str) -> tuple[str, str, dict]:
    """Turn one rule into (id, text to embed, flat metadata).

    Chroma metadata must be flat (text or numbers only), so the nested
    `source` block is flattened here. The YAML stays nested for people.
    """
    source = rule["source"]
    text = " ".join(str(rule["rule"]).split())
    metadata = {
        "type": "rule",
        "domain": domain,
        "rule_id": rule["id"],
        "vehicle_class": rule["vehicle_class"],
        "topic": rule["topic"],
        "rule_type": rule["rule_type"],
        "issuer": rule.get("authority") or rule.get("manufacturer") or "",
        "source_document": source["document"],
        "source_file": source["file"],
        "source_page": int(source["page"]),
        "source_section": source["section"],
        "last_reviewed": str(rule["last_reviewed"]),
    }
    if "printed_page" in source:
        metadata["source_printed_page"] = int(source["printed_page"])
    if source.get("also_at_pages"):
        metadata["source_also_at_pages"] = ",".join(str(p) for p in source["also_at_pages"])
    return rule["id"], text, metadata


# ---------------------------------------------------------------- documents

def repeated_lines(pages: list[str]) -> set[str]:
    """Find headers and footers: lines that recur on more than half the pages.

    Digits are masked first so "Page 3 of 21" and "Page 4 of 21" count as the
    same line.
    """
    if len(pages) < 4:
        return set()
    counts = Counter()
    for page in pages:
        counts.update({re.sub(r"\d+", "#", line.strip()) for line in page.splitlines() if line.strip()})
    return {line for line, n in counts.items() if n > len(pages) / 2}


def clean_page(page: str, boilerplate: set[str]) -> str:
    kept = [
        line for line in page.splitlines()
        if line.strip() and re.sub(r"\d+", "#", line.strip()) not in boilerplate
    ]
    return " ".join(" ".join(kept).split())


def chunk(text: str) -> list[str]:
    """Split text into pieces of about CHUNK_CHARS characters that overlap and end on a word."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start + CHUNK_CHARS // 2:
                end = space
        piece = text[start:end].strip()
        if len(piece) >= MIN_CHUNK_CHARS:
            chunks.append(piece)
        if end >= len(text):
            break
        start = end - OVERLAP_CHARS
    return chunks


def pdf_records(pdf_path: Path, domain: str) -> list[tuple[str, str, dict]]:
    pages = [page.extract_text() or "" for page in PdfReader(pdf_path).pages]
    boilerplate = repeated_lines(pages)
    records = []
    for page_number, page in enumerate(pages, start=1):
        for n, piece in enumerate(chunk(clean_page(page, boilerplate))):
            records.append((
                f"{pdf_path.name}:p{page_number}:c{n}",
                piece,
                {
                    "type": "doc",
                    "domain": domain,
                    "source_file": pdf_path.name,
                    "source_page": page_number,
                },
            ))
    return records


# ---------------------------------------------------------------- store

def open_collections():
    """Open (or create) the two collections with a local embedding model."""
    import chromadb
    from chromadb.utils import embedding_functions

    embed = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    settings = {"hnsw": {"space": "cosine"}}
    return (
        client.get_or_create_collection("rules", embedding_function=embed, configuration=settings),
        client.get_or_create_collection("docs", embedding_function=embed, configuration=settings),
    )


def replace_domain(collection, domain: str, records: list[tuple[str, str, dict]]) -> None:
    """Delete everything stored for this domain, then add the fresh records."""
    collection.delete(where={"domain": domain})
    for i in range(0, len(records), BATCH):
        ids, texts, metadatas = zip(*records[i:i + BATCH])
        collection.add(ids=list(ids), documents=list(texts), metadatas=list(metadatas))


def ingest(domain: str, rules_collection, docs_collection) -> None:
    rules = load_rules(domain)
    replace_domain(rules_collection, domain, [rule_record(r, domain) for r in rules])

    doc_records, files = [], 0
    for name, pdf_domain in PDF_DOMAIN.items():
        if pdf_domain != domain:
            continue
        pdf_path = RAW_DIR / name
        if not pdf_path.exists():
            print(f"  missing {name} (run src/fetch_docs.py)")
            continue
        doc_records.extend(pdf_records(pdf_path, domain))
        files += 1
    replace_domain(docs_collection, domain, doc_records)
    print(f"{domain}: {len(rules)} rules, {len(doc_records)} document chunks from {files} PDFs")


def main() -> int:
    domains = sorted(p.name for p in RULES_DIR.iterdir() if p.is_dir())
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domain", default="all", choices=domains + ["all"])
    args = parser.parse_args()

    rules_collection, docs_collection = open_collections()
    for domain in (domains if args.domain == "all" else [args.domain]):
        ingest(domain, rules_collection, docs_collection)
    print(f"\nStore now holds {rules_collection.count()} rules and "
          f"{docs_collection.count()} document chunks in {CHROMA_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
