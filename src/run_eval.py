"""Run every question in eval/questions.yaml and write eval/eval_report.md.

    python src/run_eval.py                  # basic path (local, free)
    python src/run_eval.py --llm            # also ask the LLM for each answer
    python src/run_eval.py --llm --only Q06,Q12

What is checked automatically:
    source   is the expected rule, or the expected file and page, ranked first?
             (also records whether it appears anywhere in the top 3)
    refusal  for "none" questions: did the system say it could not answer?

Whether the shown answer contains the question's key fact is judged by a
person: the report prints the answer next to the key fact for that check.

LLM answers are cached in eval/.llm_cache.json, keyed by the model, the prompt,
the sources and the question. Re-running costs no quota unless one of those
changed. Calls are spaced out to stay under free-tier per-minute limits.
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from retrieve import NOT_FOUND, basic_answer, open_collections, retrieve

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "questions.yaml"
REPORT = ROOT / "eval" / "eval_report.md"
CACHE = ROOT / "eval" / ".llm_cache.json"
SECONDS_BETWEEN_LLM_CALLS = 13  # free tiers often allow about 5 requests a minute


def expected_key(q: dict) -> str | None:
    """The expected source in the same form as Result.source_key."""
    e = q.get("expect") or {}
    if "rule_id" in e:
        return e["rule_id"]
    if "file" in e:
        return f"{e['file']}:{e['page']}"
    return None


def first_line(text: str, limit: int = 220) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def cell(text: str) -> str:
    """Make text safe inside a markdown table cell."""
    return text.replace("|", "/").replace("\n", " ")


# ---------------------------------------------------------------- LLM, cached

def load_cache() -> dict:
    try:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def cache_key(question: str, results, model: str) -> str:
    from llm_answer import SYSTEM_PROMPT, build_context

    raw = "\n".join([model, SYSTEM_PROMPT, build_context(results), question])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def llm_answer_cached(question: str, results, cache: dict, last_call: list) -> tuple[str | None, str]:
    """Return (answer, how). how is 'cache', 'llm' or the reason it failed."""
    from llm_answer import answer_with_fallback, providers

    available = providers()
    model = available[0].model if available else "none"
    key = cache_key(question, results, model)
    if key in cache:
        return cache[key], "cache"
    wait = SECONDS_BETWEEN_LLM_CALLS - (time.time() - last_call[0])
    if wait > 0:
        time.sleep(wait)
    answer, notes = answer_with_fallback(question, results)
    last_call[0] = time.time()
    if answer is None:
        return None, " ".join(notes) or "no LLM answer"
    cache[key] = answer
    CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return answer, "llm"


# ---------------------------------------------------------------- run

def run(use_llm: bool, only: set[str] | None) -> list[dict]:
    questions = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))
    if only:
        questions = [q for q in questions if q["id"] in only]
    collections = open_collections()   # load the store and model once, not per question
    cache, last_call = load_cache(), [0.0]
    rows = []
    for q in questions:
        # Corrections are left out so that a staff fix to an eval question cannot
        # make the eval pass; the report measures the rules and documents alone.
        results = retrieve(q["question"], q["domain"], collections, use_corrections=False)
        expected = expected_key(q)
        keys = [r.source_key for r in results]
        row = {
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "key_fact": q.get("key_fact", ""),
            "expected": expected or "(refuse)",
            "got": keys[0] if keys else "(nothing)",
            "top1": bool(expected) and keys[:1] == [expected],
            "top3": bool(expected) and expected in keys,
            "basic": basic_answer(results),
        }
        if use_llm:
            answer, how = llm_answer_cached(q["question"], results, cache, last_call)
            row["llm"], row["llm_how"] = answer, how
            print(f"{q['id']}: {how}", file=sys.stderr)
        # Refusal: the basic path cannot refuse while anything is retrieved.
        if q["type"] == "none":
            row["refused_basic"] = not results
            row["refused_llm"] = row.get("llm") == NOT_FOUND if use_llm else None
        rows.append(row)
    return rows


def mark(ok: bool | None) -> str:
    return "n/a" if ok is None else ("yes" if ok else "no")


def write_report(rows: list[dict], use_llm: bool) -> str:
    answerable = [r for r in rows if r["type"] != "none"]
    unanswerable = [r for r in rows if r["type"] == "none"]
    top1 = sum(r["top1"] for r in answerable)
    top3 = sum(r["top3"] for r in answerable)

    lines = [
        "# Evaluation report",
        "",
        f"Generated by `src/run_eval.py` on {datetime.now():%Y-%m-%d %H:%M}. "
        f"Mode: {'basic path and LLM' if use_llm else 'basic path only'}. "
        "Staff corrections are excluded.",
        "",
        "## Summary",
        "",
        "| Measure | Result |",
        "|---|---|",
        f"| Questions | {len(rows)} ({len(answerable)} answerable, {len(unanswerable)} with no answer in the data) |",
        f"| Expected source ranked first | {top1} of {len(answerable)} |",
        f"| Expected source in top 3 | {top3} of {len(answerable)} |",
    ]
    for r in unanswerable:
        lines.append(f"| {r['id']} refused (basic path) | {mark(r['refused_basic'])} |")
        if use_llm:
            lines.append(f"| {r['id']} refused (LLM) | {mark(r['refused_llm'])} |")
    if use_llm:
        not_found = sum(r.get("llm") == NOT_FOUND for r in answerable)
        failed = sum(r.get("llm") is None for r in rows)
        lines.append(f"| LLM said \"not found\" to an answerable question | {not_found} |")
        if failed:
            lines.append(f"| LLM unavailable (see notes per question) | {failed} |")

    lines += ["", "## Per question", "",
              "| Q | Type | Expected | Ranked first | First? | Top 3? |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['type']} | {cell(r['expected'])} | {cell(r['got'])} "
                     f"| {mark(r['top1']) if r['type'] != 'none' else '-'} "
                     f"| {mark(r['top3']) if r['type'] != 'none' else '-'} |")

    lines += ["", "## Answers, for checking the key fact by hand", ""]
    for r in rows:
        lines += [f"### {r['id']}: {r['question']}", "",
                  f"- Key fact: {r['key_fact']}",
                  f"- Basic answer: {first_line(r['basic'])}"]
        if use_llm:
            shown = r["llm"] if r["llm"] is not None else f"(unavailable: {r['llm_how']})"
            lines.append(f"- LLM answer: {first_line(shown, 400)}")
        lines.append("")

    lines += ["## Failures and why", "",
              "_Fill in two honest lines on the failures after reading the answers above._", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    return f"Sources ranked first: {top1}/{len(answerable)}, in top 3: {top3}/{len(answerable)}. Report: {REPORT}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--llm", action="store_true", help="also get an LLM answer per question")
    parser.add_argument("--only", help="comma-separated question ids, e.g. Q06,Q12")
    args = parser.parse_args()
    only = set(args.only.split(",")) if args.only else None
    rows = run(args.llm, only)
    print(write_report(rows, args.llm))
    return 0


if __name__ == "__main__":
    sys.exit(main())
