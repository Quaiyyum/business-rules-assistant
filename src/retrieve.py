"""Find the rules and document passages that answer a question.

This is the core of the assistant and it needs no API key: the question is
embedded locally, matched by meaning against the `rules` and `docs` collections,
and the best matches come back with their citations.

    python src/retrieve.py "Can we fit 60 mm lift blocks without a cert?"
    python src/retrieve.py "..." --domain light_vehicle
    python src/retrieve.py "..." --llm      # optional: have an LLM word the answer

Scoring: Chroma returns a cosine distance per match; similarity = 1 - distance.
Rules get a fixed boost so that a stated business rule beats a sentence buried in
a manual. Matches below MIN_SIMILARITY are dropped as unrelated.

Document chunks are small (about 500 characters) so that search is precise, but a
chunk often starts or stops mid-sentence. So each document hit is shown together
with the chunk before and after it on the same page ("search small, show big").

That passage is complete but long, so the sentence in it that best matches the
question is picked out and quoted as the answer (an "extractive" answer: the
system quotes the source word for word and never writes anything itself). The
full passage is still available with --context.
"""

import argparse
import re
import sys
from dataclasses import dataclass

import numpy as np

from ingest import get_embedder, open_collections

TOP_K = 3              # results returned
CANDIDATES = 5         # matches fetched from each collection before ranking
RULE_BOOST = 0.10      # added to a rule's similarity when ranking
MIN_SENTENCE_CHARS = 25  # ignore fragments shorter than this when choosing a quote
SECOND_QUOTE_MARGIN = 0.05  # also quote the runner-up sentence if it scores this close
MIN_SIMILARITY = 0.30  # below this (before any boost), treat as "not found"

NOT_FOUND = "Not found in the knowledge base."


@dataclass
class Result:
    id: str            # rule id, or "<file>:p<page>:c<chunk number>" for a document chunk
    kind: str          # "rule" or "doc"
    text: str
    similarity: float  # 0 to 1, higher is closer in meaning
    score: float       # similarity plus the rule boost; used for ranking
    metadata: dict
    quote: str = ""    # for document results: the sentence that best matches the question

    @property
    def citation(self) -> str:
        m = self.metadata
        if self.kind == "rule":
            return f"{m['rule_id']} ({m['source_document']}, p.{m['source_page']})"
        return f"{m['source_file']}, p.{m['source_page']}"

    @property
    def source_key(self) -> str:
        """Short machine-readable source, used by the eval: a rule id or file:page."""
        m = self.metadata
        return m["rule_id"] if self.kind == "rule" else f"{m['source_file']}:{m['source_page']}"


def _query(collection, kind: str, question: str, domain: str | None) -> list[Result]:
    found = collection.query(
        query_texts=[question],
        n_results=CANDIDATES,
        where={"domain": domain} if domain else None,
    )
    results = []
    for id_, text, metadata, distance in zip(
        found["ids"][0], found["documents"][0], found["metadatas"][0], found["distances"][0]
    ):
        similarity = 1 - distance
        boost = RULE_BOOST if kind == "rule" else 0.0
        results.append(Result(id_, kind, text, similarity, similarity + boost, metadata))
    return results


def _join(left: str, right: str) -> str:
    """Join two neighbouring chunks, removing the text they share at the seam.

    Neighbouring chunks overlap by about 50 characters, so the end of `left`
    repeats at the start of `right`. Find the longest such repeat and keep it once.
    """
    for size in range(min(len(left), len(right), 120), 0, -1):
        if left.endswith(right[:size]):
            return left + right[size:]
    return f"{left} {right}"


def _with_neighbours(docs_collection, result: Result) -> str:
    """Return a document chunk together with the chunks before and after it on its page.

    A chunk id ends in ":c<number>", its position on the page. The neighbours are
    the same id with number - 1 and number + 1. Either may not exist (first or
    last chunk on the page), and get() does not promise to keep the order asked
    for, so results are matched by id.
    """
    base, number = result.id.rsplit(":c", 1)
    number = int(number)
    before, after = f"{base}:c{number - 1}", f"{base}:c{number + 1}"
    wanted = [after] if number == 0 else [before, after]
    found = docs_collection.get(ids=wanted)
    neighbours = dict(zip(found["ids"], found["documents"]))
    text = result.text
    if before in neighbours:
        text = _join(neighbours[before], text)
    if after in neighbours:
        text = _join(text, neighbours[after])
    return text


def _sentences(passage: str) -> list[str]:
    """Split a passage into sentences and bullet points."""
    parts = re.split(r"(?<=[.?!])\s+(?=[A-Z0-9(])|\s[•▪›]\s", passage)
    return [p.strip() for p in parts if p and len(p.strip()) >= MIN_SENTENCE_CHARS]


def best_quote(question: str, passage: str) -> str:
    """Pick the sentence in the passage that is closest in meaning to the question.

    Each sentence is embedded with the same local model used for search and
    compared with the question by cosine similarity. The runner-up is included
    too when it scores almost as well, kept in the order it appears on the page.
    """
    sentences = _sentences(passage)
    if len(sentences) < 2:
        return passage
    vectors = np.asarray(get_embedder()([question] + sentences), dtype=float)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    scores = vectors[1:] @ vectors[0]
    order = np.argsort(scores)[::-1]
    chosen = [order[0]]
    if scores[order[0]] - scores[order[1]] <= SECOND_QUOTE_MARGIN:
        chosen.append(order[1])
    return " ... ".join(sentences[i] for i in sorted(chosen))


def retrieve(question: str, domain: str | None = None, collections=None) -> list[Result]:
    """Return up to TOP_K results, best first. An empty list means "not found"."""
    rules_collection, docs_collection = collections or open_collections()
    results = _query(rules_collection, "rule", question, domain)
    results += _query(docs_collection, "doc", question, domain)
    results = [r for r in results if r.similarity >= MIN_SIMILARITY]
    results = sorted(results, key=lambda r: r.score, reverse=True)[:TOP_K]
    for result in results:
        if result.kind == "doc":   # rules are never chunked, so only documents need this
            result.text = _with_neighbours(docs_collection, result)
            result.quote = best_quote(question, result.text)
    return results


def basic_answer(results: list[Result], show_context: bool = False) -> str:
    """The no-LLM answer: quote the best match word for word, with its citation.

    A rule is shown whole. A document result shows its best-matching sentence;
    the full passage is added only when show_context is True.
    """
    if not results:
        return NOT_FOUND
    best, others = results[0], results[1:]
    if best.kind == "rule":
        lines = [f"Rule: {best.text}", f"Source: {best.citation}",
                 f"Type: {best.metadata['rule_type'].replace('_', ' ')}"]
    else:
        lines = [f"Answer (quoted from source): {best.quote or best.text}", f"Source: {best.citation}"]
        if show_context:
            lines += ["", f"Full passage: {best.text}"]
    if others:
        lines.append("")
        lines.append("Also relevant:")
        for r in others:
            lines.append(f"  - {r.citation}")
            if r.kind == "doc" and r.quote:
                lines.append(f"      \"{r.quote}\"")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--domain", default=None, help="light_vehicle or heavy_vehicle")
    parser.add_argument("--llm", action="store_true", help="word the answer with an LLM (needs .env)")
    parser.add_argument("--context", action="store_true", help="also print the full passage a quote came from")
    args = parser.parse_args()

    results = retrieve(args.question, args.domain)
    answer = None
    if args.llm:
        from llm_answer import answer_with_fallback
        answer, notes = answer_with_fallback(args.question, results)
        for note in notes:
            print(f"Note: {note}")
        if answer is None:
            print("Note: no LLM could answer, so this is the quoted answer from local search (no AI wording).\n")
    print(answer if answer is not None else basic_answer(results, show_context=args.context))

    print("\nRetrieved:")
    for r in results:
        print(f"  {r.score:.2f} (similarity {r.similarity:.2f})  {r.kind:4}  {r.citation}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
