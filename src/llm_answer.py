"""Optional: have an LLM turn the retrieved sources into a worded, cited answer.

The assistant works without this module. It is used only when asked for
(`retrieve.py --llm`, or the toggle in the UI) and only if .env provides:

    OPENAI_API_KEY=...
    LLM_MODEL=...        # a pinned model version; never hard-coded here

The model is given the retrieved sources and nothing else, at temperature 0, and
is told to answer only from them and to cite each one it uses.
"""

import os

from dotenv import load_dotenv

NOT_FOUND = "Not found in the knowledge base."

SYSTEM_PROMPT = f"""You answer questions for vehicle workshop staff in New Zealand.

Use ONLY the numbered sources provided. Do not use any other knowledge.
- Start with a direct answer (yes, no, or the figure asked for), then one or two
  sentences of explanation.
- Cite every source you rely on in square brackets, exactly as its label is
  written, for example [LVV_001] or [nzta_truck_loading_code_2026.pdf p.15].
- If a source is marked as guidance, say it is a recommendation, not a legal limit.
- If the sources do not answer the question, reply exactly: {NOT_FOUND}
  Do not guess, and do not give a figure that is not in the sources."""


def is_configured() -> bool:
    load_dotenv()
    return bool(os.getenv("OPENAI_API_KEY")) and bool(os.getenv("LLM_MODEL"))


def _label(result) -> str:
    m = result.metadata
    if result.kind == "rule":
        return m["rule_id"]
    return f"{m['source_file']} p.{m['source_page']}"


def build_context(results) -> str:
    blocks = []
    for result in results:
        note = ""
        if result.kind == "rule":
            note = f" (rule, type: {result.metadata['rule_type'].replace('_', ' ')})"
        blocks.append(f"[{_label(result)}]{note}\n{result.text}")
    return "\n\n".join(blocks)


def llm_answer(question: str, results) -> str:
    """Return the LLM's cited answer, or NOT_FOUND when nothing was retrieved."""
    if not results:
        return NOT_FOUND
    if not is_configured():
        raise RuntimeError("Set OPENAI_API_KEY and LLM_MODEL in .env to use the LLM answer.")

    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=os.environ["LLM_MODEL"],
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Sources:\n\n{build_context(results)}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content.strip()
