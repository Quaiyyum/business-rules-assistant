"""Optional: have an LLM turn the retrieved sources into a worded, cited answer.

The assistant works without this module. It is used only when asked for
(`retrieve.py --llm`, or the toggle in the UI) and only if .env provides:

    LLM_API_KEY=...      # the provider's API key
    LLM_MODEL=...        # the model name; never hard-coded here
    LLM_BASE_URL=...     # optional: the provider's OpenAI-compatible address

Any provider with an OpenAI-compatible API works, because only the address, key
and model name differ. Leave LLM_BASE_URL out to use OpenAI itself. For Google
Gemini, set it to https://generativelanguage.googleapis.com/v1beta/openai/

Fallback: if the online model fails (rate limit, outage, bad key), a model running
on this computer is tried next, when .env names one:

    LOCAL_LLM_MODEL=...      # e.g. a model pulled with Ollama
    LOCAL_LLM_BASE_URL=...   # optional, defaults to Ollama's http://localhost:11434/v1

If that fails too, or none is set up, answer_with_fallback() returns no answer and
the caller shows the quoted (no-LLM) answer instead. Nothing here crashes the app.

The model is given the retrieved sources and nothing else, at temperature 0, and
is told to answer only from them and to cite each one it uses.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

NOT_FOUND = "Not found in the knowledge base."
OLLAMA_URL = "http://localhost:11434/v1"

SYSTEM_PROMPT = f"""You answer questions from vehicle workshop staff in New Zealand, using a small knowledge base of hand-written rules and passages from official documents. Staff act on these answers, so an answer that goes beyond the sources is worse than no answer.

You are given a few labelled sources found by a search. They are the closest matches, which does not guarantee they are relevant: some may be about a neighbouring topic.

How to answer:
- Use only what the sources say. Do not add facts, figures or vehicle specifications from your own knowledge. Simple arithmetic on figures that are in the sources is fine, such as three quarters of a stated weight.
- Lead with the direct answer: yes, no, the figure, or the condition that decides it. Then one or two sentences of explanation.
- Cite every source you rely on in square brackets, exactly as its label is written, for example [LVV_001] or [nzta_truck_loading_code_2026.pdf p.15].
- Sometimes a source applies but cannot settle the question alone: it may say the answer depends on a model-specific table, or on a detail the question does not give. In that case give the conditional answer. State what the source requires and what the person needs to check. That is more useful than refusing.
- A source labelled as a staff correction was written by staff after an earlier answer to a similar question was wrong. Prefer it where it conflicts with another source, and cite it like any other source.
- A rule's type is shown beside its label. If the type is guidance, say it is a recommendation and not a legal limit. Do not call anything a legal requirement or a recommendation unless its type or its own wording says so.
- If none of the sources is about what was asked, reply exactly: {NOT_FOUND}
  This includes a question about one specific vehicle's rating or specification when the sources only give general rules. Do not offer a general rule in place of the specific fact that was asked for."""


def _api_key() -> str | None:
    """The key from LLM_API_KEY, or OPENAI_API_KEY for older .env files."""
    return os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")


def is_configured() -> bool:
    load_dotenv()
    return bool(_api_key()) and bool(os.getenv("LLM_MODEL"))


def _label(result) -> str:
    m = result.metadata
    if result.kind == "rule":
        return m["rule_id"]
    if result.kind == "correction":
        return f"correction {m['correction_id']}"
    return f"{m['source_file']} p.{m['source_page']}"


def build_context(results) -> str:
    blocks = []
    for result in results:
        note = ""
        if result.kind == "rule":
            note = f" (rule, type: {result.metadata['rule_type'].replace('_', ' ')})"
        elif result.kind == "correction":
            note = f" (staff correction, {result.metadata['date']})"
        blocks.append(f"[{_label(result)}]{note}\n{result.text}")
    return "\n\n".join(blocks)


@dataclass
class Provider:
    name: str           # shown to the user, e.g. "online model" or "local model"
    base_url: str | None
    api_key: str
    model: str


def providers() -> list[Provider]:
    """The models to try, in order: the online one, then a local one, if configured."""
    load_dotenv()
    found = []
    if _api_key() and os.getenv("LLM_MODEL"):
        found.append(Provider("online model", os.getenv("LLM_BASE_URL") or None,
                              _api_key(), os.environ["LLM_MODEL"]))
    if os.getenv("LOCAL_LLM_MODEL"):
        # Local servers such as Ollama ignore the key, but the client needs one.
        found.append(Provider("local model", os.getenv("LOCAL_LLM_BASE_URL") or OLLAMA_URL,
                              "local", os.environ["LOCAL_LLM_MODEL"]))
    return found


def _ask(provider: Provider, question: str, results) -> str:
    from openai import OpenAI

    # base_url=None means "use OpenAI"; any other address points the same client
    # at a different provider. max_retries=0: if a provider fails we move on to
    # the next one straight away instead of waiting.
    client = OpenAI(api_key=provider.api_key, base_url=provider.base_url,
                    max_retries=0, timeout=60)
    response = client.chat.completions.create(
        model=provider.model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Sources:\n\n{build_context(results)}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content.strip()


def reason(error: Exception) -> str:
    """Turn an API error into a short, plain explanation."""
    status = getattr(error, "status_code", None)
    name = type(error).__name__
    if status == 429 or name == "RateLimitError":
        return "usage limit reached"
    if status in (500, 502, 503, 504) or name == "InternalServerError":
        return "service busy or down"
    if status in (401, 403) or name in ("AuthenticationError", "PermissionDeniedError"):
        return "API key rejected"
    if status == 404 or name == "NotFoundError":
        return "model not found"
    if name in ("APIConnectionError", "APITimeoutError", "ConnectError"):
        return "could not connect"
    if status == 400:
        return "request rejected (check the API key and model name)"
    return f"unexpected error ({name})"


def answer_with_fallback(question: str, results) -> tuple[str | None, list[str]]:
    """Try each configured model in turn.

    Returns (answer, notes). answer is None when no model could answer, and the
    caller should fall back to the quoted answer. notes say what happened, in
    plain words, for showing to the user.
    """
    if not results:
        return NOT_FOUND, []
    notes = []
    available = providers()
    if not available:
        return None, ["No LLM is set up in .env."]
    for provider in available:
        try:
            answer = _ask(provider, question, results)
            if provider is not available[0]:
                notes.append(f"Answer written by the {provider.name} ({provider.model}).")
            return answer, notes
        except Exception as error:  # any failure means: try the next option
            notes.append(f"The {provider.name} ({provider.model}) is unavailable: {reason(error)}.")
    return None, notes


def llm_answer(question: str, results) -> str:
    """Return a cited answer from the first model that works, or raise if none does."""
    answer, notes = answer_with_fallback(question, results)
    if answer is None:
        raise RuntimeError(" ".join(notes))
    return answer
