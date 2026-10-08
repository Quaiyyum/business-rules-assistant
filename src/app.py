"""Streamlit interface for the business rules assistant.

    streamlit run src/app.py

Ask a question, get an answer with its citation, and see the sources it came
from. Works with no API key (quoted answers from local search). If an LLM is set
up in .env, a toggle lets it word the answer, falling back to the quoted answer
if the LLM is unavailable.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so the sibling modules import

from ingest import RULES_DIR, open_collections  # noqa: E402
from llm_answer import answer_with_fallback, providers  # noqa: E402
from retrieve import NOT_FOUND, retrieve  # noqa: E402

DOMAIN_LABELS = {"light_vehicle": "Light vehicles", "heavy_vehicle": "Heavy vehicles"}


@st.cache_resource(show_spinner="Loading the knowledge base...")
def collections():
    """Load the embedding model and the store once per app session."""
    return open_collections()


def type_label(rule_type: str) -> str:
    return rule_type.replace("_", " ").capitalize()


def show_basic_answer(results) -> None:
    """The no-LLM answer: the best rule, or the best quoted sentence, with its citation."""
    best = results[0]
    if best.kind == "rule":
        st.markdown(f"**{best.metadata['rule_id']}** · {type_label(best.metadata['rule_type'])}")
        st.info(best.text)
    else:
        st.markdown("**Quoted from the source document**")
        st.info(best.quote or best.text)
    st.caption(f"Source: {best.citation}")


def show_sources(results, show_scores: bool) -> None:
    st.subheader("Sources")
    for n, r in enumerate(results, start=1):
        kind = "Rule" if r.kind == "rule" else "Document"
        score = f" · similarity {r.similarity:.2f}" if show_scores else ""
        with st.expander(f"{n}. {kind}: {r.citation}{score}"):
            if r.kind == "rule":
                st.markdown(f"Type: {type_label(r.metadata['rule_type'])} · "
                            f"Section: {r.metadata['source_section']}")
                st.write(r.text)
            else:
                if r.quote:
                    st.markdown(f"**Best-matching sentence:** {r.quote}")
                st.markdown("**Full passage:**")
                st.write(r.text)


# ---------------------------------------------------------------- page

st.set_page_config(page_title="Business Rules Assistant", page_icon="🚚", layout="centered")
st.title("Business Rules Assistant")
st.caption("NZ vehicle modification and loading rules, answered from cited sources.")

# Known domains first, in the order of DOMAIN_LABELS (light vehicles is the
# default), then any other rule folder alphabetically.
found = {p.name for p in RULES_DIR.iterdir() if p.is_dir()}
domains = [d for d in DOMAIN_LABELS if d in found] + sorted(found - set(DOMAIN_LABELS))
llm_ready = bool(providers())

with st.sidebar:
    domain = st.selectbox("Knowledge domain", domains,
                          format_func=lambda d: DOMAIN_LABELS.get(d, d.replace("_", " ").title()))
    use_llm = st.toggle("Word the answer with an LLM", value=False, disabled=not llm_ready,
                        help="Sends the question and the retrieved sources to the LLM set up in .env.")
    if not llm_ready:
        st.caption("No LLM is set up in .env, so answers are quoted from local search.")
    show_scores = st.toggle("Show retrieval scores", value=False)
    st.divider()
    st.caption("Answers come from hand-written rules and public NZ documents. "
               "This is a knowledge lookup, not certification or engineering advice.")

with st.form("ask"):
    question = st.text_input("Question", placeholder="e.g. Can we fit 60 mm lift blocks without a cert?")
    asked = st.form_submit_button("Ask")

if asked and question.strip():
    with st.spinner("Searching..."):
        results = retrieve(question.strip(), domain, collections())
    st.session_state["last"] = {"question": question.strip(), "domain": domain, "results": results}

    st.subheader("Answer")
    if not results:
        st.warning(NOT_FOUND)
    else:
        answer = None
        if use_llm:
            with st.spinner("Asking the LLM..."):
                answer, notes = answer_with_fallback(question.strip(), results)
            for note in notes:
                st.caption(f"Note: {note}")
            if answer is None:
                st.caption("No LLM could answer, so this is the quoted answer from local search.")
        if answer == NOT_FOUND:
            st.warning(NOT_FOUND)
        elif answer is not None:
            st.markdown(answer)
        else:
            show_basic_answer(results)
        show_sources(results, show_scores)
elif asked:
    st.warning("Type a question first.")
