"""Streamlit interface for the business rules assistant.

    streamlit run src/app.py

Ask a question, get an answer with its citation, and see the sources it came
from. Under each answer, staff can mark it correct, or wrong and type the right
answer; both are saved to corrections/corrections.yaml. Works with no API key (quoted answers from local search). If an LLM is set
up in .env, a toggle lets it word the answer, falling back to the quoted answer
if the LLM is unavailable.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so the sibling modules import

from corrections import append_correction  # noqa: E402
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
    if best.kind == "correction":
        st.markdown(f"**Staff correction {best.metadata['correction_id']}** · {best.metadata['date']}")
        st.info(best.text)
        st.caption(f"Given for the question: \"{best.metadata['question']}\"")
        return
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
        kind = {"rule": "Rule", "doc": "Document", "correction": "Staff correction"}[r.kind]
        score = f" · similarity {r.similarity:.2f}" if show_scores else ""
        with st.expander(f"{n}. {kind}: {r.citation}{score}"):
            if r.kind == "correction":
                st.markdown(f"Given for the question: \"{r.metadata['question']}\"")
                st.write(r.text)
            elif r.kind == "rule":
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

def show_feedback(last: dict) -> None:
    """Correct / Wrong buttons under the answer, writing to corrections.yaml."""
    st.divider()
    if last.get("saved"):
        st.success(last["saved"])
        return
    st.caption("Was this answer right?")
    left, right, _ = st.columns([1, 1, 4])
    if left.button("Correct", use_container_width=True):
        cid = append_correction("confirmed", last["domain"], last["question"],
                                last["cited"], last["shown"])
        last["saved"] = f"Thanks. Logged as confirmed ({cid})."
        st.rerun()
    if right.button("Wrong", use_container_width=True):
        last["correcting"] = True
    if last.get("correcting"):
        with st.form("correction"):
            fix = st.text_area("What is the right answer? Include where it comes from if you know.")
            if st.form_submit_button("Save correction"):
                if fix.strip():
                    cid = append_correction("corrected", last["domain"], last["question"],
                                            last["cited"], fix)
                    last["saved"] = (f"Saved as {cid}. It will be used for this question "
                                     "after the next `python src/ingest.py`.")
                    last["correcting"] = False
                    st.rerun()
                else:
                    st.warning("Type the right answer first.")


if asked and question.strip():
    q = question.strip()
    with st.spinner("Searching..."):
        results = retrieve(q, domain, collections())
    answer, notes = None, []
    if results and use_llm:
        with st.spinner("Asking the LLM..."):
            answer, notes = answer_with_fallback(q, results)
    # Keep the result between reruns, so the feedback buttons can act on it.
    st.session_state["last"] = {
        "question": q, "domain": domain, "results": results,
        "answer": answer, "notes": notes, "used_llm": use_llm,
        "cited": results[0].source_key if results else "",
        "shown": answer if answer is not None else (results[0].quote or results[0].text if results else NOT_FOUND),
    }
elif asked:
    st.warning("Type a question first.")

last = st.session_state.get("last")
if last:
    results, answer = last["results"], last["answer"]
    st.subheader("Answer")
    if not results:
        st.warning(NOT_FOUND)
    else:
        for note in last["notes"]:
            st.caption(f"Note: {note}")
        if last["used_llm"] and answer is None:
            st.caption("No LLM could answer, so this is the quoted answer from local search.")
        if answer == NOT_FOUND:
            st.warning(NOT_FOUND)
        elif answer is not None:
            st.markdown(answer)
        else:
            show_basic_answer(results)
        show_sources(results, show_scores)
    show_feedback(last)
