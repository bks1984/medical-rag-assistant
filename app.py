"""Streamlit chat UI for the Medical RAG Assistant.

    streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from medical_rag.config import SAMPLE_QUESTIONS, get_settings
from medical_rag.llm import MissingAPIKeyError, ask_llm, ask_with_prompt_engineering
from medical_rag.rag import DISCLAIMER, answer_question

st.set_page_config(page_title="Medical RAG Assistant", page_icon="🩺", layout="wide")
settings = get_settings()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("🩺 Medical RAG Assistant")
    st.caption(f"Knowledge base: **{settings.manual_name}**")
    st.caption(f"Model: `{settings.chat_model}` · Embeddings: `{settings.embedding_model}`")
    k = st.slider("Excerpts to retrieve (k)", 1, 8, settings.top_k)
    compare = st.toggle("Compare with plain LLM answers", value=False)
    st.divider()
    st.subheader("Try a use case")
    for specialty, q in SAMPLE_QUESTIONS.items():
        if st.button(specialty, use_container_width=True):
            st.session_state.pending = q
    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
    st.info(DISCLAIMER)

st.title("Clinical questions, answered from the Merck Manual")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(excerpts) -> None:
    with st.expander(f"Sources · pages {', '.join(str(e.page_number) for e in excerpts)}"):
        for i, e in enumerate(excerpts, start=1):
            st.markdown(f"**Excerpt {i} — {e.label}**")
            st.text(e.text.strip())


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("excerpts"):
            render_sources(msg["excerpts"])

question = st.chat_input("Ask a clinical question…") or st.session_state.pop("pending", None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the manual…"):
                result = answer_question(question, k=k, settings=settings)
            st.markdown(result.answer)
            if result.excerpts:
                render_sources(result.excerpts)

            if compare:
                cols = st.columns(2)
                with cols[0], st.spinner("Base LLM…"):
                    st.markdown("**Base LLM (no retrieval)**")
                    st.markdown(ask_llm(question, settings))
                with cols[1], st.spinner("Prompt-engineered LLM…"):
                    st.markdown("**Prompt-engineered LLM (no retrieval)**")
                    st.markdown(ask_with_prompt_engineering(question, settings))

            st.session_state.messages.append(
                {"role": "assistant", "content": result.answer, "excerpts": result.excerpts}
            )
        except MissingAPIKeyError as e:
            st.error(str(e))
        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:  # surface API errors instead of a stack trace
            st.error(f"Something went wrong: {e}")
