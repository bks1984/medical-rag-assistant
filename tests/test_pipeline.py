"""Offline unit tests - no API key, PDF or vector store needed."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from medical_rag import evaluate, rag
from medical_rag.config import SAMPLE_QUESTIONS, get_settings
from medical_rag.ingest import batched, tag_chunks
from medical_rag.prompts import OUT_OF_SCOPE_ANSWER
from medical_rag.retriever import Excerpt, format_context, to_excerpt


# ------------------------------------------------------------------ score parsing
@pytest.mark.parametrize(
    "reply,key,expected",
    [
        ("Step 1...\ngroundedness_score: 4", "groundedness_score", 4),
        ("blah {groundedness_score:5}", "groundedness_score", 5),
        ("reasoning\n{relevance_score: 3}\n", "relevance_score", 3),
        ("Relevance_Score = 2", "relevance_score", 2),
        ("Final score is 4", "relevance_score", 4),          # fallback: last digit on last line
        ("I cannot rate this.", "relevance_score", None),
        ("", "groundedness_score", None),
    ],
)
def test_parse_score(reply, key, expected):
    assert evaluate.parse_score(reply, key) == expected


def test_parse_score_takes_last_match():
    reply = "Example {relevance_score:4}\n...\nrelevance_score: 2"
    assert evaluate.parse_score(reply, "relevance_score") == 2


# ------------------------------------------------------------------ chunk tagging
def _doc(text, **meta):
    return SimpleNamespace(page_content=text, metadata=meta)


def test_tag_chunks_adds_one_based_page_and_drops_unsupported_metadata():
    chunks = tag_chunks([_doc("a", page=0, source="m.pdf", junk=[1, 2]), _doc("b", page=162)], "Merck")
    assert chunks[0].metadata["page_number"] == 1
    assert chunks[1].metadata["page_number"] == 163
    assert chunks[0].metadata["manual"] == "Merck"
    assert chunks[1].metadata["chunk_id"] == "chunk-00001"
    assert "junk" not in chunks[0].metadata


def test_batched():
    assert [len(b) for _, b in batched(list(range(250)), 100)] == [100, 100, 50]


# ------------------------------------------------------------------ context building
def test_to_excerpt_falls_back_to_raw_page():
    e = to_excerpt(_doc("text", page=9))
    assert e.page_number == 10 and e.manual == "Medical manual"


def test_format_context_labels_each_excerpt():
    ctx = format_context([Excerpt("Sepsis text", 2190, "Merck"), Excerpt("More", None, "Merck")])
    assert "[Excerpt 1 | Merck, page 2190]" in ctx
    assert "[Excerpt 2 | Merck, page unknown]" in ctx


def test_user_message_survives_braces_in_manual_text():
    msg = rag.build_user_message("Q?", [Excerpt("dose {x} mg/kg", 1, "Merck")])
    assert "dose {x} mg/kg" in msg and "###Question\nQ?" in msg


# ------------------------------------------------------------------ RAG flow (mocked)
def test_answer_question_uses_retrieved_context(monkeypatch):
    excerpts = [Excerpt("Appendectomy is the treatment.", 163, "Merck")]
    captured = {}
    monkeypatch.setattr(rag, "retrieve", lambda q, k=None, settings=None: excerpts)

    def fake_chat(user, system, settings=None):
        captured["user"], captured["system"] = user, system
        return "Answer:\nAppendectomy.\n\nSource:\nMerck, page 163"

    monkeypatch.setattr(rag, "chat", fake_chat)
    result = rag.answer_question("How is appendicitis treated?")
    assert "Appendectomy is the treatment." in captured["user"]
    assert "ONLY" in captured["system"]
    assert result.pages == [163] and not result.out_of_scope


def test_answer_question_without_hits_is_out_of_scope(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda q, k=None, settings=None: [])
    monkeypatch.setattr(rag, "chat", lambda *a, **k: pytest.fail("LLM must not be called"))
    assert rag.answer_question("anything").answer == OUT_OF_SCOPE_ANSWER


def test_blank_question_short_circuits(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda *a, **k: pytest.fail("must not retrieve"))
    assert "enter a question" in rag.answer_question("   ").answer


# ------------------------------------------------------------------ config
def test_env_overrides(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "900")
    monkeypatch.setenv("TOP_K", "5")
    s = get_settings()
    assert s.chunk_size == 900 and s.top_k == 5


def test_four_use_case_questions():
    assert set(SAMPLE_QUESTIONS) == {"Critical Care", "General Surgery", "Dermatology", "Neurology"}
