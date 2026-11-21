"""Stage 3: Retrieval-Augmented Generation over the medical manual."""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import Settings, get_settings
from .llm import chat
from .prompts import OUT_OF_SCOPE_ANSWER, RAG_SYSTEM, RAG_USER_TEMPLATE
from .retriever import Excerpt, format_context, retrieve

DISCLAIMER = (
    "This assistant summarises reference material for healthcare professionals. "
    "It is not a substitute for clinical judgement."
)


@dataclass
class RAGAnswer:
    question: str
    answer: str
    excerpts: list[Excerpt] = field(default_factory=list)

    @property
    def context(self) -> str:
        return format_context(self.excerpts)

    @property
    def pages(self) -> list[int]:
        return sorted({e.page_number for e in self.excerpts if e.page_number})

    @property
    def out_of_scope(self) -> bool:
        return OUT_OF_SCOPE_ANSWER.lower() in self.answer.lower()


def build_user_message(question: str, excerpts: list[Excerpt]) -> str:
    # str.replace (not .format) so braces inside the manual text can't break the template
    return RAG_USER_TEMPLATE.replace("{context}", format_context(excerpts)).replace("{question}", question)


def answer_question(question: str, k: int | None = None, settings: Settings | None = None) -> RAGAnswer:
    """Retrieve the top-k excerpts and generate a grounded, cited answer."""
    settings = settings or get_settings()
    question = question.strip()
    if not question:
        return RAGAnswer(question, "Please enter a question.")

    excerpts = retrieve(question, k=k, settings=settings)
    if not excerpts:
        return RAGAnswer(question, OUT_OF_SCOPE_ANSWER)

    answer = chat(build_user_message(question, excerpts), RAG_SYSTEM, settings=settings)
    return RAGAnswer(question, answer, excerpts)
