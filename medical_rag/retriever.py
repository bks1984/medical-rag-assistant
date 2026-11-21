"""Load the persisted vector store and retrieve the most relevant manual excerpts."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .config import Settings, get_settings


@dataclass
class Excerpt:
    text: str
    page_number: int | None
    manual: str
    chunk_id: str = ""
    chapter: str = ""

    @property
    def label(self) -> str:
        page = f"page {self.page_number}" if self.page_number else "page unknown"
        parts = [self.manual, self.chapter, page] if self.chapter else [self.manual, page]
        return ", ".join(parts)


def get_embeddings(settings: Settings):
    from langchain_openai import OpenAIEmbeddings

    kwargs = {"model": settings.embedding_model, "api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return OpenAIEmbeddings(**kwargs)


@lru_cache(maxsize=4)
def _load_store(settings: Settings):
    from langchain_chroma import Chroma

    if not settings.chroma_dir.exists():
        raise FileNotFoundError(
            f"No vector store at {settings.chroma_dir}. Run `python scripts/ingest.py` first."
        )
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(settings),
        persist_directory=str(settings.chroma_dir),
    )


def load_vectorstore(settings: Settings | None = None):
    return _load_store(settings or get_settings())


def to_excerpt(doc) -> Excerpt:
    meta = doc.metadata or {}
    page = meta.get("page_number")
    if page is None and "page" in meta:
        page = int(meta["page"]) + 1
    return Excerpt(
        text=doc.page_content,
        page_number=page,
        manual=meta.get("manual", "Medical manual"),
        chunk_id=meta.get("chunk_id", ""),
        chapter=meta.get("chapter", ""),
    )


def retrieve(question: str, k: int | None = None, settings: Settings | None = None) -> list[Excerpt]:
    """Top-k similarity search, as in the notebook (search_type='similarity', k=3)."""
    settings = settings or get_settings()
    docs = load_vectorstore(settings).similarity_search(question, k=k or settings.top_k)
    return [to_excerpt(d) for d in docs]


def format_context(excerpts: list[Excerpt]) -> str:
    """Join excerpts into one context block, each labelled with its source and page."""
    return "\n\n".join(
        f"[Excerpt {i} | {e.label}]\n{e.text.strip()}" for i, e in enumerate(excerpts, start=1)
    )
