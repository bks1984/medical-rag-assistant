"""Build the Chroma vector store from the medical manual PDF.

Pipeline (same as the notebook, made re-runnable):
    PDF --PyMuPDF--> pages --token splitter (cl100k_base)--> chunks
        --OpenAI embeddings--> Chroma (persisted on disk)

Each chunk keeps its page number so answers can cite "page N".
"""
from __future__ import annotations

import re
import shutil
import time
from typing import Callable, Iterable

from .config import Settings, get_settings

# Licensed copies carry a per-user watermark on every page (an email, an id code and a
# two-line notice). It is noise for retrieval and personal data, so it is removed.
_WATERMARK_LINE = re.compile(
    r"^\s*("
    r"\S+@\S+\.\S+"                                   # the licensee's email
    r"|(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{10}"  # licence/id code (letters AND digits, so headings like PREDNISONE survive)
    r"|This file is meant for personal use.*"
    r"|Sharing or publishing the contents.*"
    r")\s*$"
)
# Page footer:  "The Merck Manual of Diagnosis & Therapy, 19th Edition" / "Chapter 11. Acute Abdomen ..." / "163"
# Variants seen: edition and chapter on one line (with or without a space), and index pages
# where the "chapter" line is just a letter.
_FOOTER = re.compile(
    r"\n(?P<edition>The Merck Manual[^\n]*?Edition)[ \t]*\n?(?P<section>[^\n]*)\n(?P<page>\d{1,4})\s*$"
)


def clean_page_text(text: str) -> tuple[str, dict]:
    """Remove the watermark and the running footer; return (clean text, footer metadata)."""
    lines = [ln for ln in text.splitlines() if not _WATERMARK_LINE.match(ln)]
    body = "\n".join(lines).rstrip()
    meta: dict = {}
    m = _FOOTER.search("\n" + body)
    if m:
        meta = {"printed_page": int(m.group("page"))}
        section = m.group("section").strip()
        if section.startswith("Chapter"):
            meta["chapter"] = section
        body = ("\n" + body)[: m.start()].strip()
    return body, meta


def load_pages(settings: Settings):
    from langchain_community.document_loaders import PyMuPDFLoader

    if not settings.pdf_path.exists():
        raise FileNotFoundError(
            f"Manual not found at {settings.pdf_path}. "
            "Put the PDF there or set MANUAL_PDF in .env (see data/README.md)."
        )
    pages = []
    for doc in PyMuPDFLoader(str(settings.pdf_path)).load():
        doc.page_content, footer = clean_page_text(doc.page_content)
        doc.metadata.update(footer)
        if doc.page_content.strip():
            pages.append(doc)
    return pages


def make_splitter(settings: Settings):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def tag_chunks(chunks: Iterable, manual_name: str) -> list:
    """Add a human-readable page number, manual name and stable chunk id to each chunk."""
    tagged = []
    for i, doc in enumerate(chunks):
        # Prefer the page number printed in the manual's footer; fall back to the PDF index
        printed = doc.metadata.get("printed_page")
        doc.metadata["page_number"] = int(printed) if printed else int(doc.metadata.get("page", 0)) + 1
        doc.metadata["manual"] = manual_name
        doc.metadata["chunk_id"] = f"chunk-{i:05d}"
        # Chroma only stores str/int/float/bool metadata
        doc.metadata = {k: v for k, v in doc.metadata.items() if isinstance(v, (str, int, float, bool))}
        tagged.append(doc)
    return tagged


def batched(items: list, size: int):
    for start in range(0, len(items), size):
        yield start, items[start : start + size]


def build_vectorstore(
    settings: Settings | None = None,
    *,
    rebuild: bool = True,
    progress: Callable[[str], None] = print,
):
    """Load, chunk, embed and persist the manual. Returns the Chroma store."""
    from langchain_chroma import Chroma

    from .retriever import get_embeddings

    settings = settings or get_settings()

    progress(f"Loading {settings.pdf_path.name} ...")
    pages = load_pages(settings)
    progress(f"  {len(pages)} pages loaded")

    chunks = tag_chunks(make_splitter(settings).split_documents(pages), settings.manual_name)
    progress(f"  {len(chunks)} chunks (size={settings.chunk_size}, overlap={settings.chunk_overlap} tokens)")

    if rebuild and settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    store = Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(settings),
        persist_directory=str(settings.chroma_dir),
    )
    # Embed in batches to stay under per-request token limits and soften rate limiting.
    for start, batch in batched(chunks, settings.embed_batch_size):
        store.add_documents(batch, ids=[d.metadata["chunk_id"] for d in batch])
        progress(f"  embedded {min(start + len(batch), len(chunks))}/{len(chunks)}")
        time.sleep(0.3)

    progress(f"Vector store saved to {settings.chroma_dir}")
    return store
