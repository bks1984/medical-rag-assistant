"""Central settings for ingestion, retrieval, generation and evaluation.

Every value can be overridden with an environment variable (or a line in `.env`),
so experiments such as a bigger chunk size never need a code change.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional at import time so the unit tests run without it
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

ROOT = Path(__file__).resolve().parent.parent

if load_dotenv is not None:
    load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    # --- OpenAI (or any OpenAI-compatible endpoint) -------------------------
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    openai_base_url: str | None = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL") or None)

    chat_model: str = field(default_factory=lambda: _env("CHAT_MODEL", "gpt-4o-mini"))
    judge_model: str = field(default_factory=lambda: _env("JUDGE_MODEL", "gpt-4o-mini"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "text-embedding-3-small"))

    # --- Generation ----------------------------------------------------------
    max_tokens: int = field(default_factory=lambda: int(_env("MAX_TOKENS", "512")))
    temperature: float = field(default_factory=lambda: float(_env("TEMPERATURE", "0")))
    top_p: float = field(default_factory=lambda: float(_env("TOP_P", "0.95")))

    # --- Data & chunking -----------------------------------------------------
    pdf_path: Path = field(default_factory=lambda: Path(_env("MANUAL_PDF", str(ROOT / "data" / "medical_diagnosis_manual.pdf"))))
    manual_name: str = field(default_factory=lambda: _env("MANUAL_NAME", "The Merck Manual of Diagnosis & Therapy"))
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "500")))       # tokens (cl100k_base)
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "50")))  # tokens
    embed_batch_size: int = field(default_factory=lambda: int(_env("EMBED_BATCH_SIZE", "100")))

    # --- Vector store & retrieval -------------------------------------------
    chroma_dir: Path = field(default_factory=lambda: Path(_env("CHROMA_DIR", str(ROOT / "chroma_store"))))
    collection_name: str = field(default_factory=lambda: _env("COLLECTION_NAME", "merck_manual"))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "3")))

    # --- Outputs -------------------------------------------------------------
    results_dir: Path = field(default_factory=lambda: Path(_env("RESULTS_DIR", str(ROOT / "results"))))


def get_settings() -> Settings:
    """Read settings fresh from the environment."""
    return Settings()


# The four use-case questions from the problem statement.
SAMPLE_QUESTIONS: dict[str, str] = {
    "Critical Care": "What is the protocol for managing sepsis in a critical care unit?",
    "General Surgery": (
        "What are the common symptoms for appendicitis, and can it be cured via medicine? "
        "If not, what surgical procedure should be followed to treat it?"
    ),
    "Dermatology": (
        "What are the effective treatments or solutions for addressing sudden patchy hair loss, "
        "commonly seen as localized bald spots on the scalp, and what could be the possible causes behind it?"
    ),
    "Neurology": (
        "What treatments are recommended for a person who has sustained a physical injury to brain tissue, "
        "resulting in temporary or permanent impairment of brain function?"
    ),
}
