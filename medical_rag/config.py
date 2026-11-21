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


GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Per-provider defaults. Groq serves an OpenAI-compatible chat API but no embedding models,
# so with Groq the embeddings run locally with sentence-transformers (free, no key).
_PROVIDER_DEFAULTS = {
    # gpt-oss-120b is a reasoning model: its hidden reasoning counts toward max_tokens,
    # so it gets a larger budget and low reasoning effort.
    "groq": {"chat_model": "openai/gpt-oss-120b", "embedding_provider": "huggingface",
             "max_tokens": "2048", "reasoning_effort": "low"},
    "openai": {"chat_model": "gpt-4o-mini", "embedding_provider": "openai",
               "max_tokens": "512", "reasoning_effort": ""},
}
_EMBEDDING_DEFAULTS = {
    "huggingface": "BAAI/bge-small-en-v1.5",
    "openai": "text-embedding-3-small",
}


def _llm_provider() -> str:
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit:
        if explicit not in _PROVIDER_DEFAULTS:
            raise ValueError(f"LLM_PROVIDER must be one of {sorted(_PROVIDER_DEFAULTS)}, got {explicit!r}")
        return explicit
    # Auto-detect: Groq if only a Groq key is present, otherwise OpenAI
    return "groq" if os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY") else "openai"


def _provider_default(key: str) -> str:
    return _PROVIDER_DEFAULTS[_llm_provider()][key]


def _embedding_provider() -> str:
    return _env("EMBEDDING_PROVIDER", _provider_default("embedding_provider")).lower()


@dataclass(frozen=True)
class Settings:
    # --- LLM provider: "groq" or "openai" (both via the OpenAI-compatible client) --------
    llm_provider: str = field(default_factory=_llm_provider)
    api_key: str = field(
        default_factory=lambda: _env("GROQ_API_KEY", "") if _llm_provider() == "groq" else _env("OPENAI_API_KEY", "")
    )
    base_url: str | None = field(
        default_factory=lambda: GROQ_BASE_URL if _llm_provider() == "groq" else (os.getenv("OPENAI_BASE_URL") or None)
    )

    chat_model: str = field(default_factory=lambda: _env("CHAT_MODEL", _provider_default("chat_model")))
    judge_model: str = field(default_factory=lambda: _env("JUDGE_MODEL", _env("CHAT_MODEL", _provider_default("chat_model"))))

    # --- Embeddings: "huggingface" (local) or "openai" ---------------------------------
    embedding_provider: str = field(default_factory=_embedding_provider)
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", _EMBEDDING_DEFAULTS[_embedding_provider()]))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))  # only for OpenAI embeddings

    # --- Generation ----------------------------------------------------------
    max_tokens: int = field(default_factory=lambda: int(_env("MAX_TOKENS", _provider_default("max_tokens"))))
    # For reasoning models (gpt-oss: low|medium|high, qwen3: none|default). Empty = don't send.
    reasoning_effort: str = field(default_factory=lambda: os.getenv("REASONING_EFFORT", _provider_default("reasoning_effort")))
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
    # One collection per embedding provider, since their vector sizes differ (384 vs 1,536)
    collection_name: str = field(default_factory=lambda: _env("COLLECTION_NAME", f"merck_manual_{_embedding_provider()}"))
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
