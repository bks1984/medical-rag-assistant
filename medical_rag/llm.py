"""Thin wrapper around the OpenAI chat API.

Stage 1 of the notebook (`ask_llm`) and stage 2 (`response` with a system prompt) are both
covered by `chat()`. The client is created lazily so importing this module never needs a key.
"""
from __future__ import annotations

import re
from functools import lru_cache

from .config import Settings, get_settings


class MissingAPIKeyError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _client(api_key: str, base_url: str | None):
    from openai import OpenAI

    # Generous retries: Groq's free tier has low tokens-per-minute limits, and the client
    # backs off using the server's retry-after header.
    return OpenAI(api_key=api_key, base_url=base_url, max_retries=8)


def get_client(settings: Settings | None = None):
    settings = settings or get_settings()
    if not settings.api_key:
        key_name = "GROQ_API_KEY" if settings.llm_provider == "groq" else "OPENAI_API_KEY"
        raise MissingAPIKeyError(f"{key_name} is not set. Copy .env.example to .env and add your key.")
    return _client(settings.api_key, settings.base_url)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Some Groq-hosted reasoning models (e.g. Qwen) prepend <think>...</think>; drop it."""
    return _THINK_RE.sub("", text or "").strip()


def chat(
    user_prompt: str,
    system_prompt: str | None = None,
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    settings: Settings | None = None,
) -> str:
    """Send one chat turn and return the reply text."""
    settings = settings or get_settings()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    extra = {"reasoning_effort": settings.reasoning_effort} if settings.reasoning_effort else {}
    completion = get_client(settings).chat.completions.create(
        model=model or settings.chat_model,
        messages=messages,
        # never go below the provider default: reasoning models need room before answering
        max_tokens=max(max_tokens or 0, settings.max_tokens),
        temperature=settings.temperature if temperature is None else temperature,
        top_p=settings.top_p if top_p is None else top_p,
        **extra,
    )
    return strip_reasoning(completion.choices[0].message.content)


def ask_llm(question: str, settings: Settings | None = None) -> str:
    """Stage 1 - the plain LLM, no system prompt and no retrieval."""
    return chat(question, settings=settings)


def ask_with_prompt_engineering(question: str, settings: Settings | None = None) -> str:
    """Stage 2 - the LLM with a role-setting system prompt, still no retrieval."""
    from .prompts import PROMPT_ENG_SYSTEM

    return chat(question, PROMPT_ENG_SYSTEM, settings=settings)
