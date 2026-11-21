"""Thin wrapper around the OpenAI chat API.

Stage 1 of the notebook (`ask_llm`) and stage 2 (`response` with a system prompt) are both
covered by `chat()`. The client is created lazily so importing this module never needs a key.
"""
from __future__ import annotations

from functools import lru_cache

from .config import Settings, get_settings


class MissingAPIKeyError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _client(api_key: str, base_url: str | None):
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


def get_client(settings: Settings | None = None):
    settings = settings or get_settings()
    if not settings.openai_api_key:
        raise MissingAPIKeyError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return _client(settings.openai_api_key, settings.openai_base_url)


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

    completion = get_client(settings).chat.completions.create(
        model=model or settings.chat_model,
        messages=messages,
        max_tokens=max_tokens or settings.max_tokens,
        temperature=settings.temperature if temperature is None else temperature,
        top_p=settings.top_p if top_p is None else top_p,
    )
    return (completion.choices[0].message.content or "").strip()


def ask_llm(question: str, settings: Settings | None = None) -> str:
    """Stage 1 - the plain LLM, no system prompt and no retrieval."""
    return chat(question, settings=settings)


def ask_with_prompt_engineering(question: str, settings: Settings | None = None) -> str:
    """Stage 2 - the LLM with a role-setting system prompt, still no retrieval."""
    from .prompts import PROMPT_ENG_SYSTEM

    return chat(question, PROMPT_ENG_SYSTEM, settings=settings)
