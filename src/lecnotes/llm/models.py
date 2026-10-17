"""Registry of the LLMs we are allowed to call, with their free-tier limits.

Limits change without notice. Values here are the Gemini API free tier as last checked;
verify at https://ai.google.dev/gemini-api/docs/rate-limits (or AI Studio's usage page)
and edit this file — it is the single source of truth for the rate limiter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from lecnotes.errors import UnknownModel

Provider = Literal["google"]


@dataclass(frozen=True)
class ModelInfo:
    name: str
    provider: Provider
    requests_per_minute: int
    requests_per_day: int
    context_tokens: int
    video_input: bool


_MODELS = [
    ModelInfo(
        name="gemini-2.5-flash",
        provider="google",
        requests_per_minute=10,
        requests_per_day=250,
        context_tokens=1_048_576,
        video_input=True,
    ),
    ModelInfo(
        name="gemini-2.5-flash-lite",
        provider="google",
        requests_per_minute=15,
        requests_per_day=1_000,
        context_tokens=1_048_576,
        video_input=True,
    ),
    ModelInfo(
        name="gemini-2.5-pro",
        provider="google",
        requests_per_minute=5,
        requests_per_day=100,
        context_tokens=1_048_576,
        video_input=True,
    ),
]

MODELS: dict[str, ModelInfo] = {m.name: m for m in _MODELS}


def get_model(name: str) -> ModelInfo:
    try:
        return MODELS[name]
    except KeyError:
        raise UnknownModel(name, known=sorted(MODELS)) from None
