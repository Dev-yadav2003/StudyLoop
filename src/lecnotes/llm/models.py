"""Registry of the LLMs we are allowed to call, with their free-tier limits.

Google no longer publishes free-tier RPM/RPD (checked 2026-10-17): they are only shown in
AI Studio for your key. Values below are conservative and marked unverified — confirm them
in AI Studio and edit this file, the single source of truth for the rate limiter.

Listing a model via the API does not mean a new key may call it: `gemini-2.5-pro` and
`gemini-2.5-flash-lite` are listed but return 404 "no longer available to new users"
(probed 2026-10-17), so they are deliberately not registered — a config naming them fails at
startup, not mid-job. `gemini-3.5-flash-lite` works but ignores `temperature` ("fixed
sampling defaults"), which breaks reproducibility, so it isn't registered either.
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
    ModelInfo(  # fallback when the primary is out of quota
        name="gemini-3.1-flash-lite",
        provider="google",
        requests_per_minute=10,  # unverified, conservative
        requests_per_day=100,  # unverified, conservative
        context_tokens=1_048_576,
        video_input=True,
    ),
    ModelInfo(  # newer generation than 2.5 Flash: used as an independent eval judge
        name="gemini-3-flash-preview",
        provider="google",
        requests_per_minute=5,  # unverified, conservative
        requests_per_day=20,  # unverified, conservative
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
