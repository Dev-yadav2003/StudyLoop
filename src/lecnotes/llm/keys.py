"""API-key failover: several Gemini keys, used one at a time.

A key is benched only for problems that are the key's own — invalid/revoked/forbidden, or
its quota exhausted — and the call moves straight on to the next key with the SAME model, so
quality doesn't drop because one key ran out. Server errors and timeouts are not the key's
fault: they are retried on the same key, then the next model.

Requests are paced by the per-model rate limiter shared by all keys. Daily budgets are
tracked per (key, model) in `llm/quota.py`, so an exhausted key is skipped before calling.
Gemini limits apply per project: keys of one project share one budget.

A model that answers 503 "overloaded" is unavailable for every key, so it is not a key
problem: `ModelBreaker` skips that model for a while instead of waiting on it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.exceptions import (
    ModelAuthenticationError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
)

PER_MINUTE_BENCH_S = 65.0
OVERLOAD_BENCH_S = 120.0  # skip an overloaded model this long, then probe it again
DAILY_BENCH_S = 3600.0  # daily quotas reset at midnight Pacific; recheck hourly
BROKEN_KEY_BENCH_S = 24 * 3600.0

_AUTH_MARKERS = (
    "API_KEY_INVALID",
    "API key not valid",
    "API key expired",
    "PERMISSION_DENIED",
    "SERVICE_DISABLED",
    "CONSUMER_SUSPENDED",
)


@dataclass(frozen=True)
class KeyProblem:
    seconds: float
    all_models: bool
    """True: the key itself is broken. False: only its quota for this model is used up —
    Gemini quotas are per key AND per model, so the key may still serve the fallback model."""
    daily: bool = False
    """The per-day budget is used up: the key is done with this model until the reset."""


def key_problem(exc: BaseException) -> KeyProblem | None:
    """How long to bench the key that raised `exc`; None if the key is not at fault."""
    text = f"{type(exc).__name__}: {exc}"
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    broken = isinstance(exc, ModelAuthenticationError | ModelPermissionDeniedError)
    # An invalid key comes back as HTTP 400 (ModelInvalidRequestError): recognised by text.
    if broken or status in (401, 403) or any(marker in text for marker in _AUTH_MARKERS):
        return KeyProblem(BROKEN_KEY_BENCH_S, all_models=True)
    if isinstance(exc, ModelRateLimitError) or status == 429 or "RESOURCE_EXHAUSTED" in text:
        daily = "PerDay" in text or "per day" in text.lower()
        return KeyProblem(
            DAILY_BENCH_S if daily else PER_MINUTE_BENCH_S, all_models=False, daily=daily
        )
    return None


_OVERLOAD_MARKERS = ("503 ", "UNAVAILABLE", "overloaded", "high demand")


def is_overloaded(exc: BaseException) -> bool:
    """Gemini's 503: the model itself is busy for everyone, whichever key asks."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    return status == 503 or any(marker in str(exc) for marker in _OVERLOAD_MARKERS)


class ModelBreaker:
    """Models answering "overloaded": tried last until OVERLOAD_BENCH_S passes. After that the
    next call probes the model again (and re-benches it if it is still busy)."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._until: dict[str, float] = {}
        self._clock = clock

    def busy(self, model: str) -> bool:
        return self._until.get(model, 0.0) > self._clock()

    def trip(self, model: str, seconds: float = OVERLOAD_BENCH_S) -> None:
        self._until[model] = self._clock() + seconds

    def order(self, models: list[str]) -> list[str]:
        """Healthy models first, in preference order; busy ones last (still tried if
        nothing else is left)."""
        return [m for m in models if not self.busy(m)] + [m for m in models if self.busy(m)]


class KeyPool:
    """Which keys may be used now for a model (in order of preference), and which are benched
    — per (key, model) for quota, per key for a broken key."""

    def __init__(self, size: int, *, clock: Callable[[], float] = time.monotonic) -> None:
        if size < 1:
            raise ValueError("at least one API key is required")
        self._size = size
        self._until: dict[tuple[int, str | None], float] = {}
        self._clock = clock

    @property
    def size(self) -> int:
        return self._size

    def available(self, model: str) -> list[int]:
        now = self._clock()
        return [
            i
            for i in range(self._size)
            if self._until.get((i, None), 0.0) <= now and self._until.get((i, model), 0.0) <= now
        ]

    def bench(self, key: int, model: str, problem: KeyProblem) -> None:
        slot = (key, None if problem.all_models else model)
        self._until[slot] = max(self._until.get(slot, 0.0), self._clock() + problem.seconds)
