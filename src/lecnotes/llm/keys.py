"""API-key failover: several Gemini keys, used one at a time.

A key is benched only for problems that are the key's own — invalid/revoked/forbidden, or
its quota exhausted — and the call moves straight on to the next key with the SAME model, so
quality doesn't drop because one key ran out. Server errors and timeouts are not the key's
fault: they are retried on the same key, then the next model.

This is failover, not throughput multiplication: requests are still paced by the per-model
rate limiter shared by all keys. (Rotating free-tier keys to get around quotas would break
Google's Gemini API terms.)
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
        return KeyProblem(DAILY_BENCH_S if daily else PER_MINUTE_BENCH_S, all_models=False)
    return None


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
