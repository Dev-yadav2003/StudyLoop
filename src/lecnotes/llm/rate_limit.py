"""Process-wide token-bucket rate limiters, one per (model, RPM).

The Render worker runs one job at a time in one process, so in-memory limiting is enough.
Bucket size 1 = no bursts: requests are spaced evenly, which is what free-tier RPM wants.
"""

from __future__ import annotations

from functools import cache

from langchain_core.rate_limiters import InMemoryRateLimiter

from lecnotes.llm.models import get_model


def requests_per_minute(model: str, override: float | None = None) -> float:
    return override if override is not None else get_model(model).requests_per_minute


@cache
def rate_limiter(model: str, rpm: float) -> InMemoryRateLimiter:
    """Same arguments → same limiter object, so every caller shares one bucket per model."""
    return InMemoryRateLimiter(
        requests_per_second=rpm / 60,
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )
