"""Daily request budgets per (API key, model): count them, learn the limit, plan around it.

Gemini's free tier caps requests per day per project and per model (measured 2026-10-19:
20/day for gemini-2.5-flash, from the 429's `quotaValue`), and resets at midnight Pacific.
The gateway asks the store which keys still have budget BEFORE calling, picks the least-used
one, and records every request that reached the model. A 429 remains the source of truth:
it marks the key exhausted for the rest of the Pacific day and teaches the store the limit.

Keys are stored only as a short hash (`key_id`), never in clear.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime, timedelta, timezone, tzinfo
from typing import Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from lecnotes.domain import KeyUsage

# Seconds added after the reset before a waiting job retries (clock skew, slow rollover).
RESET_MARGIN_S = 300.0


def _pacific() -> tzinfo:
    try:
        return ZoneInfo("America/Los_Angeles")
    except ZoneInfoNotFoundError:  # no tz database: PST is the conservative (later) reset
        return timezone(timedelta(hours=-8), "PST")


PACIFIC = _pacific()


def quota_day(now: datetime) -> date:
    """The quota day a moment belongs to: Gemini's daily limits reset at midnight Pacific."""
    return now.astimezone(PACIFIC).date()


def seconds_until_reset(now: datetime) -> float:
    local = now.astimezone(PACIFIC)
    midnight = datetime.combine(local.date() + timedelta(days=1), datetime.min.time(), PACIFIC)
    return (midnight - local).total_seconds()


def key_id(key: str) -> str:
    """Stable, non-reversible name for an API key (reordering keys keeps their history)."""
    return hashlib.sha256(key.encode()).hexdigest()[:12]


_QUOTA_VALUE = re.compile(r"quota_?value['\"]?\s*[:=]\s*['\"]?(\d+)", re.IGNORECASE)


def daily_limit_from(error_text: str) -> int | None:
    """The per-day request limit a 429 reports (`quotaValue`), if it states one."""
    if "PerDay" not in error_text and "per day" not in error_text.lower():
        return None
    match = _QUOTA_VALUE.search(error_text)
    return int(match.group(1)) if match else None


class UsageStore(Protocol):
    """Where request counts live. Postgres in production (survives restarts and is shared by
    workers), memory in tests and `--no-db` runs."""

    async def usage(self, day: date, model: str) -> dict[str, KeyUsage]: ...

    async def record(
        self, day: date, key: str, model: str, *, tokens_in: int, tokens_out: int
    ) -> None: ...

    async def mark_exhausted(self, day: date, key: str, model: str, limit: int | None) -> None: ...

    async def daily_limit(self, model: str) -> int | None: ...


class MemoryUsageStore:
    def __init__(self) -> None:
        self._rows: dict[tuple[date, str, str], KeyUsage] = {}
        self._limits: dict[str, int] = {}

    async def usage(self, day: date, model: str) -> dict[str, KeyUsage]:
        return {k: u for (d, k, m), u in self._rows.items() if d == day and m == model}

    async def record(
        self, day: date, key: str, model: str, *, tokens_in: int, tokens_out: int
    ) -> None:
        old = self._rows.get((day, key, model), KeyUsage())
        self._rows[(day, key, model)] = old.model_copy(update={"requests": old.requests + 1})

    async def mark_exhausted(self, day: date, key: str, model: str, limit: int | None) -> None:
        old = self._rows.get((day, key, model), KeyUsage())
        self._rows[(day, key, model)] = old.model_copy(update={"exhausted": True})
        if limit is not None:
            self._limits[model] = limit

    async def daily_limit(self, model: str) -> int | None:
        return self._limits.get(model)
