"""Postgres usage store: counts per (key, model, day), exhaustion, and learned limits."""

from datetime import date

from sqlalchemy.ext.asyncio import AsyncEngine

from lecnotes.domain import KeyUsage
from lecnotes.storage.repositories.key_usage import KeyUsageRepository

DAY, NEXT_DAY = date(2026, 10, 19), date(2026, 10, 20)
FLASH, LITE = "gemini-2.5-flash", "gemini-2.5-flash-lite"


async def test_requests_are_counted_per_key_model_and_day(engine: AsyncEngine) -> None:
    repo = KeyUsageRepository(engine)
    for _ in range(3):
        await repo.record(DAY, "k0", FLASH, tokens_in=100, tokens_out=10)
    await repo.record(DAY, "k1", FLASH, tokens_in=1, tokens_out=1)
    await repo.record(DAY, "k0", LITE, tokens_in=1, tokens_out=1)

    assert await repo.usage(DAY, FLASH) == {
        "k0": KeyUsage(requests=3),
        "k1": KeyUsage(requests=1),
    }
    assert await repo.usage(NEXT_DAY, FLASH) == {}  # a new Pacific day starts at zero


async def test_exhaustion_and_learned_limit(engine: AsyncEngine) -> None:
    repo = KeyUsageRepository(engine)
    await repo.record(DAY, "k0", FLASH, tokens_in=1, tokens_out=1)
    await repo.mark_exhausted(DAY, "k0", FLASH, limit=20)
    await repo.mark_exhausted(DAY, "k1", FLASH, limit=None)  # keeps the learned limit

    assert await repo.usage(DAY, FLASH) == {
        "k0": KeyUsage(requests=1, exhausted=True),
        "k1": KeyUsage(requests=0, exhausted=True),
    }
    assert await repo.daily_limit(FLASH) == 20
    assert await repo.daily_limit(LITE) is None
