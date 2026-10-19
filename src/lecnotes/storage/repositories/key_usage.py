"""Gemini requests per (API key hash, model, Pacific day): the gateway's `UsageStore`.

Shared by every worker and kept across restarts, so a key that ran out today is skipped
without a wasted request after a redeploy. Counting is a single upsert per request.
"""

from __future__ import annotations

from datetime import date

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from lecnotes.domain import KeyUsage
from lecnotes.storage.tables import ApiKeyUsageRow


class KeyUsageRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def usage(self, day: date, model: str) -> dict[str, KeyUsage]:
        stmt = select(
            ApiKeyUsageRow.key_id, ApiKeyUsageRow.requests, ApiKeyUsageRow.exhausted
        ).where(ApiKeyUsageRow.day == day, ApiKeyUsageRow.model == model)
        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).all()
        return {k: KeyUsage(requests=r, exhausted=e) for k, r, e in rows}

    async def record(
        self, day: date, key: str, model: str, *, tokens_in: int, tokens_out: int
    ) -> None:
        stmt = insert(ApiKeyUsageRow).values(
            key_id=key, model=model, day=day, requests=1, tokens_in=tokens_in, tokens_out=tokens_out
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[ApiKeyUsageRow.key_id, ApiKeyUsageRow.model, ApiKeyUsageRow.day],
            set_={
                "requests": ApiKeyUsageRow.requests + 1,
                "tokens_in": ApiKeyUsageRow.tokens_in + tokens_in,
                "tokens_out": ApiKeyUsageRow.tokens_out + tokens_out,
                "updated_at": func.now(),
            },
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)

    async def mark_exhausted(self, day: date, key: str, model: str, limit: int | None) -> None:
        stmt = insert(ApiKeyUsageRow).values(
            key_id=key, model=model, day=day, exhausted=True, daily_limit=limit
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[ApiKeyUsageRow.key_id, ApiKeyUsageRow.model, ApiKeyUsageRow.day],
            set_={
                "exhausted": True,
                "daily_limit": func.coalesce(stmt.excluded.daily_limit, ApiKeyUsageRow.daily_limit),
                "updated_at": func.now(),
            },
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)

    async def daily_limit(self, model: str) -> int | None:
        """The most recent per-day limit a 429 reported for `model`."""
        stmt = (
            select(ApiKeyUsageRow.daily_limit)
            .where(ApiKeyUsageRow.model == model, ApiKeyUsageRow.daily_limit.is_not(None))
            .order_by(ApiKeyUsageRow.updated_at.desc())
            .limit(1)
        )
        async with self._engine.connect() as conn:
            return (await conn.execute(stmt)).scalar_one_or_none()
