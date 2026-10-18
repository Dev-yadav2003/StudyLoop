"""What each account has used: video minutes and LLM tokens (a ledger, one row per charge)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import func, insert, select
from sqlalchemy.ext.asyncio import AsyncEngine

from lecnotes.storage.tables import UsageRow

Kind = Literal["video", "qa"]


@dataclass(frozen=True)
class UsageTotals:
    video_minutes: float
    tokens: int


class UsageRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def totals(self, user_id: uuid.UUID) -> UsageTotals:
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    select(
                        func.coalesce(func.sum(UsageRow.video_minutes), 0.0),
                        func.coalesce(func.sum(UsageRow.tokens), 0),
                    ).where(UsageRow.user_id == user_id)
                )
            ).one()
        return UsageTotals(video_minutes=float(row[0]), tokens=int(row[1]))

    async def has_video(self, user_id: uuid.UUID, video_id: str) -> bool:
        """A video is charged once per account (adding it to another chapter is free)."""
        async with self._engine.connect() as conn:
            found = await conn.execute(
                select(UsageRow.id)
                .where(
                    UsageRow.user_id == user_id,
                    UsageRow.kind == "video",
                    UsageRow.ref == video_id,
                )
                .limit(1)
            )
            return found.first() is not None

    async def charge(
        self, user_id: uuid.UUID, kind: Kind, ref: str, *, video_minutes: float = 0, tokens: int = 0
    ) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                insert(UsageRow).values(
                    user_id=user_id, kind=kind, ref=ref, video_minutes=video_minutes, tokens=tokens
                )
            )
