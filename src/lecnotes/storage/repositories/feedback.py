"""Reader feedback on notes (step 9.3)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import case, func, insert, select
from sqlalchemy.ext.asyncio import AsyncEngine

from lecnotes.domain import Feedback
from lecnotes.storage.tables import FeedbackRow, NotesRow


@dataclass(frozen=True)
class StoredFeedback:
    id: int
    notes_id: uuid.UUID
    video_id: str
    title: str
    feedback: Feedback
    trace_id: str | None
    created_at: datetime


@dataclass(frozen=True)
class FeedbackSummary:
    up: int
    down: int


class FeedbackRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def add(
        self, notes_id: uuid.UUID, feedback: Feedback, *, trace_id: str | None
    ) -> tuple[int, datetime]:
        async with self._engine.begin() as conn:
            row = (
                await conn.execute(
                    insert(FeedbackRow)
                    .values(
                        notes_id=notes_id,
                        unit_key=feedback.unit_id,
                        rating=feedback.rating,
                        reason=feedback.reason,
                        trace_id=trace_id,
                    )
                    .returning(FeedbackRow.id, FeedbackRow.created_at)
                )
            ).one()
        return row.id, row.created_at

    async def recent(self, *, limit: int = 50, rating: int | None = None) -> list[StoredFeedback]:
        query = (
            select(
                FeedbackRow.id,
                FeedbackRow.notes_id,
                FeedbackRow.unit_key,
                FeedbackRow.rating,
                FeedbackRow.reason,
                FeedbackRow.trace_id,
                FeedbackRow.created_at,
                NotesRow.video,
                NotesRow.title,
            )
            .join(NotesRow, NotesRow.id == FeedbackRow.notes_id)
            .order_by(FeedbackRow.created_at.desc(), FeedbackRow.id.desc())
            .limit(limit)
        )
        if rating is not None:
            query = query.where(FeedbackRow.rating == rating)
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query)).all()
        return [
            StoredFeedback(
                id=r.id,
                notes_id=r.notes_id,
                video_id=r.video["video_id"],
                title=r.title,
                feedback=Feedback(rating=r.rating, reason=r.reason, unit_id=r.unit_key),
                trace_id=r.trace_id,
                created_at=r.created_at,
            )
            for r in rows
        ]

    async def summary(self) -> FeedbackSummary:
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    select(
                        func.count(case((FeedbackRow.rating == 1, 1))),
                        func.count(case((FeedbackRow.rating == -1, 1))),
                    )
                )
            ).one()
        return FeedbackSummary(up=row[0], down=row[1])
