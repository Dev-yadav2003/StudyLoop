from __future__ import annotations

from typing import Any

from sqlalchemy import func, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from lecnotes.cache.keys import StageKey
from lecnotes.storage.tables import StageCacheRow


class PostgresStageCache:
    """`stage_cache` table. Each call is its own short transaction, independent of the job's:
    a cached stage output stays valid even if the job later fails."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def get(self, key: StageKey) -> dict[str, Any] | None:
        # One round trip: count the hit and return the payload atomically.
        stmt = (
            update(StageCacheRow)
            .where(StageCacheRow.key == key.digest)
            .values(hit_count=StageCacheRow.hit_count + 1, last_hit_at=func.now())
            .returning(StageCacheRow.payload)
        )
        async with self._engine.begin() as conn:
            return (await conn.execute(stmt)).scalar_one_or_none()

    async def put(self, key: StageKey, payload: dict[str, Any]) -> None:
        values = {
            "key": key.digest,
            "stage": key.stage,
            "stage_version": key.stage_version,
            "prompt_version": key.prompt_version,
            "model": key.model,
            "video_id": key.video_id,
            "payload": payload,
        }
        stmt = insert(StageCacheRow).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[StageCacheRow.key],
            set_={"payload": stmt.excluded.payload, "created_at": func.now()},
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)
