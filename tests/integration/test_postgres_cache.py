from collections.abc import Awaitable, Callable

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from lecnotes.adapters.cache.postgres_cache import PostgresStageCache
from lecnotes.ports.cache import StageCache
from lecnotes.storage.tables import StageCacheRow
from tests.cache_contract import CONTRACT, make_key


@pytest.mark.parametrize("check", CONTRACT, ids=lambda f: f.__name__)
async def test_contract(
    engine: AsyncEngine, check: Callable[[StageCache], Awaitable[None]]
) -> None:
    await check(PostgresStageCache(engine))


async def test_row_metadata_and_hit_count(engine: AsyncEngine) -> None:
    cache = PostgresStageCache(engine)
    key = make_key(stage_version=3)
    await cache.get(key)  # miss: no row, nothing counted
    await cache.put(key, {"v": 1})
    await cache.get(key)
    await cache.get(key)

    async with engine.connect() as conn:
        stmt = select(StageCacheRow).where(StageCacheRow.key == key.digest)
        row = (await conn.execute(stmt)).one()
    assert (row.stage, row.stage_version, row.video_id) == ("extract", 3, "dQw4w9WgXcQ")
    assert row.hit_count == 2
    assert row.last_hit_at is not None
