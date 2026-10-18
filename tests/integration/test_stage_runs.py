import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from lecnotes.domain import StageRun, StageRunStatus
from lecnotes.storage.db import create_engine
from lecnotes.storage.repositories.stage_runs import PostgresStageRunSink, latency_percentiles
from lecnotes.storage.tables import StageRunRow
from tests.postgres_server import _free_port

T0 = datetime(2026, 10, 17, 12, 0, tzinfo=UTC)
JOB = uuid4()


def run(
    stage: str,
    latency_ms: float,
    *,
    version: int = 1,
    cache_hit: bool = False,
    started_at: datetime = T0,
    **extra: object,
) -> StageRun:
    return StageRun.model_validate(
        {
            "job_id": JOB,
            "stage": stage,
            "stage_version": version,
            "status": StageRunStatus.OK,
            "cache_hit": cache_hit,
            "latency_ms": latency_ms,
            "started_at": started_at,
        }
        | extra
    )


async def test_record_writes_a_row(engine: AsyncEngine) -> None:
    sink = PostgresStageRunSink(engine)
    failed = run(
        "perceive",
        61_234.5,
        status=StageRunStatus.FAILED,
        error_type="QuotaExceeded",
        video_id="dQw4w9WgXcQ",
        prompt_version=2,
        model="gemini-2.5-flash",
        tokens_in=51_234,
        tokens_out=3_210,
        cost_usd=Decimal("0.023395"),
        trace_id="abc123",
    )
    await sink.record(failed)

    async with engine.connect() as conn:
        row = (await conn.execute(select(StageRunRow))).one()
    assert row.status == "failed"
    assert row.error_type == "QuotaExceeded"
    assert row.cost_usd == Decimal("0.023395")  # NUMERIC: exact, no float drift
    assert (row.tokens_in, row.tokens_out, row.prompt_version) == (51_234, 3_210, 2)
    assert row.started_at == T0


async def test_p50_p95_per_stage(engine: AsyncEngine) -> None:
    sink = PostgresStageRunSink(engine)
    for ms in range(1, 101):  # perceive v1: 1..100 ms
        await sink.record(run("perceive", ms))
    for ms in (10, 20, 30, 40):  # extract v1
        await sink.record(run("extract", ms))
    for ms in (400, 600):  # extract v2: versions are reported separately
        await sink.record(run("extract", ms, version=2))
    for _ in range(50):  # cache hits must not drag the median down
        await sink.record(run("perceive", 2, cache_hit=True))

    stats = {(s.stage, s.stage_version): s for s in await latency_percentiles(engine)}

    # percentile_cont interpolates: p50 of 1..100 = 50.5; p95 at rank 0.95*99 = 94.05 → 95.05
    perceive = stats[("perceive", 1)]
    assert perceive.runs == 100
    assert perceive.p50_ms == pytest.approx(50.5)
    assert perceive.p95_ms == pytest.approx(95.05)
    # [10,20,30,40]: p50 = 25 ; p95 at rank 2.85 → 30 + 0.85*10 = 38.5
    assert stats[("extract", 1)].p50_ms == pytest.approx(25)
    assert stats[("extract", 1)].p95_ms == pytest.approx(38.5)
    assert stats[("extract", 2)].p50_ms == pytest.approx(500)

    with_hits = {
        (s.stage, s.stage_version): s
        for s in await latency_percentiles(engine, include_cache_hits=True)
    }
    assert with_hits[("perceive", 1)].runs == 150


async def test_since_filter(engine: AsyncEngine) -> None:
    sink = PostgresStageRunSink(engine)
    await sink.record(run("render", 900, started_at=T0 - timedelta(days=40)))
    await sink.record(run("render", 5))

    [render] = await latency_percentiles(engine, since=T0 - timedelta(days=1))
    assert (render.runs, render.p50_ms) == (1, 5)


async def test_sink_never_raises(caplog: pytest.LogCaptureFixture) -> None:
    dead = create_engine(
        f"postgresql+psycopg://postgres@127.0.0.1:{_free_port()}/none", connect_timeout_s=1
    )
    try:
        with caplog.at_level(logging.ERROR):
            await PostgresStageRunSink(dead).record(run("extract", 1))  # must not raise
    finally:
        await dead.dispose()
    assert "stage_run.record_failed" in caplog.text
