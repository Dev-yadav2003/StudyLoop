"""`stage_runs`: permanent per-stage history (Langfuse Hobby keeps only 30 days)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func, insert, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine

from lecnotes.domain import StageRun
from lecnotes.storage.tables import StageRunRow

# stdlib logger: rendered as JSON with bound job/stage context by observability.logging.
log = logging.getLogger(__name__)


class PostgresStageRunSink:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def record(self, run: StageRun) -> None:
        values = run.model_dump(mode="python")
        values["status"] = run.status.value
        try:
            async with self._engine.begin() as conn:
                await conn.execute(insert(StageRunRow).values(values))
        except SQLAlchemyError:
            # Never fail a job because a metrics row couldn't be written.
            log.exception("stage_run.record_failed stage=%s job_id=%s", run.stage, run.job_id)


@dataclass(frozen=True)
class StageLatency:
    stage: str
    stage_version: int
    runs: int
    p50_ms: float
    p95_ms: float


async def latency_percentiles(
    engine: AsyncEngine,
    *,
    since: datetime | None = None,
    include_cache_hits: bool = False,
) -> list[StageLatency]:
    """p50/p95 latency per (stage, version). Cache hits are excluded by default: a 2 ms hit
    would make perceive's median meaningless."""
    p50 = func.percentile_cont(0.5).within_group(StageRunRow.latency_ms)
    p95 = func.percentile_cont(0.95).within_group(StageRunRow.latency_ms)
    stmt = (
        select(
            StageRunRow.stage,
            StageRunRow.stage_version,
            func.count().label("runs"),
            p50.label("p50"),
            p95.label("p95"),
        )
        .group_by(StageRunRow.stage, StageRunRow.stage_version)
        .order_by(StageRunRow.stage, StageRunRow.stage_version)
    )
    if not include_cache_hits:
        stmt = stmt.where(StageRunRow.cache_hit.is_(False))
    if since is not None:
        stmt = stmt.where(StageRunRow.started_at >= since)
    async with engine.connect() as conn:
        rows = (await conn.execute(stmt)).all()
    return [
        StageLatency(
            stage=r.stage,
            stage_version=r.stage_version,
            runs=r.runs,
            p50_ms=float(r.p50),
            p95_ms=float(r.p95),
        )
        for r in rows
    ]
