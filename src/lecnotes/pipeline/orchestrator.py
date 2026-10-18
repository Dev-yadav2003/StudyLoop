"""Runs stages: cache check → span → run → cache put → metrics. Nothing else.

The pipeline is a fixed linear chain per video (no branching, so no LangGraph): the orchestrator
is a loop. `rerun_from` forces a stage and everything after it to recompute, ignoring the
cache; outputs are re-cached under the same keys.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from lecnotes.cache.keys import StageKey
from lecnotes.domain import StageRun, StageRunStatus
from lecnotes.errors import LecNotesError
from lecnotes.observability.logging import get_logger
from lecnotes.observability.metrics import usage_totals
from lecnotes.observability.tracing import Span, traced_stage
from lecnotes.pipeline.context import RunContext
from lecnotes.pipeline.stage import Stage
from lecnotes.ports.cache import StageCache
from lecnotes.ports.metrics import StageRunSink

log = get_logger(__name__)

AnyStage = Stage[Any, Any]


class StageFailed(LecNotesError):
    """A stage raised. Carries the stage name; user message/retryability come from the cause."""

    def __init__(self, stage: str, cause: BaseException) -> None:
        super().__init__(f"stage {stage!r} failed: {type(cause).__name__}: {cause}")
        self.stage = stage
        self.cause = cause
        if isinstance(cause, LecNotesError):
            self.user_message = cause.user_message
            self.retryable = cause.retryable


class Orchestrator:
    def __init__(
        self,
        cache: StageCache,
        sink: StageRunSink,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._cache = cache
        self._sink = sink
        self._clock = clock

    async def run_chain(
        self,
        stages: Sequence[AnyStage],
        inp: BaseModel,
        ctx: RunContext,
        *,
        rerun_from: str | None = None,
    ) -> BaseModel:
        check_chain(stages, type(inp))
        names = [s.name for s in stages]
        if rerun_from is not None and rerun_from not in names:
            raise ValueError(f"rerun_from={rerun_from!r} is not one of {names}")
        force = False
        value = inp
        for stage in stages:
            force = force or stage.name == rerun_from
            value = await self.run_stage(stage, value, ctx, force=force)
        return value

    async def run_stage(
        self, stage: AnyStage, inp: BaseModel, ctx: RunContext, *, force: bool = False
    ) -> BaseModel:
        key = self._key(stage, inp, ctx)
        started_at, t0 = self._clock(), time.perf_counter()
        ctx.take_usage()  # usage belongs to the stage that reports it

        with traced_stage(
            stage.name,
            stage.version,
            prompt_version=stage.prompt_version,
            metadata={"cache_key": key.digest, "forced": force},
        ) as span:
            if stage.cacheable and not force:
                cached = await self._cache.get(key)
                if cached is not None:
                    out = stage.output_model.model_validate(cached)
                    span.update(metadata={"cache_hit": True})
                    await self._record(stage, ctx, span, key, started_at, t0, cache_hit=True)
                    log.info("stage.cache_hit")
                    return out

            log.info("stage.start", forced=force)
            try:
                out = await stage.run(inp, ctx)
                if not isinstance(out, stage.output_model):
                    raise TypeError(
                        f"{stage.name} returned {type(out).__name__}, "
                        f"declared {stage.output_model.__name__}"
                    )
            except Exception as exc:
                await self._record(stage, ctx, span, key, started_at, t0, error=exc)
                log.warning("stage.failed", error_type=type(exc).__name__, error=str(exc))
                raise StageFailed(stage.name, exc) from exc

            if stage.cacheable:
                await self._cache.put(key, out.model_dump(mode="json"))
            await self._record(stage, ctx, span, key, started_at, t0)
            log.info("stage.done")
            return out

    def _key(self, stage: AnyStage, inp: BaseModel, ctx: RunContext) -> StageKey:
        video = ctx.video
        return StageKey.build(
            stage=stage.name,
            stage_version=stage.version,
            video_id=video.video_id if video else None,
            prompt_version=stage.prompt_version,
            model=stage.model,
            input={"input": inp, "video": video},
            config=stage.config(ctx),
        )

    async def _record(
        self,
        stage: AnyStage,
        ctx: RunContext,
        span: Span,
        key: StageKey,
        started_at: datetime,
        t0: float,
        *,
        cache_hit: bool = False,
        error: BaseException | None = None,
    ) -> None:
        totals = usage_totals(ctx.take_usage())
        run = StageRun(
            job_id=ctx.job_id,
            video_id=ctx.video.video_id if ctx.video else None,
            stage=stage.name,
            stage_version=stage.version,
            prompt_version=stage.prompt_version,
            model=totals.model or stage.model,
            status=StageRunStatus.FAILED if error else StageRunStatus.OK,
            cache_hit=cache_hit,
            latency_ms=(time.perf_counter() - t0) * 1000,
            tokens_in=totals.tokens_in,
            tokens_out=totals.tokens_out,
            cost_usd=totals.cost_usd,
            error_type=type(error).__name__ if error else None,
            trace_id=span.trace_id,
            cache_key=key.digest,
            started_at=started_at,
        )
        span.update(
            metadata={
                "tokens_in": run.tokens_in,
                "tokens_out": run.tokens_out,
                "cost_usd": str(run.cost_usd),
            }
        )
        await self._sink.record(run)


def check_chain(stages: Sequence[AnyStage], input_type: type[BaseModel]) -> None:
    """Fail before running anything if stage types don't line up or names repeat."""
    names = [s.name for s in stages]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate stage names in chain: {names}")
    expected = input_type
    for stage in stages:
        if not issubclass(expected, stage.input_model):
            raise TypeError(
                f"{stage.name} expects {stage.input_model.__name__}, "
                f"but receives {expected.__name__}"
            )
        expected = stage.output_model
