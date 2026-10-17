"""Composition root: the only place that knows which concrete adapters are used.

The CLI (now) and the API (Phase 8) build their services here. Everything else depends on
ports, which is what keeps the pipeline testable offline.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx

from lecnotes.adapters.cache.memory import InMemoryStageCache
from lecnotes.adapters.cache.postgres_cache import PostgresStageCache
from lecnotes.adapters.catalog.youtube_data_api import YouTubeDataApiCatalog
from lecnotes.adapters.perception.gemini_url import GeminiUrlPerceiver
from lecnotes.config import Settings
from lecnotes.llm.gateway import LLMGateway
from lecnotes.observability.metrics import InMemoryStageRunSink, TeeStageRunSink
from lecnotes.pipeline.orchestrator import AnyStage, Orchestrator
from lecnotes.pipeline.registry import Pipeline
from lecnotes.pipeline.stages.align.align import AlignStage
from lecnotes.pipeline.stages.align.chunk import ChunkStage
from lecnotes.pipeline.stages.extract import ExtractStage
from lecnotes.pipeline.stages.merge import MergeStage
from lecnotes.pipeline.stages.perceive import PerceiveStage
from lecnotes.pipeline.stages.render import RenderStage
from lecnotes.pipeline.stages.resolve import ResolveStage
from lecnotes.pipeline.stages.validate import ValidateStage
from lecnotes.pipeline.stages.verify import VerifyStage
from lecnotes.ports.cache import StageCache
from lecnotes.ports.metrics import StageRunSink
from lecnotes.ports.perception import Perceiver
from lecnotes.ports.video_catalog import VideoCatalog
from lecnotes.storage.db import create_engine
from lecnotes.storage.repositories.stage_runs import PostgresStageRunSink


@dataclass
class Services:
    pipeline: Pipeline
    runs: InMemoryStageRunSink
    """In-memory copy of every StageRun of this process (CLI summary, debugging)."""


def video_stages(settings: Settings, perceiver: Perceiver, gateway: LLMGateway) -> list[AnyStage]:
    stages: list[AnyStage] = [
        PerceiveStage(perceiver),
        ValidateStage(),
        AlignStage(),
        ChunkStage(min_s=settings.chunk_min_s, max_s=settings.chunk_max_s),
        ExtractStage(
            gateway, model=settings.text_model, prompt_version=settings.extract_prompt_version
        ),
    ]
    if settings.features.merge:
        stages.append(
            MergeStage(
                gateway, model=settings.text_model, prompt_version=settings.merge_prompt_version
            )
        )
    if settings.features.verify:
        stages.append(
            VerifyStage(
                gateway, model=settings.text_model, prompt_version=settings.verify_prompt_version
            )
        )
    return [*stages, RenderStage()]


def assemble(
    *,
    settings: Settings,
    catalog: VideoCatalog,
    perceiver: Perceiver,
    gateway: LLMGateway,
    cache: StageCache,
    sinks: Sequence[StageRunSink] = (),
) -> Services:
    runs = InMemoryStageRunSink()
    orchestrator = Orchestrator(cache, TeeStageRunSink(*sinks, runs))
    pipeline = Pipeline(
        orchestrator, ResolveStage(catalog), video_stages(settings, perceiver, gateway)
    )
    return Services(pipeline=pipeline, runs=runs)


@asynccontextmanager
async def production_services(
    settings: Settings, *, use_db: bool = True
) -> AsyncIterator[Services]:
    """Real Gemini + YouTube Data API; Postgres cache/metrics unless `use_db=False`."""
    gateway = LLMGateway.from_settings(settings)
    async with httpx.AsyncClient(timeout=20) as http:
        catalog = YouTubeDataApiCatalog(http, settings.youtube_api_key.get_secret_value())
        perceiver = GeminiUrlPerceiver.from_settings(settings, gateway)
        if not use_db:
            yield assemble(
                settings=settings,
                catalog=catalog,
                perceiver=perceiver,
                gateway=gateway,
                cache=InMemoryStageCache(),
            )
            return
        engine = create_engine(settings.database_url.get_secret_value())
        try:
            yield assemble(
                settings=settings,
                catalog=catalog,
                perceiver=perceiver,
                gateway=gateway,
                cache=PostgresStageCache(engine),
                sinks=[PostgresStageRunSink(engine)],
            )
        finally:
            await engine.dispose()
