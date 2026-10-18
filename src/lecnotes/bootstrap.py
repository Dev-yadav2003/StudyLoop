"""Composition root: the only place that knows which concrete adapters are used.

The CLI (now) and the API (Phase 8) build their services here. Everything else depends on
ports, which is what keeps the pipeline testable offline.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from sqlalchemy.ext.asyncio import AsyncEngine

from lecnotes.adapters.cache.memory import InMemoryStageCache
from lecnotes.adapters.cache.postgres_cache import PostgresStageCache
from lecnotes.adapters.catalog.youtube_data_api import YouTubeDataApiCatalog
from lecnotes.adapters.perception.gemini_url import GeminiUrlPerceiver
from lecnotes.config import Settings
from lecnotes.jobs.queue import JobQueue
from lecnotes.jobs.runner import JobRunner
from lecnotes.llm.gateway import LLMGateway
from lecnotes.observability.metrics import InMemoryStageRunSink, TeeStageRunSink
from lecnotes.pipeline.notes_editor import NotesEditor
from lecnotes.pipeline.orchestrator import AnyStage, Orchestrator
from lecnotes.pipeline.qa_service import QaService
from lecnotes.pipeline.registry import Pipeline
from lecnotes.pipeline.stages.align.align import AlignStage
from lecnotes.pipeline.stages.align.chunk import ChunkStage
from lecnotes.pipeline.stages.extract import ExtractStage
from lecnotes.pipeline.stages.merge import MergeStage
from lecnotes.pipeline.stages.perceive import PerceiveStage
from lecnotes.pipeline.stages.qa import QaStage
from lecnotes.pipeline.stages.render import RenderStage
from lecnotes.pipeline.stages.resolve import ResolveStage
from lecnotes.pipeline.stages.validate import ValidateStage
from lecnotes.pipeline.stages.verify import VerifyStage
from lecnotes.ports.cache import StageCache
from lecnotes.ports.metrics import StageRunSink
from lecnotes.ports.perception import Perceiver
from lecnotes.ports.video_catalog import VideoCatalog
from lecnotes.storage.db import create_engine
from lecnotes.storage.repositories.feedback import FeedbackRepository
from lecnotes.storage.repositories.library import LibraryRepository
from lecnotes.storage.repositories.notes import NotesRepository
from lecnotes.storage.repositories.stage_runs import PostgresStageRunSink
from lecnotes.storage.repositories.usage import UsageRepository


@dataclass
class Services:
    pipeline: Pipeline
    qa: QaService
    editor: NotesEditor
    catalog: VideoCatalog
    """Resolves a link to its metadata (the API checks a video's length before queueing it)."""
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
    qa = QaService(
        orchestrator,
        QaStage(gateway, model=settings.text_model, prompt_version=settings.qa_prompt_version),
    )
    editor = NotesEditor(orchestrator, RenderStage())
    return Services(pipeline=pipeline, qa=qa, editor=editor, catalog=catalog, runs=runs)


@asynccontextmanager
async def production_services(
    settings: Settings, *, use_db: bool = True, engine: AsyncEngine | None = None
) -> AsyncIterator[Services]:
    """Real Gemini + YouTube Data API; Postgres cache/metrics unless `use_db=False`.
    Pass `engine` to share one connection pool (the API process does)."""
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
        async with _engine(settings, engine) as db:
            yield assemble(
                settings=settings,
                catalog=catalog,
                perceiver=perceiver,
                gateway=gateway,
                cache=PostgresStageCache(db),
                sinks=[PostgresStageRunSink(db)],
            )


@dataclass
class ApiContainer:
    """Everything the HTTP layer needs. Tests build one from fakes; production uses
    `api_container`."""

    settings: Settings
    engine: AsyncEngine
    queue: JobQueue
    notes: NotesRepository
    services: Services
    runner: JobRunner
    runner_task: asyncio.Task[None] | None = None

    @property
    def feedback(self) -> FeedbackRepository:
        return FeedbackRepository(self.engine)

    def library(self, user_id: uuid.UUID) -> LibraryRepository:
        """The library of one account; every query is scoped to it."""
        return LibraryRepository(self.engine, user_id)

    @property
    def usage(self) -> UsageRepository:
        return UsageRepository(self.engine)


@asynccontextmanager
async def api_container(settings: Settings) -> AsyncIterator[ApiContainer]:
    """Production wiring for the API process, incl. the in-process runner (if enabled).
    On shutdown an in-flight job is cancelled; its lease expires and it resumes later."""
    async with (
        _engine(settings) as engine,
        production_services(settings, engine=engine) as services,
    ):
        queue = JobQueue(
            engine, lease_s=settings.job_lease_s, max_attempts=settings.job_max_attempts
        )
        notes = NotesRepository(engine)
        runner = JobRunner(queue, services.pipeline, notes, poll_s=settings.worker_poll_s)
        container = ApiContainer(settings, engine, queue, notes, services, runner)
        stop = asyncio.Event()
        if settings.run_worker:
            container.runner_task = asyncio.create_task(runner.run_forever(stop))
        try:
            yield container
        finally:
            stop.set()
            if container.runner_task is not None:
                container.runner_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await container.runner_task


@asynccontextmanager
async def _engine(
    settings: Settings, existing: AsyncEngine | None = None
) -> AsyncIterator[AsyncEngine]:
    """An existing engine is borrowed (its owner disposes it); otherwise create and dispose."""
    if existing is not None:
        yield existing
        return
    engine = create_engine(settings.database_url.get_secret_value())
    try:
        yield engine
    finally:
        await engine.dispose()
