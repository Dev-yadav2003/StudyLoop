import asyncio
import threading

import pytest
from langchain_core.exceptions import ModelRateLimitError
from langchain_core.messages import BaseMessage
from sqlalchemy.ext.asyncio import AsyncEngine

from lecnotes.adapters.cache.postgres_cache import PostgresStageCache
from lecnotes.adapters.llm.fake import FakeChatModel, FakeLLM, Scripted
from lecnotes.domain import JobOptions, JobStatus
from lecnotes.jobs.queue import JobQueue
from lecnotes.jobs.runner import GENERIC_FAILURE, JobRunner
from lecnotes.storage.repositories.notes import NotesRepository
from tests.pipeline_fakes import URL, VID, default_script, fake_services, make_settings


def runner_for(
    engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
    *,
    llm: FakeLLM | None = None,
    script: dict | None = None,
    owner: str = "w1",
    lease_s: float = 120,
):
    services, fake = fake_services(
        make_settings(monkeypatch), PostgresStageCache(engine), llm=llm, script=script
    )
    queue = JobQueue(engine, lease_s=lease_s)
    runner = JobRunner(queue, services.pipeline, NotesRepository(engine), owner=owner)
    return runner, services, queue, fake


async def test_job_succeeds_and_notes_are_stored(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, _, queue, _ = runner_for(engine, monkeypatch)
    job, _ = await queue.enqueue(URL, VID, JobOptions())

    assert await runner.run_once() is True
    assert await runner.run_once() is False  # queue empty

    done = await queue.get(job.id)
    assert done is not None
    assert done.status is JobStatus.SUCCEEDED
    stored = await NotesRepository(engine).get_by_job(job.id)
    assert stored is not None
    doc = stored.document
    assert doc.video.video_id == VID
    assert [u.concept for u in doc.units] == ["enumerate()"]
    assert "### enumerate()" in doc.markdown
    assert doc.evidence is not None
    assert (await NotesRepository(engine).get(stored.id)) == stored


async def test_unavailable_video_fails_with_user_message(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, _, queue, _ = runner_for(engine, monkeypatch)
    job, _ = await queue.enqueue("https://youtu.be/aaaaaaaaaaa", "aaaaaaaaaaa", JobOptions())

    await runner.run_once()

    failed = await queue.get(job.id)
    assert failed is not None
    assert failed.status is JobStatus.FAILED
    assert failed.error_type == "VideoUnavailable"
    assert failed.error is not None
    assert "public videos" in failed.error


async def test_quota_error_requeues_for_later(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = default_script() | {"extract": ModelRateLimitError("429")}
    runner, _, queue, _ = runner_for(engine, monkeypatch, script=script)
    job, _ = await queue.enqueue(URL, VID, JobOptions())

    await runner.run_once()

    waiting = await queue.get(job.id)
    assert waiting is not None
    assert (waiting.status, waiting.error_type) == (JobStatus.QUEUED, "QuotaExceeded")
    assert await runner.run_once() is False  # backoff: not due yet


async def test_unexpected_exception_fails_with_generic_message(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = default_script() | {"extract": RuntimeError("bug")}
    runner, _, queue, _ = runner_for(engine, monkeypatch, script=script)
    job, _ = await queue.enqueue(URL, VID, JobOptions())

    await runner.run_once()

    failed = await queue.get(job.id)
    assert failed is not None
    assert failed.status is JobStatus.FAILED
    assert failed.error == GENERIC_FAILURE


class BlockingLLM(FakeLLM):
    """Signals when extract starts, then blocks — so the test can 'kill' the worker there."""

    def __init__(self) -> None:
        super().__init__(default_script())
        self.started = threading.Event()
        self.release = threading.Event()

    def next(self, prompt_id: str, model: FakeChatModel, messages: list[BaseMessage]) -> Scripted:
        if prompt_id == "extract":
            self.started.set()
            self.release.wait(timeout=10)
        return super().next(prompt_id, model, messages)


async def test_job_killed_mid_run_resumes_from_cache(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocking = BlockingLLM()
    dying, _, queue, _ = runner_for(engine, monkeypatch, llm=blocking, owner="dying", lease_s=0.5)
    job, _ = await queue.enqueue(URL, VID, JobOptions())

    task = asyncio.create_task(dying.run_once())
    # perceive → chunk are done and cached; extract is now running
    assert await asyncio.to_thread(blocking.started.wait, 10)
    task.cancel()  # the process dies
    with pytest.raises(asyncio.CancelledError):
        await task
    blocking.release.set()

    stuck = await queue.get(job.id)
    assert stuck is not None
    assert stuck.status is JobStatus.RUNNING  # nobody marked it done or failed

    await asyncio.sleep(0.7)  # lease expires
    survivor, services, _, _ = runner_for(engine, monkeypatch, owner="survivor")
    assert await survivor.run_once() is True

    done = await queue.get(job.id)
    assert done is not None
    assert done.status is JobStatus.SUCCEEDED
    assert done.attempts == 2
    hits = {r.stage: r.cache_hit for r in services.runs.runs}
    for stage in ("perceive", "validate", "align", "chunk"):
        assert hits[stage], stage
    assert hits["extract"] is False  # resumed exactly where it died
