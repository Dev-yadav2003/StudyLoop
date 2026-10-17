"""In-process job runner (Render free = one small process: one job at a time).

Lease a job → heartbeat while the pipeline runs → save notes → complete. Typed errors carry
their user message and retryability; quota errors retry after an hour, transient ones after a
minute. Unexpected exceptions fail the job with a generic message and are logged in full.

Cancellation (process shutdown, Render restart) is deliberately NOT caught: the job stays
`running` with a lease that expires, and the next worker re-leases it and resumes from the
stage cache.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket

from lecnotes.errors import CatalogQuotaExceeded, LecNotesError, QuotaExceeded
from lecnotes.jobs.queue import JobQueue, Lease
from lecnotes.observability.logging import bind_context, get_logger
from lecnotes.pipeline.orchestrator import StageFailed
from lecnotes.pipeline.registry import Pipeline
from lecnotes.storage.repositories.notes import NotesRepository

log = get_logger(__name__)

GENERIC_FAILURE = LecNotesError.user_message  # same wording as failures inside a stage
GAVE_UP = "This video failed repeatedly, so we stopped retrying. Please try again later."


def default_owner() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


class JobRunner:
    def __init__(
        self,
        queue: JobQueue,
        pipeline: Pipeline,
        notes: NotesRepository,
        *,
        owner: str | None = None,
        poll_s: float = 2.0,
        heartbeat_s: float = 30.0,
        quota_retry_s: float = 3600.0,
        transient_retry_s: float = 60.0,
    ) -> None:
        self._queue = queue
        self._pipeline = pipeline
        self._notes = notes
        self.owner = owner or default_owner()
        self.poll_s = poll_s
        self.heartbeat_s = heartbeat_s
        self.quota_retry_s = quota_retry_s
        self.transient_retry_s = transient_retry_s

    async def run_forever(self, stop: asyncio.Event) -> None:
        log.info("runner.started", owner=self.owner)
        while not stop.is_set():
            try:
                worked = await self.run_once()
            except Exception:
                log.exception("runner.loop_error")  # e.g. DB briefly unreachable
                worked = False
            if not worked:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=self.poll_s)
        log.info("runner.stopped", owner=self.owner)

    async def run_once(self) -> bool:
        """Process at most one job. True if a job was leased."""
        lease = await self._queue.lease(self.owner)
        if lease is None:
            return False
        job = lease.job
        with bind_context(job_id=str(job.id), video_id=job.video_id):
            if job.attempts > self._queue.max_attempts:
                await self._queue.fail(
                    lease, error_type="GaveUp", user_message=GAVE_UP, retryable=False
                )
                log.warning("job.gave_up", attempts=job.attempts)
                return True
            heartbeat = asyncio.create_task(self._heartbeat(lease))
            try:
                await self._process(lease)
            finally:
                heartbeat.cancel()
        return True

    async def _process(self, lease: Lease) -> None:
        job = lease.job
        log.info("job.start", attempt=job.attempts, rerun_from=lease.rerun_from)
        try:
            doc = await self._pipeline.run(
                job.url, job_id=job.id, options=job.options, rerun_from=lease.rerun_from
            )
            await self._notes.save(job.id, doc)
            await self._queue.complete(lease)
            log.info("job.succeeded", units=len(doc.units))
        except LecNotesError as exc:
            cause = exc.cause if isinstance(exc, StageFailed) else exc
            quota = isinstance(cause, QuotaExceeded | CatalogQuotaExceeded)
            updated = await self._queue.fail(
                lease,
                error_type=type(cause).__name__,
                user_message=exc.user_message,
                retryable=exc.retryable,
                retry_in_s=self.quota_retry_s if quota else self.transient_retry_s,
            )
            log.warning("job.error", error_type=type(cause).__name__, status=updated.status)
        except Exception as exc:
            log.exception("job.unexpected_error")
            await self._queue.fail(
                lease,
                error_type=type(exc).__name__,
                user_message=GENERIC_FAILURE,
                retryable=False,
            )

    async def _heartbeat(self, lease: Lease) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_s)
            if not await self._queue.heartbeat(lease):
                log.warning("job.lease_lost")  # another worker took over after expiry
                return
