"""Postgres job queue: the `jobs` table is the queue (no Redis, no extra service).

- `enqueue`: idempotent per (video, options) — an active job (queued/running/succeeded) with
  the same request key is returned instead of creating a new one; a concurrent duplicate
  insert loses on the partial unique index and also gets the existing job.
- `lease`: one atomic UPDATE … WHERE id = (SELECT … FOR UPDATE SKIP LOCKED LIMIT 1). Takes
  the oldest due queued job, or a running job whose lease expired (its worker died) — which
  is how a job killed mid-run resumes: the stage cache makes the finished stages free.
- `heartbeat` extends the lease, but only for the current owner.
- `fail`: retryable errors go back to `queued` with a backoff until `max_attempts`; anything
  else is `failed` with the user-facing message.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from lecnotes.cache.keys import canonical_json
from lecnotes.domain import Job, JobOptions, JobStatus
from lecnotes.storage.tables import ACTIVE_JOB_STATUSES, JobRow

_ACTIVE = [s.value for s in ACTIVE_JOB_STATUSES]


def request_key(video_id: str, options: JobOptions) -> str:
    return hashlib.sha256(canonical_json({"video": video_id, "options": options})).hexdigest()


def stored_options(raw: dict[str, Any]) -> JobOptions:
    """Options as stored with the job, minus fields since removed from `JobOptions` (rows
    written before the cleanup carry `generate_questions` and would otherwise fail to load)."""
    return JobOptions.model_validate({k: v for k, v in raw.items() if k in JobOptions.model_fields})


def to_job(row: Any) -> Job:
    return Job(
        id=row.id,
        url=row.url,
        video_id=row.video_id,
        options=stored_options(row.options),
        status=JobStatus(row.status),
        error=row.error,
        error_type=row.error_type,
        attempts=row.attempts,
        created_at=row.created_at,
        updated_at=row.updated_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
    )


@dataclass(frozen=True)
class Lease:
    job: Job
    owner: str
    rerun_from: str | None


class JobQueue:
    def __init__(
        self,
        engine: AsyncEngine,
        *,
        lease_s: float = 120,
        max_attempts: int = 3,
    ) -> None:
        self._engine = engine
        self.lease_s = lease_s
        self.max_attempts = max_attempts

    async def enqueue(self, url: str, video_id: str, options: JobOptions) -> tuple[Job, bool]:
        """(job, created). Returns the existing active job for the same video + options."""
        key = request_key(video_id, options)
        values = {
            "id": uuid.uuid4(),
            "url": url,
            "video_id": video_id,
            "request_key": key,
            "options": options.model_dump(mode="json"),
            "status": JobStatus.QUEUED.value,
        }
        stmt = (
            insert(JobRow)
            .values(values)
            .on_conflict_do_nothing(
                index_elements=[JobRow.request_key],
                index_where=JobRow.status.in_(_ACTIVE),
            )
            .returning(JobRow)
        )
        async with self._engine.begin() as conn:
            row = (await conn.execute(stmt)).one_or_none()
            if row is not None:
                return to_job(row), True
            existing = (
                await conn.execute(
                    select(JobRow).where(JobRow.request_key == key, JobRow.status.in_(_ACTIVE))
                )
            ).one()
            return to_job(existing), False

    async def get(self, job_id: uuid.UUID) -> Job | None:
        async with self._engine.connect() as conn:
            row = (await conn.execute(select(JobRow).where(JobRow.id == job_id))).one_or_none()
        return None if row is None else to_job(row)

    async def lease(self, owner: str) -> Lease | None:
        due = (
            select(JobRow.id)
            .where(
                ((JobRow.status == JobStatus.QUEUED.value) & (JobRow.run_after <= func.now()))
                | ((JobRow.status == JobStatus.RUNNING.value) & (JobRow.leased_until < func.now()))
            )
            .order_by(JobRow.created_at)
            .limit(1)
            .with_for_update(skip_locked=True)
            .scalar_subquery()
        )
        stmt = (
            update(JobRow)
            .where(JobRow.id == due)
            .values(
                status=JobStatus.RUNNING.value,
                lease_owner=owner,
                leased_until=func.now() + timedelta(seconds=self.lease_s),
                attempts=JobRow.attempts + 1,
                started_at=func.coalesce(JobRow.started_at, func.now()),
                updated_at=func.now(),
            )
            .returning(JobRow)
        )
        async with self._engine.begin() as conn:
            row = (await conn.execute(stmt)).one_or_none()
        if row is None:
            return None
        return Lease(job=to_job(row), owner=owner, rerun_from=row.rerun_from)

    async def heartbeat(self, lease: Lease) -> bool:
        """Extend the lease. False if another worker took the job (our lease had expired)."""
        stmt = (
            update(JobRow)
            .where(JobRow.id == lease.job.id, JobRow.lease_owner == lease.owner)
            .values(leased_until=func.now() + timedelta(seconds=self.lease_s))
        )
        async with self._engine.begin() as conn:
            return (await conn.execute(stmt)).rowcount == 1

    async def complete(self, lease: Lease) -> None:
        await self._finish(lease, status=JobStatus.SUCCEEDED, error=None, error_type=None)

    async def fail(
        self,
        lease: Lease,
        *,
        error_type: str,
        user_message: str,
        retryable: bool,
        retry_in_s: float = 60,
        count_attempt: bool = True,
    ) -> Job:
        if retryable and (not count_attempt or lease.job.attempts < self.max_attempts):
            stmt = (
                update(JobRow)
                .where(JobRow.id == lease.job.id, JobRow.lease_owner == lease.owner)
                .values(
                    status=JobStatus.QUEUED.value,
                    error=None,
                    error_type=error_type,
                    lease_owner=None,
                    leased_until=None,
                    run_after=func.now() + timedelta(seconds=retry_in_s),
                    updated_at=func.now(),
                    # Leasing counted this run; give it back when it doesn't count.
                    attempts=JobRow.attempts - (0 if count_attempt else 1),
                )
            )
            async with self._engine.begin() as conn:
                await conn.execute(stmt)
        else:
            await self._finish(
                lease, status=JobStatus.FAILED, error=user_message, error_type=error_type
            )
        job = await self.get(lease.job.id)
        assert job is not None
        return job

    async def request_rerun(self, job_id: uuid.UUID, from_stage: str | None) -> Job | None:
        """Re-queue a finished job (admin). Running jobs are left alone."""
        stmt = (
            update(JobRow)
            .where(JobRow.id == job_id, JobRow.status != JobStatus.RUNNING.value)
            .values(
                status=JobStatus.QUEUED.value,
                rerun_from=from_stage,
                error=None,
                attempts=0,
                run_after=func.now(),
                finished_at=None,
                updated_at=func.now(),
            )
            .returning(JobRow)
        )
        async with self._engine.begin() as conn:
            row = (await conn.execute(stmt)).one_or_none()
        return None if row is None else to_job(row)

    async def _finish(
        self, lease: Lease, *, status: JobStatus, error: str | None, error_type: str | None
    ) -> None:
        stmt = (
            update(JobRow)
            .where(JobRow.id == lease.job.id, JobRow.lease_owner == lease.owner)
            .values(
                status=status.value,
                error=error,
                error_type=error_type,
                lease_owner=None,
                leased_until=None,
                rerun_from=None,
                finished_at=func.now(),
                updated_at=func.now(),
            )
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)
