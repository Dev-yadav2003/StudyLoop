from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from lecnotes.domain import Job, JobOptions, JobStatus
from lecnotes.storage.tables import JobRow


class JobRepository:
    """Maps domain `Job` <-> `jobs` rows. The caller owns the transaction (commit/rollback)."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, job: Job) -> None:
        self._session.add(
            JobRow(
                id=job.id,
                url=job.url,
                options=job.options.model_dump(mode="json"),
                status=job.status.value,
                error=job.error,
                attempts=job.attempts,
                created_at=job.created_at,
                updated_at=job.updated_at,
            )
        )
        await self._session.flush()

    async def get(self, job_id: UUID) -> Job | None:
        row = await self._session.get(JobRow, job_id)
        return None if row is None else _to_domain(row)


def _to_domain(row: JobRow) -> Job:
    return Job(
        id=row.id,
        url=row.url,
        options=JobOptions.model_validate(row.options),
        status=JobStatus(row.status),
        error=row.error,
        attempts=row.attempts,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )
