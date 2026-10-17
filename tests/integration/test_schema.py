from datetime import UTC, datetime
from uuid import uuid4

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from lecnotes.domain import Job, JobOptions, JobStatus, SubjectPreset
from lecnotes.storage.db import normalize_url
from lecnotes.storage.repositories.jobs import JobRepository
from lecnotes.storage.tables import Base
from tests.integration.conftest import alembic_config, create_database

NOW = datetime(2026, 10, 17, 12, 0, tzinfo=UTC)


def test_migrations_match_models_and_round_trip(server_url: str) -> None:
    """Schema is created by migration, not by hand — and the migration matches the models."""
    url = create_database(server_url)
    cfg = alembic_config(url)
    command.upgrade(cfg, "head")

    sync_engine = create_engine(normalize_url(url))
    try:
        with sync_engine.connect() as conn:
            ctx = MigrationContext.configure(conn, opts={"compare_type": True})
            assert compare_metadata(ctx, Base.metadata) == []

        command.downgrade(cfg, "base")
        with sync_engine.connect() as conn:
            tables = conn.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            ).scalars()
            assert set(tables) <= {"alembic_version"}
        command.upgrade(cfg, "head")
    finally:
        sync_engine.dispose()


async def test_insert_and_read_job(session: AsyncSession) -> None:
    job = Job(
        id=uuid4(),
        url="https://youtu.be/dQw4w9WgXcQ",
        options=JobOptions(subject_preset=SubjectPreset.STEM, generate_questions=True),
        created_at=NOW,
        updated_at=NOW,
    )
    repo = JobRepository(session)
    await repo.add(job)
    await session.commit()
    session.expunge_all()  # force a real round trip

    loaded = await repo.get(job.id)

    assert loaded == job
    assert await repo.get(uuid4()) is None


async def test_failed_job_round_trips_with_error(session: AsyncSession) -> None:
    job = Job(
        id=uuid4(),
        url="u",
        status=JobStatus.FAILED,
        error="Video is private",
        attempts=2,
        created_at=NOW,
        updated_at=NOW,
    )
    repo = JobRepository(session)
    await repo.add(job)
    await session.commit()
    session.expunge_all()

    assert await repo.get(job.id) == job


async def test_db_rejects_unknown_status(session: AsyncSession) -> None:
    """The CHECK constraint guards data written outside the repository too."""
    with pytest.raises(IntegrityError, match="ck_jobs_status"):
        await session.execute(
            text("INSERT INTO jobs (id, url, options, status) VALUES (:id, 'u', '{}', 'paused')"),
            {"id": uuid4()},
        )
