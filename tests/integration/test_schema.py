from datetime import UTC, datetime
from uuid import uuid4

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from lecnotes.storage.db import normalize_url
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


async def test_db_rejects_unknown_status(session: AsyncSession) -> None:
    """The CHECK constraint guards data written outside the repository too."""
    with pytest.raises(IntegrityError, match="ck_jobs_status"):
        await session.execute(
            text(
                "INSERT INTO jobs (id, url, request_key, options, status) "
                "VALUES (:id, 'u', 'k', '{}', 'paused')"
            ),
            {"id": uuid4()},
        )
