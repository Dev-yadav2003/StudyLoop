"""Database engine + sessions (async SQLAlchemy over psycopg 3).

Sized for Render free (512 MB) + Supabase free: a tiny pool. Prepared statements are
disabled because Supabase's pooler (pgbouncer, transaction mode) cannot route them.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine as _create_async_engine

_DRIVER = "postgresql+psycopg"


def normalize_url(url: str) -> str:
    """Supabase/Heroku-style `postgres://` or `postgresql://` → SQLAlchemy psycopg 3 URL."""
    for prefix in ("postgresql+psycopg://", "postgresql://", "postgres://"):
        if url.startswith(prefix):
            return f"{_DRIVER}://{url.removeprefix(prefix)}"
    raise ValueError("DATABASE_URL must be a postgres:// or postgresql:// URL")


def create_engine(url: str, *, pool_size: int = 3, connect_timeout_s: int = 10) -> AsyncEngine:
    return _create_async_engine(
        normalize_url(url),
        pool_size=pool_size,
        max_overflow=2,
        pool_pre_ping=True,  # free Supabase drops idle connections
        connect_args={
            "prepare_threshold": None,
            # Without it an unreachable DB blocks for minutes (seen: 130 s on Windows) — and
            # would stall a job just to write a metrics row.
            "connect_timeout": connect_timeout_s,
        },
    )


def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


MIGRATIONS = Path(__file__).parent / "migrations"


@cache
def code_schema_revision() -> str:
    """The newest migration this code expects (Alembic head)."""
    head = ScriptDirectory(str(MIGRATIONS)).get_current_head()
    if head is None:
        raise RuntimeError(f"no migrations found in {MIGRATIONS}")
    return head


async def db_schema_revision(conn: AsyncConnection) -> str | None:
    """The migration the database is at (None if it was never migrated)."""
    try:
        return (await conn.execute(text("select version_num from alembic_version"))).scalar()
    except ProgrammingError:  # no alembic_version table
        return None
