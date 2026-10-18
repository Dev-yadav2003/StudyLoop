"""Integration tests run against a real PostgreSQL (see tests/postgres_server.py)."""

from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from dataclasses import dataclass

import httpx
import jwt
import psycopg
import pytest
from alembic import command
from alembic.config import Config
from pydantic import SecretStr
from sqlalchemy import make_url, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from apps.api.app import create_app
from apps.api.auth import StaticKeyAuthenticator
from lecnotes.adapters.cache.postgres_cache import PostgresStageCache
from lecnotes.bootstrap import ApiContainer
from lecnotes.jobs.queue import JobQueue
from lecnotes.jobs.runner import JobRunner
from lecnotes.storage.db import create_engine, session_factory
from lecnotes.storage.repositories.notes import NotesRepository
from lecnotes.storage.repositories.stage_runs import PostgresStageRunSink
from lecnotes.storage.tables import Base
from tests.pipeline_fakes import fake_services, make_settings
from tests.postgres_server import ephemeral_postgres, find_pg_bin

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_asyncio_loop_factories(
    config: pytest.Config, item: pytest.Item
) -> Mapping[str, Callable[[], asyncio.AbstractEventLoop]]:
    # psycopg 3 async does not support Windows' default ProactorEventLoop.
    if sys.platform == "win32":
        return {"selector": asyncio.SelectorEventLoop}
    return {"default": asyncio.new_event_loop}


@pytest.fixture(scope="session")
def server_url() -> Iterator[str]:
    if url := os.environ.get("TEST_DATABASE_URL"):
        yield url
        return
    bin_dir = find_pg_bin()
    if bin_dir is None:
        pytest.skip("no PostgreSQL: set TEST_DATABASE_URL or install postgres (PG_BIN)")
    with ephemeral_postgres(bin_dir) as url:
        yield url


def alembic_config(url: str) -> Config:
    cfg = Config(os.path.join(REPO, "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", url)
    cfg.attributes["configure_logger"] = False  # keep pytest's logging setup
    return cfg


def create_database(server_url: str) -> str:
    """A fresh, empty database on the test server; returns its SQLAlchemy URL."""
    name = f"t_{uuid.uuid4().hex[:12]}"
    admin = make_url(server_url).set(drivername="postgresql")
    with psycopg.connect(admin.render_as_string(hide_password=False), autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{name}"')  # type: ignore[arg-type]
    return make_url(server_url).set(database=name).render_as_string(hide_password=False)


@pytest.fixture(scope="session")
def database_url(server_url: str) -> str:
    """One migrated database shared by the session; tables are truncated after each test."""
    url = create_database(server_url)
    command.upgrade(alembic_config(url), "head")
    return url


@pytest.fixture
async def engine(database_url: str) -> AsyncIterator[AsyncEngine]:
    eng = create_engine(database_url, pool_size=1)
    try:
        yield eng
    finally:
        tables = ", ".join(t.name for t in Base.metadata.sorted_tables)
        async with eng.begin() as conn:
            await conn.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
        await eng.dispose()


@pytest.fixture
async def session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    async with session_factory(engine)() as s:
        yield s


# --- the HTTP API over real Postgres, fakes only at the edges ---------------------------------

ADMIN = {"Authorization": "Bearer s3cret"}
TEST_JWT_KEY = "test-signing-key-at-least-32-bytes-long!"
OWNER_EMAIL = "owner@example.com"
GUEST_MINUTES = 20.0  # the fake video is 12.6 min: one fits, a second doesn't


def bearer(email: str, user_id: uuid.UUID | None = None, **claims: object) -> dict[str, str]:
    """Headers of a signed-in user (a Supabase-shaped token signed with the test key)."""
    payload = {
        "sub": str(user_id or uuid.uuid5(uuid.NAMESPACE_URL, email)),
        "email": email,
        "aud": "authenticated",
        "exp": int(time.time()) + 3600,
    } | claims
    return {"Authorization": f"Bearer {jwt.encode(payload, TEST_JWT_KEY, algorithm='HS256')}"}


@dataclass
class Api:
    client: httpx.AsyncClient
    """Signed in as the owner by default; pass `headers=bearer(...)` to act as someone else."""
    runner: JobRunner

    async def submit(
        self, url: str, preset: str = "general", headers: dict[str, str] | None = None
    ) -> dict:
        """Add a lesson the way the UI does (chapter → lesson); returns its job."""
        chapter = await self.client.post("/v1/chapters", json={"name": "Tests"}, headers=headers)
        assert chapter.status_code == 201, chapter.text
        lesson = await self.client.post(
            f"/v1/chapters/{chapter.json()['id']}/lessons",
            json={"url": url, "subject_preset": preset},
            headers=headers,
        )
        assert lesson.status_code == 201, lesson.text
        return (await self.client.get(f"/v1/jobs/{lesson.json()['job_id']}")).json()


@pytest.fixture
async def api(engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Api]:
    settings = make_settings(
        monkeypatch,
        admin_token=SecretStr("s3cret"),
        run_worker=False,
        owner_emails=[OWNER_EMAIL],
        guest_video_minutes=GUEST_MINUTES,
    )
    services, _ = fake_services(
        settings, PostgresStageCache(engine), sinks=[PostgresStageRunSink(engine)]
    )
    queue, notes = JobQueue(engine), NotesRepository(engine)
    runner = JobRunner(queue, services.pipeline, notes, owner="test")
    app = create_app(
        ApiContainer(settings, engine, queue, notes, services, runner),
        auth=StaticKeyAuthenticator(TEST_JWT_KEY),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=bearer(OWNER_EMAIL)
    ) as client:
        yield Api(client, runner)
