"""Postgres schema. Change it only together with an Alembic migration (tested for drift).

Conventions:
  - enum-like columns are TEXT + CHECK built from the domain StrEnums (native PG enums are
    painful to evolve in migrations);
  - all timestamps are timestamptz with server defaults;
  - structured payloads (stage outputs, units, options) are JSONB of Pydantic dumps.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    MetaData,
    Numeric,
    SmallInteger,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from lecnotes.domain import JobStatus, StageRunStatus

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {  # noqa: RUF012 - SQLAlchemy reads this class attribute
        dict[str, Any]: JSONB,
        datetime: DateTime(timezone=True),
        str: Text,
    }


def _one_of(column: str, enum: type[StrEnum], name: str) -> CheckConstraint:
    values = ", ".join(f"'{member.value}'" for member in enum)
    return CheckConstraint(f"{column} IN ({values})", name=name)


def _created() -> Mapped[datetime]:
    return mapped_column(server_default=func.now())


ACTIVE_JOB_STATUSES = (JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.SUCCEEDED)


class JobRow(Base):
    """A job is also a queue entry: leased with FOR UPDATE SKIP LOCKED (jobs/queue.py)."""

    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    url: Mapped[str]
    video_id: Mapped[str | None]
    request_key: Mapped[str]
    options: Mapped[dict[str, Any]]
    status: Mapped[str]
    error: Mapped[str | None]
    error_type: Mapped[str | None]
    attempts: Mapped[int] = mapped_column(server_default="0")
    rerun_from: Mapped[str | None]
    lease_owner: Mapped[str | None]
    leased_until: Mapped[datetime | None]
    run_after: Mapped[datetime] = mapped_column(server_default=func.now())
    created_at: Mapped[datetime] = _created()
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())
    started_at: Mapped[datetime | None]
    finished_at: Mapped[datetime | None]

    __table_args__ = (
        _one_of("status", JobStatus, "status"),
        Index(None, "status", "run_after"),
        # Idempotency: one active job per (video, options). Failed jobs may be resubmitted.
        Index(
            "uq_jobs_request_key_active",
            "request_key",
            unique=True,
            postgresql_where=text(
                "status IN (" + ", ".join(f"'{s.value}'" for s in ACTIVE_JOB_STATUSES) + ")"
            ),
        ),
    )


class StageCacheRow(Base):
    """Whole stage outputs keyed by a content hash (see cache/keys.py)."""

    __tablename__ = "stage_cache"

    key: Mapped[str] = mapped_column(primary_key=True)
    stage: Mapped[str]
    stage_version: Mapped[int]
    prompt_version: Mapped[int | None]
    model: Mapped[str | None]
    video_id: Mapped[str | None]
    payload: Mapped[dict[str, Any]]
    hit_count: Mapped[int] = mapped_column(server_default="0")
    created_at: Mapped[datetime] = _created()
    last_hit_at: Mapped[datetime | None]

    __table_args__ = (
        CheckConstraint("length(key) = 64", name="key_is_sha256"),
        Index(None, "video_id", "stage"),
    )


class StageRunRow(Base):
    """One execution (or cache hit) of one stage. Permanent history: no FK to jobs, so eval
    runs can record rows without a job, and deleting a job never deletes its metrics."""

    __tablename__ = "stage_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID]
    video_id: Mapped[str | None]
    stage: Mapped[str]
    stage_version: Mapped[int]
    prompt_version: Mapped[int | None]
    model: Mapped[str | None]
    status: Mapped[str]
    cache_hit: Mapped[bool]
    latency_ms: Mapped[float]
    tokens_in: Mapped[int]
    tokens_out: Mapped[int]
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6))
    error_type: Mapped[str | None]
    trace_id: Mapped[str | None]
    cache_key: Mapped[str | None]
    started_at: Mapped[datetime]

    __table_args__ = (
        _one_of("status", StageRunStatus, "status"),
        Index(None, "stage", "stage_version", "started_at"),
        Index(None, "job_id"),
    )


class NotesRow(Base):
    __tablename__ = "notes"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), unique=True
    )
    title: Mapped[str]
    markdown: Mapped[str]
    video: Mapped[dict[str, Any]]
    evidence: Mapped[dict[str, Any] | None]
    verify: Mapped[dict[str, Any] | None]
    revision: Mapped[int] = mapped_column(server_default="0")
    """Bumped on every edit; an edit based on an older revision is rejected (409)."""
    created_at: Mapped[datetime] = _created()
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class UnitRow(Base):
    """A KnowledgeUnit of a notes document, in order."""

    __tablename__ = "units"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    notes_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("notes.id", ondelete="CASCADE"))
    unit_key: Mapped[str]
    position: Mapped[int]
    data: Mapped[dict[str, Any]]
    edited_at: Mapped[datetime | None]
    """Set when a user edited this unit (step 9.2); None = as extracted."""
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())

    __table_args__ = (UniqueConstraint("notes_id", "unit_key"),)


class FeedbackRow(Base):
    """Thumbs up/down (+ optional reason) on notes or one of their units (step 9.3). Also sent
    to Langfuse as a score on the job's trace."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    notes_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("notes.id", ondelete="CASCADE"), index=True
    )
    unit_key: Mapped[str | None]
    rating: Mapped[int] = mapped_column(SmallInteger)
    reason: Mapped[str | None]
    trace_id: Mapped[str | None]
    created_at: Mapped[datetime] = _created()

    __table_args__ = (CheckConstraint("rating IN (-1, 1)", name="rating"),)
