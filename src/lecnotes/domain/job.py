from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Self
from uuid import UUID

from pydantic import Field, model_validator

from lecnotes.domain.base import DomainModel, NonEmptyStr


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class SubjectPreset(StrEnum):
    STEM = "stem"
    THEORY = "theory"
    GENERAL = "general"


class JobOptions(DomainModel):
    """Everything that changes the output. Part of job idempotency (same url + options)."""

    subject_preset: SubjectPreset = SubjectPreset.GENERAL
    generate_questions: bool = False


class Job(DomainModel):
    id: UUID
    url: NonEmptyStr
    video_id: str | None = None
    options: JobOptions = JobOptions()
    status: JobStatus = JobStatus.QUEUED
    error: str | None = Field(default=None, description="User-readable failure reason")
    error_type: str | None = Field(
        default=None, description="Last error class; also set while a retry is waiting"
    )
    attempts: int = Field(default=0, ge=0)
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @model_validator(mode="after")
    def _error_matches_status(self) -> Self:
        if self.status is JobStatus.FAILED and not self.error:
            raise ValueError("a failed job must carry an error message")
        if self.status is not JobStatus.FAILED and self.error:
            raise ValueError(f"a {self.status} job must not carry an error message")
        return self


class StageRunStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"


class StageRun(DomainModel):
    """One execution (or cache hit) of one stage. Kept forever for version-to-version history."""

    job_id: UUID
    video_id: str | None = None
    stage: NonEmptyStr
    stage_version: int = Field(ge=1)
    prompt_version: int | None = Field(default=None, ge=1)
    model: str | None = None
    status: StageRunStatus
    cache_hit: bool = False
    latency_ms: float = Field(ge=0)
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    cost_usd: Decimal = Field(default=Decimal(0), ge=0)
    error_type: str | None = None
    trace_id: str | None = None
    cache_key: str | None = Field(
        default=None, description="Stage cache key: equal keys = identical inputs/config/versions"
    )
    started_at: datetime

    @model_validator(mode="after")
    def _error_type_matches_status(self) -> Self:
        if (self.status is StageRunStatus.FAILED) != (self.error_type is not None):
            raise ValueError("error_type is required iff status is 'failed'")
        return self
