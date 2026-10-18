"""HTTP request/response shapes. Domain models are reused where they're already the public
shape (VideoMeta, KnowledgeUnit); everything else is defined here, so the API contract —
and the TypeScript client generated from /openapi.json — doesn't change by accident."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field, StringConstraints

from lecnotes.domain import (
    JobStatus,
    KnowledgeUnit,
    QaLevel,
    QuestionLevel,
    SubjectPreset,
    VideoMeta,
)


class ErrorBody(BaseModel):
    code: str = Field(description="Machine-readable error type, e.g. InvalidLink")
    message: str = Field(description="Safe to show to the user")
    request_id: str | None = None


class ErrorOut(BaseModel):
    error: ErrorBody


StageState = Literal["pending", "done", "cached", "failed"]


class StageProgress(BaseModel):
    stage: str
    state: StageState
    latency_ms: float | None = None
    error_type: str | None = None


class JobOut(BaseModel):
    id: UUID
    url: str
    video_id: str | None
    subject_preset: SubjectPreset
    status: JobStatus
    error: str | None = Field(description="User-facing reason when status is failed")
    error_type: str | None
    attempts: int
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    stages: list[StageProgress]
    notes_id: UUID | None = None


class NotesOut(BaseModel):
    id: UUID
    job_id: UUID
    title: str
    video: VideoMeta
    markdown: str
    units: list[KnowledgeUnit]
    edited_units: list[str] = Field(description="Ids of units a user has edited")
    revision: int = Field(description="Bumped on every edit")


class StageRunOut(BaseModel):
    stage: str
    stage_version: int
    prompt_version: int | None
    model: str | None
    status: str
    cache_hit: bool
    latency_ms: float
    tokens_in: int
    tokens_out: int
    cost_usd: str
    error_type: str | None
    trace_id: str | None
    started_at: datetime


class RerunOut(BaseModel):
    job: JobOut
    rerun_from: str | None


class QaCreate(BaseModel):
    level: QaLevel = QaLevel.MIXED
    count: int = Field(default=5, ge=1, le=10)


class UnitRef(BaseModel):
    id: str
    concept: str


class TimestampLink(BaseModel):
    start_s: float
    label: str = Field(examples=["03:29"])
    url: str


class QuestionOut(BaseModel):
    id: str
    level: QuestionLevel
    question: str
    answer: str
    units: list[UnitRef]
    timestamps: list[TimestampLink]


class QuestionSetOut(BaseModel):
    notes_id: UUID
    questions: list[QuestionOut]


class FeedbackOut(BaseModel):
    id: int
    created_at: datetime
    scored: bool = Field(description="Also recorded as a Langfuse score on the job's trace")


class FeedbackItemOut(BaseModel):
    id: int
    notes_id: UUID
    video_id: str
    title: str
    rating: int
    reason: str | None
    unit_id: str | None
    trace_id: str | None
    created_at: datetime


class FeedbackListOut(BaseModel):
    up: int
    down: int
    items: list[FeedbackItemOut]


Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]
Title = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class ChapterCreate(BaseModel):
    name: Name = Field(examples=["Python"])


class LessonOut(BaseModel):
    id: UUID
    chapter_id: UUID
    title: str = Field(description="The custom title, else the video title, else the link")
    custom_title: str | None
    url: str
    video_id: str | None
    job_id: UUID
    status: JobStatus
    notes_id: UUID | None
    created_at: datetime


class ChapterOut(BaseModel):
    id: UUID
    name: str
    created_at: datetime
    lessons: list[LessonOut]


class LessonCreate(BaseModel):
    url: str = Field(min_length=1, examples=["https://www.youtube.com/watch?v=Ek49cGiAOwo"])
    subject_preset: SubjectPreset = SubjectPreset.GENERAL
    title: Title | None = None


class LessonUpdate(BaseModel):
    title: Title | None = Field(default=None, description="null resets to the video title")
    chapter_id: UUID | None = Field(default=None, description="move to another chapter")


class MeOut(BaseModel):
    email: str
    is_owner: bool
    video_minutes_used: float
    video_minutes_limit: float | None = Field(description="None = unlimited (owner)")
    tokens_used: int
    tokens_limit: int | None = Field(description="None = unlimited (owner)")
