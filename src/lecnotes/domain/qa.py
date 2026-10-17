from enum import StrEnum

from pydantic import Field

from lecnotes.domain.base import DomainModel, NonEmptyStr
from lecnotes.domain.knowledge import KnowledgeUnit, VideoSpan


class QuestionLevel(StrEnum):
    RECALL = "recall"  # the lecture states it
    APPLICATION = "application"  # apply a taught concept to a new small case


class QaLevel(StrEnum):
    """What the user asks for."""

    RECALL = "recall"
    APPLICATION = "application"
    MIXED = "mixed"


class Question(DomainModel):
    """A grounded practice question: it cites the note units its answer comes from, and its
    timestamps are those units' teaching moments (computed, never written by the model)."""

    id: NonEmptyStr
    level: QuestionLevel
    question: NonEmptyStr
    answer: NonEmptyStr
    unit_ids: list[NonEmptyStr] = Field(min_length=1)
    spans: list[VideoSpan] = Field(min_length=1)


class QaRequest(DomainModel):
    units: list[KnowledgeUnit] = Field(min_length=1)
    level: QaLevel = QaLevel.MIXED
    count: int = Field(default=5, ge=1, le=10)


class QuestionSet(DomainModel):
    questions: list[Question]
    dropped: list[str] = Field(default_factory=list, description="Model questions rejected, why")
