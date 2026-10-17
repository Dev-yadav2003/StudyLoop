from enum import StrEnum

from pydantic import Field

from lecnotes.domain.base import DomainModel, NonEmptyStr
from lecnotes.domain.knowledge import VideoSpan


class QuestionLevel(StrEnum):
    RECALL = "recall"
    APPLICATION = "application"


class Question(DomainModel):
    """A grounded study question: every answer cites the units and timestamps it comes from."""

    id: NonEmptyStr
    level: QuestionLevel
    question: NonEmptyStr
    answer: NonEmptyStr
    unit_ids: list[NonEmptyStr] = Field(min_length=1)
    spans: list[VideoSpan] = Field(min_length=1)
