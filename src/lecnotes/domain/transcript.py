from enum import StrEnum
from typing import Self

from pydantic import model_validator

from lecnotes.domain.base import DomainModel, NonEmptyStr, Seconds, check_time_order


class TranscriptSource(StrEnum):
    GEMINI = "gemini"
    YOUTUBE_CAPTIONS = "youtube_captions"
    MANUAL = "manual"


class TranscriptSegment(DomainModel):
    """A span of speech. `id` is unique within one video and is cited as evidence."""

    id: NonEmptyStr
    start_s: Seconds
    end_s: Seconds
    text: NonEmptyStr
    source: TranscriptSource = TranscriptSource.GEMINI

    @model_validator(mode="after")
    def _times_ordered(self) -> Self:
        check_time_order(self.start_s, self.end_s, allow_equal=False, what=f"segment {self.id}")
        return self
