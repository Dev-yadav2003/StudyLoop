from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from lecnotes.domain.base import (
    DomainModel,
    NonEmptyStr,
    Score,
    Seconds,
    VideoId,
    check_time_order,
)
from lecnotes.domain.transcript import TranscriptSegment
from lecnotes.domain.visual import BoardState


class AlignmentMethod(StrEnum):
    TEMPORAL = "temporal"
    LEXICAL = "lexical"


class AlignmentLink(DomainModel):
    """A weak, scored link between speech and a board state. Many-to-many by design."""

    segment_id: NonEmptyStr
    board_state_id: NonEmptyStr
    method: AlignmentMethod
    score: Score


class AlignedChunk(DomainModel):
    """A self-contained 2-5 minute slice of evidence: the unit of work for `extract`."""

    id: NonEmptyStr
    video_id: VideoId
    start_s: Seconds
    end_s: Seconds
    segments: list[TranscriptSegment]
    board_states: list[BoardState]
    links: list[AlignmentLink] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        check_time_order(self.start_s, self.end_s, allow_equal=False, what=f"chunk {self.id}")
        seg_ids = {s.id for s in self.segments}
        board_ids = {b.id for b in self.board_states}
        dangling = [
            link
            for link in self.links
            if link.segment_id not in seg_ids or link.board_state_id not in board_ids
        ]
        if dangling:
            raise ValueError(f"chunk {self.id}: {len(dangling)} link(s) reference missing ids")
        return self
