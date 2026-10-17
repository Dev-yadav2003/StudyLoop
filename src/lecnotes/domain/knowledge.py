from typing import Self

from pydantic import Field, model_validator

from lecnotes.domain.base import DomainModel, NonEmptyStr, Seconds, VideoId, check_time_order


class VideoSpan(DomainModel):
    """Where something happens in a video. A merged unit keeps one span per occurrence."""

    video_id: VideoId
    start_s: Seconds
    end_s: Seconds

    @model_validator(mode="after")
    def _times_ordered(self) -> Self:
        check_time_order(self.start_s, self.end_s, allow_equal=False, what="span")
        return self


class Evidence(DomainModel):
    """IDs of the transcript segments and board states (within one video) backing a claim."""

    video_id: VideoId
    segment_ids: list[NonEmptyStr] = Field(default_factory=list)
    board_state_ids: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _not_empty(self) -> Self:
        if not self.segment_ids and not self.board_state_ids:
            raise ValueError("evidence must cite at least one segment or board state")
        return self


class CodeBlock(DomainModel):
    source: NonEmptyStr
    language: str | None = None


class KnowledgeUnit(DomainModel):
    """One concept as taught: explanation, examples, verbatim board text/code, with provenance.

    No evidence = no unit: every unit must cite where it came from.
    """

    id: NonEmptyStr
    concept: NonEmptyStr
    explanation: NonEmptyStr
    examples: list[NonEmptyStr] = Field(default_factory=list)
    board_text: str | None = None
    code: list[CodeBlock] = Field(default_factory=list)
    section: str | None = None
    spans: list[VideoSpan] = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)

    @property
    def start_s(self) -> float:
        return self.spans[0].start_s

    @property
    def end_s(self) -> float:
        return self.spans[0].end_s

    @property
    def video_ids(self) -> list[str]:
        return list(dict.fromkeys(span.video_id for span in self.spans))
