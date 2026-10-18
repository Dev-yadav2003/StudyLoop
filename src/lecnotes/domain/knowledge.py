from typing import Self

from pydantic import Field, model_validator

from lecnotes.domain.base import DomainModel, NonEmptyStr, Seconds, VideoId, check_time_order


def primary_first(spans: list["VideoSpan"]) -> list["VideoSpan"]:
    """Longest span first (where the concept is actually taught — teaching takes time, passing
    mentions are brief), then the other moments in time order."""
    if not spans:
        return []
    primary = max(spans, key=lambda s: (s.end_s - s.start_s, -s.start_s))
    rest = sorted((s for s in spans if s is not primary), key=lambda s: s.start_s)
    return [primary, *rest]


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
    output: str | None = Field(
        default=None, description="What the program printed, as shown in the lecture"
    )


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
    spans: list[VideoSpan] = Field(
        min_length=1, description="Primary teaching moment first, other moments in time order"
    )
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


class UnitEdit(DomainModel):
    """A user's correction to one unit. Only the content is editable; where it was taught
    (spans) and what it cites (evidence) stay as extracted. Omitted fields are unchanged;
    `board_text`/`section` can be cleared with an empty string."""

    concept: NonEmptyStr | None = None
    explanation: NonEmptyStr | None = None
    examples: list[NonEmptyStr] | None = None
    board_text: str | None = None
    code: list[CodeBlock] | None = None
    section: str | None = None

    @model_validator(mode="after")
    def _not_empty(self) -> Self:
        if not self.model_fields_set:
            raise ValueError("an edit must change at least one field")
        required = {"concept", "explanation", "examples", "code"} & self.model_fields_set
        if nulled := sorted(n for n in required if getattr(self, n) is None):
            raise ValueError(f"cannot be null (send [] to empty a list): {', '.join(nulled)}")
        return self

    def apply(self, unit: KnowledgeUnit) -> KnowledgeUnit:
        changes = {name: getattr(self, name) for name in self.model_fields_set}
        for name in ("board_text", "section"):
            if name in changes:
                changes[name] = (changes[name] or "").strip() or None
        return KnowledgeUnit.model_validate(unit.model_dump() | changes)
