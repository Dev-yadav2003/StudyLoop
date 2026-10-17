from collections import Counter
from typing import Self

from pydantic import Field, model_validator

from lecnotes.domain.base import DomainModel, VideoId
from lecnotes.domain.transcript import TranscriptSegment
from lecnotes.domain.usage import LLMUsage
from lecnotes.domain.visual import BoardState


class PerceptionResult(DomainModel):
    """Raw evidence extracted from one video: what was said and what was shown.

    Ordering, overlaps and duplicates are repaired by the validate stage, not here. The only
    invariant enforced is that IDs are unique, since everything downstream cites them.
    """

    video_id: VideoId
    segments: list[TranscriptSegment]
    board_states: list[BoardState]

    @model_validator(mode="after")
    def _ids_unique(self) -> Self:
        ids = [s.id for s in self.segments] + [b.id for b in self.board_states]
        dupes = sorted(i for i, n in Counter(ids).items() if n > 1)
        if dupes:
            raise ValueError(f"duplicate evidence ids: {dupes}")
        return self


class PerceptionRun(DomainModel):
    """A perception result plus how it was produced (for metrics, caching and debugging)."""

    result: PerceptionResult
    usage: list[LLMUsage]
    dropped: list[str] = Field(default_factory=list, description="Model items rejected, why")
    repaired: list[str] = Field(default_factory=list, description="Model items fixed, how")
