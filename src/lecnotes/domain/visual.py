from enum import StrEnum
from typing import Self

from pydantic import model_validator

from lecnotes.domain.base import DomainModel, NonEmptyStr, Seconds, check_time_order

UNCLEAR_MARKER = "[unclear]"


class BoardType(StrEnum):
    SLIDE = "slide"
    WHITEBOARD = "whiteboard"
    CODE = "code"
    SCREEN = "screen"
    OTHER = "other"


class BoardState(DomainModel):
    """One stable state of what is visible (a slide, a board snapshot, a screen of code).

    A new state starts whenever visible content changes. `text` is verbatim; it may be empty
    here (e.g. a diagram with no text) — the validate stage decides what to drop.
    """

    id: NonEmptyStr
    t_first_seen: Seconds
    t_last_seen: Seconds
    type: BoardType
    text: str = ""
    description: str = ""

    @model_validator(mode="after")
    def _times_ordered(self) -> Self:
        # A state may be seen at a single sampled frame, so equal times are allowed.
        check_time_order(
            self.t_first_seen, self.t_last_seen, allow_equal=True, what=f"board state {self.id}"
        )
        return self

    @property
    def is_empty(self) -> bool:
        return not self.text.strip() and not self.description.strip()
