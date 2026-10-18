from typing import Literal

from pydantic import Field, field_validator

from lecnotes.domain.base import DomainModel

Rating = Literal[-1, 1]


class Feedback(DomainModel):
    """A reader's verdict on finished notes (or one unit of them): thumbs up/down + why."""

    rating: Rating = Field(description="1 = useful, -1 = not useful")
    reason: str | None = Field(default=None, max_length=1000)
    unit_id: str | None = Field(default=None, description="The unit it is about; None = all")

    @field_validator("reason")
    @classmethod
    def _blank_is_none(cls, v: str | None) -> str | None:
        return (v or "").strip() or None
