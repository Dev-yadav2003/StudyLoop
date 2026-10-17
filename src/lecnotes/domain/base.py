"""Shared building blocks for domain models. Imports nothing from the project."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Seconds = Annotated[float, Field(ge=0, description="Offset from the start of the video, seconds")]
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Score = Annotated[float, Field(ge=0, le=1)]
VideoId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{11}$")]


class DomainModel(BaseModel):
    """Immutable, strict base: stage outputs never change after creation, unknown fields fail."""

    model_config = ConfigDict(frozen=True, extra="forbid")


def check_time_order(start: float, end: float, *, allow_equal: bool, what: str) -> None:
    if end < start or (end == start and not allow_equal):
        op = ">=" if allow_equal else ">"
        raise ValueError(f"{what}: end ({end}) must be {op} start ({start})")
