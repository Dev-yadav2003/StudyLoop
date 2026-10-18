"""What the perception model returns (LLM-facing), and its conversion to domain objects.

Kept separate from `domain.PerceptionResult` on purpose:
  - the model should not invent IDs or satisfy our strict invariants — our code assigns IDs
    and repairs/drops bad items (recording why) instead of failing the whole video;
  - times are "MM:SS" strings: the format Gemini uses for video timestamps.

Field order matters: models generate fields in order, so `board_states` comes first. If the
output is ever truncated, the scarce signal (board/code text) survives; speech has a
fallback (captions), board text does not.

This schema is part of the perceive prompt contract: change it → new prompt version.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from lecnotes.domain import BoardState, BoardType, PerceptionResult, TranscriptSegment
from lecnotes.timefmt import parse_timestamp

MIN_SEGMENT_S = 1.0


class PerceivedBoardState(BaseModel):
    first_seen: str = Field(description="When this exact content first appears, MM:SS")
    last_seen: str = Field(description="Last moment this exact content is visible, MM:SS")
    type: Literal["slide", "whiteboard", "code", "screen", "other"]
    text: str = Field(
        description=(
            "ALL text visible in this state, copied verbatim: same characters, symbols, "
            "line breaks and code indentation. [unclear] for unreadable parts. "
            "Empty string if there is no text."
        )
    )
    description: str = Field(
        description="One line describing diagrams/figures/plots, else empty string"
    )


class PerceivedSegment(BaseModel):
    start: str = Field(description="Start time, MM:SS")
    end: str = Field(description="End time, MM:SS")
    text: str = Field(description="Exactly what the teacher says, 1-3 sentences")


class PerceiveOutput(BaseModel):
    board_states: list[PerceivedBoardState]
    segments: list[PerceivedSegment]


@dataclass
class Conversion:
    result: PerceptionResult
    dropped: list[str] = field(default_factory=list)
    repaired: list[str] = field(default_factory=list)


def to_perception_result(raw: PerceiveOutput, video_id: str) -> Conversion:
    """Assign IDs, parse times, and enforce domain invariants item by item."""
    dropped: list[str] = []
    repaired: list[str] = []

    boards: list[BoardState] = []
    for i, b in enumerate(raw.board_states):
        where = f"board_states[{i}]"
        try:
            first, last = parse_timestamp(b.first_seen), parse_timestamp(b.last_seen)
        except ValueError as exc:
            dropped.append(f"{where}: {exc}")
            continue
        if last < first:
            dropped.append(f"{where}: last_seen {b.last_seen} before first_seen {b.first_seen}")
            continue
        boards.append(
            BoardState(
                id=f"b{len(boards) + 1:04d}",
                t_first_seen=first,
                t_last_seen=last,
                type=BoardType(b.type),
                text=b.text,
                description=b.description.strip(),
            )
        )

    segments: list[TranscriptSegment] = []
    for i, s in enumerate(raw.segments):
        where = f"segments[{i}]"
        if not s.text.strip():
            dropped.append(f"{where}: empty text")
            continue
        try:
            start, end = parse_timestamp(s.start), parse_timestamp(s.end)
        except ValueError as exc:
            dropped.append(f"{where}: {exc}")
            continue
        if end < start:
            dropped.append(f"{where}: end {s.end} before start {s.start}")
            continue
        if end == start:
            # 1 fps sampling often yields equal times for a short utterance; keep the words.
            end = start + MIN_SEGMENT_S
            repaired.append(f"{where}: zero-length segment at {s.start} extended to 1 s")
        segments.append(
            TranscriptSegment(id=f"s{len(segments) + 1:04d}", start_s=start, end_s=end, text=s.text)
        )

    result = PerceptionResult(video_id=video_id, segments=segments, board_states=boards)
    return Conversion(result=result, dropped=dropped, repaired=repaired)
