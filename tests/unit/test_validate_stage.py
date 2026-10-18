from uuid import uuid4

import pytest

from lecnotes.domain import (
    BoardState,
    BoardType,
    PerceptionResult,
    PerceptionRun,
    TranscriptSegment,
    ValidatedEvidence,
    VideoMeta,
)
from lecnotes.errors import PerceptionError
from lecnotes.pipeline.context import RunContext
from lecnotes.pipeline.stages.validate import ValidateStage, unclear_ratio

VID = "dQw4w9WgXcQ"
VIDEO = VideoMeta(video_id=VID, title="L", duration_s=600)


def seg(id: str, start: float, end: float, text: str = "words") -> TranscriptSegment:
    return TranscriptSegment(id=id, start_s=start, end_s=end, text=text)


def board(
    id: str,
    first: float,
    last: float,
    text: str = "x = 1",
    type: BoardType = BoardType.CODE,
    description: str = "",
) -> BoardState:
    return BoardState(
        id=id, t_first_seen=first, t_last_seen=last, type=type, text=text, description=description
    )


async def validate(
    segments: list[TranscriptSegment], boards: list[BoardState] | None = None
) -> ValidatedEvidence:
    run = PerceptionRun(
        result=PerceptionResult(video_id=VID, segments=segments, board_states=boards or []),
        usage=[],
    )
    return await ValidateStage().run(run, RunContext(job_id=uuid4(), video=VIDEO))


def actions(out: ValidatedEvidence) -> list[tuple[str, str]]:
    return [(i.action, i.item_id) for i in out.issues]


async def test_out_of_order_segments_are_sorted() -> None:
    out = await validate([seg("s2", 10, 12), seg("s1", 0, 5), seg("s3", 20, 25)])
    assert [s.id for s in out.evidence.segments] == ["s1", "s2", "s3"]
    assert out.issues == []


async def test_segments_past_the_end_are_dropped_or_clipped() -> None:
    out = await validate([seg("s1", 590, 640), seg("s2", 600, 610), seg("s3", 700, 710)])
    [s1] = out.evidence.segments
    assert s1.end_s == 600
    assert actions(out) == [("repaired", "s1"), ("dropped", "s2"), ("dropped", "s3")]


async def test_duplicate_segment_is_dropped() -> None:
    out = await validate(
        [
            seg("s1", 0, 5, "Enumerate gives the index."),
            seg("s2", 1, 6, "enumerate gives the index"),
        ]
    )
    [s1] = out.evidence.segments
    assert (s1.id, s1.end_s) == ("s1", 6)
    assert actions(out) == [("merged", "s2")]
    assert "duplicate of s1" in out.issues[0].reason


async def test_overlap_is_trimmed_not_concatenated() -> None:
    out = await validate([seg("s1", 0, 5, "first idea"), seg("s2", 4, 9, "second idea")])
    s1, s2 = out.evidence.segments
    assert (s1.start_s, s1.end_s, s2.start_s, s2.end_s) == (0, 5, 5, 9)
    assert s2.text == "second idea"
    assert actions(out) == [("repaired", "s2")]


async def test_overlap_leaving_a_sliver_is_merged() -> None:
    out = await validate([seg("s1", 0, 5, "first idea"), seg("s2", 3, 5.2, "and more")])
    [s1] = out.evidence.segments
    assert s1.text == "first idea and more"
    assert s1.end_s == 5.2
    assert actions(out) == [("merged", "s2")]


async def test_tiny_overlap_within_tolerance_is_fine() -> None:
    out = await validate([seg("s1", 0, 5), seg("s2", 4.9, 9, "other")])
    assert len(out.evidence.segments) == 2
    assert out.issues == []


async def test_empty_boards_dropped_and_boards_clipped() -> None:
    out = await validate(
        [seg("s1", 0, 5)],
        [
            board("b1", 0, 30, text="   "),
            board("b2", 30, 700),
            board("b3", 650, 660, text="late"),
        ],
    )
    [b2] = out.evidence.board_states
    assert b2.t_last_seen == 600
    assert actions(out) == [("dropped", "b1"), ("repaired", "b2"), ("dropped", "b3")]


async def test_board_with_only_a_description_is_kept() -> None:
    out = await validate([], [board("b1", 0, 10, text="", description="A tree diagram")])
    assert [b.id for b in out.evidence.board_states] == ["b1"]


async def test_near_identical_consecutive_boards_merge() -> None:
    out = await validate(
        [seg("s1", 0, 5)],
        [
            board("b1", 0, 40, "for i, x in enumerate(xs):\n    print(i, x)"),
            board("b2", 41, 90, "for i, x in enumerate(xs):\n  print(i, x)"),  # re-indent only
            board("b3", 91, 120, "zip(a, b)"),
        ],
    )
    b1, b3 = out.evidence.board_states
    assert (b1.id, b1.t_last_seen) == ("b1", 90)
    assert b3.id == "b3"
    assert actions(out) == [("merged", "b2")]


async def test_same_slide_shown_again_later_stays_separate() -> None:
    out = await validate(
        [seg("s1", 0, 5)],
        [
            board("b1", 0, 40, "Agenda", type=BoardType.SLIDE),
            board("b2", 41, 90, "def f(): ..."),
            board("b3", 91, 120, "Agenda", type=BoardType.SLIDE),
        ],
    )
    assert [b.id for b in out.evidence.board_states] == ["b1", "b2", "b3"]


async def test_same_text_different_type_is_not_merged() -> None:
    out = await validate(
        [],
        [
            board("b1", 0, 40, "x = 1", BoardType.SLIDE),
            board("b2", 41, 90, "x = 1", BoardType.CODE),
        ],
    )
    assert len(out.evidence.board_states) == 2


async def test_unclear_boards_are_flagged() -> None:
    out = await validate(
        [],
        [
            board("b1", 0, 40, "x = [unclear] + [unclear]"),
            board("b2", 41, 90, "y = 1 + [unclear] + 2 + 3 + 4 + 5"),
        ],
    )
    assert out.unclear_ratio == {"b1": pytest.approx(2 / 5), "b2": pytest.approx(1 / 13)}
    assert actions(out) == [("flagged", "b1")]


async def test_nothing_usable_is_rejected() -> None:
    with pytest.raises(PerceptionError, match="nothing usable"):
        await validate([seg("s1", 700, 710)], [board("b1", 0, 10, text="")])


def test_unclear_ratio_of_empty_text() -> None:
    assert unclear_ratio("") == 0


def test_thresholds_are_part_of_the_cache_config() -> None:
    ctx = RunContext(job_id=uuid4(), video=VIDEO)
    assert ValidateStage().config(ctx) != ValidateStage(duplicate_ratio=90).config(ctx)
