from uuid import uuid4

import pytest

from lecnotes.domain import (
    BoardState,
    BoardType,
    PerceptionResult,
    TranscriptSegment,
    ValidatedEvidence,
    VideoMeta,
)
from lecnotes.pipeline.context import RunContext
from lecnotes.pipeline.stages.align.align import AlignStage
from lecnotes.pipeline.stages.align.temporal import temporal_links, windows

DURATION = 200.0


def seg(id: str, start: float, end: float) -> TranscriptSegment:
    return TranscriptSegment(id=id, start_s=start, end_s=end, text="words")


def board(id: str, first: float, last: float) -> BoardState:
    return BoardState(id=id, t_first_seen=first, t_last_seen=last, type=BoardType.SLIDE, text=id)


# Timeline: b1 [0,60)  b2 [60,120)  b3 [120,200]
BOARDS = [board("b1", 0, 59), board("b2", 60, 119), board("b3", 120, 200)]


def links_of(
    segments: list[TranscriptSegment], boards: list[BoardState] = BOARDS
) -> dict[tuple[str, str], float]:
    links = temporal_links(segments, boards, DURATION)
    return {(lk.segment_id, lk.board_state_id): lk.score for lk in links}


def test_windows_run_to_next_start_and_last_to_video_end() -> None:
    assert [(w.board_id, w.start, w.end) for w in windows(BOARDS, DURATION)] == [
        ("b1", 0, 60),
        ("b2", 60, 120),
        ("b3", 120, 200),
    ]


def test_core_window_by_midpoint() -> None:
    # s1 straddles the b1/b2 change but its midpoint (58) is in b1's window.
    links = links_of([seg("s0", 5, 15), seg("s1", 50, 66)])
    assert links[("s0", "b1")] == 1.0
    assert links[("s1", "b1")] == 1.0
    assert ("s0", "b2") not in links


def test_extended_window_before_and_after_decays() -> None:
    links = links_of([seg("s1", 50, 60), seg("s2", 120, 130)])
    # s1 mid 55: 5 s before b2 appears → 0.5 * (1 - 5/15)
    assert links[("s1", "b2")] == pytest.approx(0.5 * (1 - 5 / 15), abs=1e-4)
    # s2 mid 125: 5 s after b2's window/visibility ended (120) → 0.5 * (1 - 5/30)
    assert links[("s2", "b2")] == pytest.approx(0.5 * (1 - 5 / 30), abs=1e-4)
    assert links[("s2", "b3")] == 1.0


def test_far_speech_is_not_linked() -> None:
    links = links_of([seg("s1", 170, 180)])
    assert set(links) == {("s1", "b3")}


def test_speech_before_the_first_board_gets_an_extended_link() -> None:
    boards = [board("b1", 20, 100)]
    links = links_of([seg("s1", 5, 15)], boards)  # mid 10: 10 s before b1 appears
    assert links == {("s1", "b1"): pytest.approx(0.5 * (1 - 10 / 15), abs=1e-4)}


def test_board_still_visible_after_next_appears_gets_visible_link() -> None:
    # Slide b1 stays on screen (picture-in-picture) while code b2 is shown.
    boards = [board("b1", 0, 100), board("b2", 60, 100)]
    links = links_of([seg("s1", 70, 80)], boards)
    assert links[("s1", "b2")] == 1.0
    assert links[("s1", "b1")] == 0.6


def test_no_boards_no_links() -> None:
    assert temporal_links([seg("s1", 0, 5)], [], DURATION) == []


async def test_align_stage() -> None:
    evidence = PerceptionResult(
        video_id="dQw4w9WgXcQ", segments=[seg("s1", 5, 15)], board_states=BOARDS
    )
    ctx = RunContext(
        job_id=uuid4(), video=VideoMeta(video_id="dQw4w9WgXcQ", title="L", duration_s=DURATION)
    )
    out = await AlignStage().run(ValidatedEvidence(evidence=evidence), ctx)
    assert out.evidence == evidence
    assert [(lk.segment_id, lk.board_state_id, lk.score) for lk in out.links] == [("s1", "b1", 1.0)]
