import random
from collections import Counter
from uuid import uuid4

import pytest

from lecnotes.domain import (
    Alignment,
    AlignmentLink,
    AlignmentMethod,
    BoardState,
    BoardType,
    Chunks,
    PerceptionResult,
    TranscriptSegment,
    VideoMeta,
)
from lecnotes.pipeline.context import RunContext
from lecnotes.pipeline.stages.align.chunk import ChunkStage
from lecnotes.pipeline.stages.align.temporal import temporal_links

VID = "dQw4w9WgXcQ"
MIN, MAX = 120.0, 300.0


def board(id: str, first: float, last: float) -> BoardState:
    return BoardState(id=id, t_first_seen=first, t_last_seen=last, type=BoardType.SLIDE, text=id)


def segments_every(step: float, until: float, start: float = 0.0) -> list[TranscriptSegment]:
    out, t, n = [], start, 1
    while t + step <= until:
        out.append(TranscriptSegment(id=f"s{n:04d}", start_s=t, end_s=t + step, text=f"w{n}"))
        t, n = t + step, n + 1
    return out


async def chunk(
    boards: list[BoardState],
    segments: list[TranscriptSegment],
    duration: float,
    extra_links: list[AlignmentLink] | None = None,
) -> Chunks:
    links = temporal_links(segments, boards, duration) + (extra_links or [])
    alignment = Alignment(
        evidence=PerceptionResult(video_id=VID, segments=segments, board_states=boards),
        links=links,
    )
    ctx = RunContext(job_id=uuid4(), video=VideoMeta(video_id=VID, title="L", duration_s=duration))
    return await ChunkStage(min_s=MIN, max_s=MAX).run(alignment, ctx)


def assert_invariants(out: Chunks, segments: list[TranscriptSegment]) -> None:
    seen = Counter(s.id for c in out.chunks for s in c.segments)
    assert set(seen) == {s.id for s in segments}, "every segment is in a chunk"
    assert all(n == 1 for n in seen.values()), "no segment is in two chunks"
    for c in out.chunks:
        for b in c.board_states:  # boards are carried whole, never cut
            assert b.text == b.id


async def test_regular_lecture_chunks_stay_within_bounds() -> None:
    boards = [board(f"b{i:02d}", i * 60, i * 60 + 59) for i in range(10)]  # 10 x 1-min slides
    segments = segments_every(10, 600)

    out = await chunk(boards, segments, 600)

    assert_invariants(out, segments)
    spans = [c.end_s - c.start_s for c in out.chunks]
    assert all(MIN <= s <= MAX for s in spans), spans
    assert [c.id for c in out.chunks] == [f"c{i:03d}" for i in range(1, len(out.chunks) + 1)]


async def test_cuts_happen_only_at_board_boundaries() -> None:
    boards = [board(f"b{i:02d}", i * 70, i * 70 + 69) for i in range(9)]  # 70 s slides
    out = await chunk(boards, segments_every(10, 630), 630)
    starts = {b.t_first_seen for b in boards}
    assert all(c.start_s in starts for c in out.chunks)


async def test_long_single_board_is_split_by_segments_with_board_copied() -> None:
    boards = [board("b1", 0, 700)]
    segments = segments_every(20, 700)

    out = await chunk(boards, segments, 700)

    assert_invariants(out, segments)
    assert len(out.chunks) >= 3
    assert all(c.end_s - c.start_s <= MAX for c in out.chunks)
    assert all([b.id for b in c.board_states] == ["b1"] for c in out.chunks)


async def test_no_boards_chunks_by_time() -> None:
    segments = segments_every(15, 900)
    out = await chunk([], segments, 900)
    assert_invariants(out, segments)
    assert all(c.end_s - c.start_s <= MAX for c in out.chunks)
    assert all(c.board_states == [] for c in out.chunks)


async def test_speech_before_first_board_is_kept() -> None:
    boards = [board("b1", 30, 200)]
    segments = segments_every(10, 200)
    out = await chunk(boards, segments, 200)
    assert_invariants(out, segments)
    assert out.chunks[0].start_s == 0


async def test_short_tail_is_merged_into_previous_chunk() -> None:
    # 4 x 70 s slides = 280 s fits one chunk; a final 40 s slide would be a tiny chunk alone.
    boards = [board(f"b{i}", i * 70, i * 70 + 69) for i in range(4)] + [board("b4", 280, 320)]
    segments = segments_every(10, 320)
    out = await chunk(boards, segments, 320)
    assert_invariants(out, segments)
    # 320 s > MAX, so it can't be one chunk; but no chunk may be a lone 40 s tail.
    assert all(c.end_s - c.start_s >= MIN for c in out.chunks)


async def test_lexically_linked_earlier_board_is_attached() -> None:
    boards = [board(f"b{i}", i * 100, i * 100 + 99) for i in range(6)]
    segments = segments_every(20, 600)
    late = segments[-1]  # spoken at the end, refers back to b0
    lexical = AlignmentLink(
        segment_id=late.id, board_state_id="b0", method=AlignmentMethod.LEXICAL, score=1.0
    )

    out = await chunk(boards, segments, 600, [lexical])

    last = next(c for c in out.chunks if late in c.segments)
    assert "b0" in [b.id for b in last.board_states]
    assert any(lk.board_state_id == "b0" for lk in last.links)


def test_bounds_validated() -> None:
    with pytest.raises(ValueError, match="min_s"):
        ChunkStage(min_s=400, max_s=300)


async def test_invariants_hold_on_random_lectures() -> None:
    rng = random.Random(1234)  # seeded: failures are reproducible
    for _ in range(200):
        duration = rng.uniform(300, 3600)
        t, boards = rng.uniform(0, 40), []
        while t < duration:
            length = rng.choice([rng.uniform(5, 90), rng.uniform(90, 900)])
            boards.append(board(f"b{len(boards):03d}", t, min(t + length, duration)))
            t += length
        segments = segments_every(rng.uniform(3, 20), duration)

        out = await chunk(boards, segments, duration)

        assert_invariants(out, segments)
        for c in out.chunks:
            # Only a piece that couldn't be split (fewer than 2 segments) may exceed max.
            assert c.end_s - c.start_s <= MAX or len(c.segments) < 2
        starts = [c.start_s for c in out.chunks]
        assert starts == sorted(starts)
