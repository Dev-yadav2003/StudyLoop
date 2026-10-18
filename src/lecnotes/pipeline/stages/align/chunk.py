"""Group aligned evidence into 2-5 minute chunks: the unit of work for `extract` (no AI).

1. Blocks: each board's core window + the segments whose midpoint falls in it. Speech before
   the first board is a board-less lead-in block; a board-less video is one block.
2. Partition blocks optimally (DP): no chunk over `max_s`, as little time under `min_s` as
   possible, then as few chunks as possible — fuller chunks = more context per LLM call and
   fewer requests against the free tier's daily limit. Cuts happen only between blocks.
3. A block longer than `max_s` (one slide talked over for 10 min) is split at *segment*
   boundaries and the board is copied, intact, into each piece: board text is never cut.
4. Boards linked to a chunk's speech outside their window (lexical / extended, strength
   ≥ `attach_strength`) are attached too: that's how back-references reach `extract`.

Every segment ends up in exactly one chunk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lecnotes.domain import (
    AlignedChunk,
    Alignment,
    AlignmentLink,
    BoardState,
    Chunks,
    TranscriptSegment,
)
from lecnotes.pipeline.context import RunContext
from lecnotes.pipeline.stage import Stage
from lecnotes.pipeline.stages.align.lexical import link_strength
from lecnotes.pipeline.stages.align.temporal import midpoint, windows

MIN_SPAN_S = 1.0


@dataclass
class _Piece:
    start: float
    end: float
    board_ids: list[str]
    segments: list[TranscriptSegment] = field(default_factory=list)

    @property
    def span(self) -> float:
        return self.end - self.start


def _blocks(alignment: Alignment, duration_s: float) -> list[_Piece]:
    ev = alignment.evidence
    wins = windows(ev.board_states, duration_s)
    if not wins:
        return [_Piece(0.0, duration_s, [], sorted(ev.segments, key=lambda s: s.start_s))]

    pieces = [_Piece(w.start, w.end, [w.board_id]) for w in wins]
    if wins[0].start > 0:
        pieces.insert(0, _Piece(0.0, wins[0].start, []))
    for s in sorted(ev.segments, key=lambda s: s.start_s):
        t = midpoint(s)
        home = next(
            (p for p in reversed(pieces) if p.start <= t),
            pieces[0],
        )
        home.segments.append(s)
    return pieces


def _split(block: _Piece, max_s: float) -> list[_Piece]:
    """Pieces with speech never exceed `max_s` (unless one segment alone does). Silence that
    doesn't fit — before the first segment or after the last — becomes a segment-less piece
    (board still attached), so it can't stretch a piece past the limit."""
    if block.span <= max_s:
        return [block]
    ids = block.board_ids
    out: list[_Piece] = []
    current = _Piece(block.start, block.start, ids)
    for s in block.segments:
        if s.end_s - current.start > max_s:
            if current.segments:
                current.end = current.segments[-1].end_s
                out.append(current)
                current = _Piece(current.end, current.end, ids)
            if s.end_s - current.start > max_s and s.start_s > current.start:
                out.append(_Piece(current.start, s.start_s, ids))  # long silence before s
                current = _Piece(s.start_s, s.start_s, ids)
        current.segments.append(s)
    if current.segments and block.end - current.start > max_s:
        current.end = current.segments[-1].end_s
        out += [current, _Piece(current.end, block.end, ids)]  # long silence after
    else:
        current.end = block.end
        out.append(current)
    return out


_SHORTFALL_WEIGHT = 1_000.0  # 1 s below min_s costs more than any number of extra chunks


def _pack(pieces: list[_Piece], min_s: float, max_s: float) -> list[list[_Piece]]:
    """Optimal partition of consecutive pieces (DP over cut points): minimise time below
    `min_s` first, then the number of chunks. Greedy filling fails on tails — e.g. slides of
    70+70+70+70+40 s greedily give 280 + a lone 40 s; the DP finds 140 + 180."""
    n = len(pieces)
    best = [0.0] + [float("inf")] * n
    cut = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i - 1, -1, -1):
            span = pieces[i - 1].end - pieces[j].start
            if span > max_s and i - j > 1:
                break  # spans only grow as j moves left; a lone oversized piece is allowed
            cost = best[j] + 1 + _SHORTFALL_WEIGHT * max(0.0, min_s - span)
            if cost < best[i]:
                best[i], cut[i] = cost, j
    groups: list[list[_Piece]] = []
    i = n
    while i > 0:
        groups.append(pieces[cut[i] : i])
        i = cut[i]
    return groups[::-1]


class ChunkStage(Stage[Alignment, Chunks]):
    name = "chunk"
    version = 1
    input_model = Alignment
    output_model = Chunks

    def __init__(
        self, *, min_s: float = 120.0, max_s: float = 300.0, attach_strength: float = 0.5
    ) -> None:
        if not 0 < min_s <= max_s:
            raise ValueError("need 0 < min_s <= max_s")
        self.min_s = min_s
        self.max_s = max_s
        self.attach_strength = attach_strength

    def config(self, ctx: RunContext) -> dict[str, Any]:
        return {"min_s": self.min_s, "max_s": self.max_s, "attach_strength": self.attach_strength}

    async def run(self, inp: Alignment, ctx: RunContext) -> Chunks:
        video = ctx.require_video()
        pieces = [p for b in _blocks(inp, video.duration_s) for p in _split(b, self.max_s)]
        groups = _pack(pieces, self.min_s, self.max_s)
        boards = {b.id: b for b in inp.evidence.board_states}
        strength = link_strength(inp.links)
        return Chunks(
            chunks=[
                self._chunk(i, g, video.video_id, boards, strength, inp.links)
                for i, g in enumerate(groups, start=1)
            ]
        )

    def _chunk(
        self,
        index: int,
        group: list[_Piece],
        video_id: str,
        boards: dict[str, BoardState],
        strength: dict[tuple[str, str], float],
        links: list[AlignmentLink],
    ) -> AlignedChunk:
        segments = [s for p in group for s in p.segments]
        seg_ids = {s.id for s in segments}
        board_ids = list(dict.fromkeys(b for p in group for b in p.board_ids))
        for (seg_id, board_id), value in sorted(strength.items()):
            if seg_id in seg_ids and value >= self.attach_strength and board_id not in board_ids:
                board_ids.append(board_id)
        chunk_boards = sorted((boards[b] for b in board_ids), key=lambda b: b.t_first_seen)
        start, end = group[0].start, group[-1].end
        return AlignedChunk(
            id=f"c{index:03d}",
            video_id=video_id,
            start_s=start,
            end_s=max(end, start + MIN_SPAN_S),
            segments=segments,
            board_states=chunk_boards,
            links=[
                lk for lk in links if lk.segment_id in seg_ids and lk.board_state_id in board_ids
            ],
        )
