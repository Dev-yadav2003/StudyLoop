"""Temporal alignment (Lecture2Notes' rule): speech from a board state's start to the next
state's start belongs to that state. Segments are sentence-sized, so a segment is assigned by
its midpoint ("snapped to sentence boundaries").

Links are weak and many-to-many by design (LPM paper): besides the core window, a state
still visible after the next one appeared (picture-in-picture, a slide beside code) gets a
"visible" link, and speech shortly before/after the window gets a decaying "extended" link —
teachers announce a slide before it appears and keep talking about it after it's gone.
"""

from __future__ import annotations

from dataclasses import dataclass

from lecnotes.domain import AlignmentLink, AlignmentMethod, BoardState, TranscriptSegment

CORE_SCORE = 1.0
VISIBLE_SCORE = 0.6
EXTENDED_MAX = 0.5
EXTENDED_MIN = 0.1


@dataclass(frozen=True)
class Window:
    board_id: str
    start: float
    end: float
    """Next state's start (or the video end for the last state)."""
    visible_until: float


def windows(boards: list[BoardState], duration_s: float) -> list[Window]:
    ordered = sorted(boards, key=lambda b: b.t_first_seen)
    out = []
    for i, b in enumerate(ordered):
        end = ordered[i + 1].t_first_seen if i + 1 < len(ordered) else duration_s
        out.append(Window(b.id, b.t_first_seen, max(end, b.t_first_seen), b.t_last_seen))
    return out


def midpoint(segment: TranscriptSegment) -> float:
    return (segment.start_s + segment.end_s) / 2


def _decay(distance: float, width: float) -> float:
    return max(EXTENDED_MIN, EXTENDED_MAX * (1 - distance / width))


def temporal_links(
    segments: list[TranscriptSegment],
    boards: list[BoardState],
    duration_s: float,
    *,
    before_s: float = 15.0,
    after_s: float = 30.0,
) -> list[AlignmentLink]:
    links = []
    for w in windows(boards, duration_s):
        for s in segments:
            t = midpoint(s)
            if w.start <= t < w.end or (t == w.end == duration_s):
                score = CORE_SCORE
            elif w.end <= t <= w.visible_until:
                score = VISIBLE_SCORE
            elif w.start - before_s <= t < w.start:
                score = _decay(w.start - t, before_s)
            elif max(w.end, w.visible_until) < t <= max(w.end, w.visible_until) + after_s:
                score = _decay(t - max(w.end, w.visible_until), after_s)
            else:
                continue
            links.append(
                AlignmentLink(
                    segment_id=s.id,
                    board_state_id=w.board_id,
                    method=AlignmentMethod.TEMPORAL,
                    score=round(score, 4),
                )
            )
    return links
