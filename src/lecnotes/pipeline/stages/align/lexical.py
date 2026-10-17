"""Lexical alignment: does the speech mention what's on the board?

Board identifiers are fuzzy-matched against the sentence's words *and adjacent word pairs
glued together*, with rapidfuzz `ratio`. That catches ASR splits ("a numerate" →
`anumerate` ≈ `enumerate`) without the false positives of `partial_ratio` over the whole
sentence (`data` would match inside "updated").

Lexical links may reach boards outside the temporal windows (a teacher referring back to an
earlier slide), within `window_s` of the board's visible span. Pointing phrases ("as you can
see", "here") add a small boost only where a temporal link already exists.
"""

from __future__ import annotations

import keyword
import re
from collections import defaultdict
from collections.abc import Iterable
from itertools import pairwise

from rapidfuzz import fuzz

from lecnotes.domain import AlignmentLink, AlignmentMethod, BoardState, TranscriptSegment

MIN_TOKEN_CHARS = 4
MATCH_RATIO = 85.0
POINTING_BOOST = 0.1

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_WORD = re.compile(r"[a-z0-9_]+")
_STOP = frozenset(keyword.kwlist) | {
    "about", "also", "because", "been", "from", "have", "here", "into", "just", "like",
    "more", "only", "some", "than", "that", "their", "them", "then", "there", "these",
    "they", "this", "those", "very", "want", "what", "when", "where", "which", "will",
    "with", "would", "your", "self", "true", "false", "none", "print",
}  # fmt: skip
_POINTING = re.compile(
    r"\b(this|these|here|over here|as you can see|look at|on the (board|screen|slide))\b"
)


def board_tokens(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for token in _IDENT.findall(text):
        t = token.lower()
        if len(t) >= MIN_TOKEN_CHARS and t not in _STOP:
            seen[t] = None
    return list(seen)


def _candidates(sentence: str) -> set[str]:
    words = _WORD.findall(sentence.lower())
    return set(words) | {a + b for a, b in pairwise(words)}


def lexical_score(tokens: Iterable[str], sentence: str) -> float:
    """0 = no board identifier mentioned; 0.5 = one; 1.0 = two or more."""
    candidates = _candidates(sentence)
    if not candidates:
        return 0.0
    matched = sum(1 for t in tokens if any(fuzz.ratio(t, c) >= MATCH_RATIO for c in candidates))
    return min(1.0, matched / 2)


def is_pointing(sentence: str) -> bool:
    return bool(_POINTING.search(sentence.lower()))


def lexical_links(
    segments: list[TranscriptSegment],
    boards: list[BoardState],
    temporal: list[AlignmentLink],
    *,
    window_s: float = 180.0,
) -> list[AlignmentLink]:
    temporal_pairs = {(lk.segment_id, lk.board_state_id) for lk in temporal}
    tokens = {b.id: board_tokens(b.text) for b in boards}
    links = []
    for s in segments:
        mid = (s.start_s + s.end_s) / 2
        pointing = is_pointing(s.text)
        for b in boards:
            if (
                not tokens[b.id]
                or mid < b.t_first_seen - window_s
                or mid > b.t_last_seen + window_s
            ):
                continue
            score = lexical_score(tokens[b.id], s.text)
            if score == 0:
                continue
            if pointing and (s.id, b.id) in temporal_pairs:
                score = min(1.0, score + POINTING_BOOST)
            links.append(
                AlignmentLink(
                    segment_id=s.id,
                    board_state_id=b.id,
                    method=AlignmentMethod.LEXICAL,
                    score=score,
                )
            )
    return links


def link_strength(links: Iterable[AlignmentLink]) -> dict[tuple[str, str], float]:
    """Combined evidence per (segment, board): best temporal score + half the lexical score.
    A lexical match lifts an extended-window link; a lexical match alone reaches 0.5."""
    temporal: dict[tuple[str, str], float] = defaultdict(float)
    lexical: dict[tuple[str, str], float] = defaultdict(float)
    for lk in links:
        pair = (lk.segment_id, lk.board_state_id)
        bucket = lexical if lk.method is AlignmentMethod.LEXICAL else temporal
        bucket[pair] = max(bucket[pair], lk.score)
    return {
        pair: min(1.0, temporal[pair] + 0.5 * lexical[pair])
        for pair in temporal.keys() | lexical.keys()
    }
