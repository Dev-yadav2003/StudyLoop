"""Board recall: what fraction of the hand-typed board/slide/code lines did perception capture?

A reference line counts as captured if some predicted line is similar enough
(rapidfuzz `ratio` >= threshold). Choices:
  - case-sensitive: code is case-sensitive, and `List` vs `list` is a real error;
  - whitespace collapsed within a line: indentation is checked by code tests later, not here;
  - `ratio`, not `partial_ratio`: a short reference line must not "match" inside a long one;
  - reference lines containing [unclear] (the human couldn't read them either) are skipped.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process

from evals.golden_set import RefBoard
from lecnotes.domain import UNCLEAR_MARKER, BoardState

DEFAULT_THRESHOLD = 85.0
MIN_LINE_CHARS = 2


def _norm(line: str) -> str:
    return " ".join(line.split())


def _lines(texts: Iterable[str]) -> list[str]:
    return [
        n for text in texts for line in text.splitlines() if len(n := _norm(line)) >= MIN_LINE_CHARS
    ]


@dataclass(frozen=True)
class BoardRecall:
    matched: int
    total: int
    misses: list[str] = field(default_factory=list)
    by_type: dict[str, tuple[int, int]] = field(default_factory=dict)
    """board type -> (matched, total)."""

    @property
    def recall(self) -> float | None:
        return self.matched / self.total if self.total else None


def board_recall(
    reference: list[RefBoard],
    predicted: list[BoardState],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> BoardRecall:
    candidates = _lines(b.text for b in predicted)
    matched, total, misses = 0, 0, []
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for board in reference:
        for line in _lines(board.lines):
            if UNCLEAR_MARKER in line:
                continue
            total += 1
            by_type[board.type][1] += 1
            best = process.extractOne(line, candidates, scorer=fuzz.ratio) if candidates else None
            if best is not None and best[1] >= threshold:
                matched += 1
                by_type[board.type][0] += 1
            else:
                misses.append(line)
    return BoardRecall(
        matched=matched,
        total=total,
        misses=misses,
        by_type={t: (m, n) for t, (m, n) in by_type.items()},
    )
