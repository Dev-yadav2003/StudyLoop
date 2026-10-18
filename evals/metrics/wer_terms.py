"""Speech accuracy: word error rate overall, and on technical terms.

Overall WER hides what matters most for notes: a lecture can have 8% WER and still turn
`enumerate` into "a numerate" every time. The term error rate counts how many reference
occurrences of each technical term also occur in the hypothesis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import jiwer

_NON_WORD = re.compile(r"[^a-z0-9']+")


def normalize(text: str) -> str:
    """Lowercase, punctuation → space, collapse whitespace. Applied to both sides equally."""
    return " ".join(_NON_WORD.sub(" ", text.lower()).split())


def wer(reference: str, hypothesis: str) -> float:
    ref = normalize(reference)
    if not ref:
        raise ValueError("reference transcript is empty")
    return float(jiwer.wer(ref, normalize(hypothesis)))


def _count(term: str, text: str) -> int:
    norm_term = normalize(term)
    if not norm_term:
        return 0
    return len(re.findall(rf"(?<!\S){re.escape(norm_term)}(?!\S)", text))


@dataclass(frozen=True)
class TermScore:
    error_rate: float
    """1 - found/expected over all term occurrences; 0 = every term occurrence recovered."""
    per_term: dict[str, tuple[int, int]]
    """term -> (occurrences in reference, occurrences in hypothesis)."""

    @property
    def missed(self) -> list[str]:
        return [t for t, (ref, hyp) in self.per_term.items() if hyp < ref]


def term_error_rate(reference: str, hypothesis: str, terms: list[str]) -> TermScore | None:
    """None when none of the terms occur in the reference (nothing to measure)."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    per_term = {t: (_count(t, ref), _count(t, hyp)) for t in terms}
    per_term = {t: counts for t, counts in per_term.items() if counts[0] > 0}
    expected = sum(r for r, _ in per_term.values())
    if expected == 0:
        return None
    found = sum(min(r, h) for r, h in per_term.values())
    return TermScore(error_rate=1 - found / expected, per_term=per_term)
