"""Redundancy: share of units that have a near-duplicate elsewhere in the same notes.

Deliberately uses different signals from the merge stage's grouping (which matches fuzzy
concept names and shared code), so merge isn't graded by its own logic: two units are
duplicates when their *explanations* are very similar, or their core concept names are
*exactly* equal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rapidfuzz import fuzz

from lecnotes.domain import KnowledgeUnit
from lecnotes.pipeline.stages.merge import core_concept

EXPLANATION_RATIO = 75.0


@dataclass(frozen=True)
class RedundancyScore:
    units: int
    duplicated: int
    pairs: list[tuple[str, str]] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.duplicated / self.units if self.units else 0.0


def redundancy(units: list[KnowledgeUnit], *, ratio: float = EXPLANATION_RATIO) -> RedundancyScore:
    pairs: list[tuple[str, str]] = []
    for i, a in enumerate(units):
        for b in units[i + 1 :]:
            same_name = bool(core_concept(a.concept)) and core_concept(a.concept) == core_concept(
                b.concept
            )
            similar = fuzz.token_set_ratio(a.explanation, b.explanation) >= ratio
            if same_name or similar:
                pairs.append((a.id, b.id))
    duplicated = {i for pair in pairs for i in pair}
    return RedundancyScore(units=len(units), duplicated=len(duplicated), pairs=pairs)
