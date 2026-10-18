"""Section fragmentation: how often the notes, read in lecture order, return to a section
they already left. 0 = every section is one contiguous stretch of the lecture.

    fragmentation = (section runs - distinct sections) / (units - 1)

Rendered notes group units by section, so a fragmented section pulls later material up
next to earlier material and the notes stop following the lecture.
"""

from __future__ import annotations

from dataclasses import dataclass

from lecnotes.domain import KnowledgeUnit


@dataclass(frozen=True)
class SectionScore:
    sections: int
    runs: int
    units: int

    @property
    def fragmentation(self) -> float:
        return (self.runs - self.sections) / (self.units - 1) if self.units > 1 else 0.0


def section_fragmentation(units: list[KnowledgeUnit]) -> SectionScore:
    order = [u.section or "" for u in sorted(units, key=lambda u: (u.start_s, u.id))]
    runs = sum(1 for i, s in enumerate(order) if i == 0 or s != order[i - 1])
    return SectionScore(sections=len(set(order)), runs=runs, units=len(order))
