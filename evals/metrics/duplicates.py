"""Duplicate rate (LLM judge): share of notes that repeat a concept another note already
teaches. For a group of k notes about one concept, k-1 are redundant.

Independent of the merge stage's candidate grouping (fuzzy names / shared code), which
misses duplicates whose names differ — the case this metric exists to catch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from lecnotes.domain import KnowledgeUnit, LLMUsage
from lecnotes.llm.gateway import LLMGateway, ModelOptions
from lecnotes.prompts.loader import load

PROMPT_ID = "judge/duplicates"


class DuplicateGroup(BaseModel):
    unit_ids: list[str]
    concept: str


class DuplicateJudgement(BaseModel):
    groups: list[DuplicateGroup]


@dataclass(frozen=True)
class DuplicateScore:
    units: int
    redundant: int
    groups: list[list[str]] = field(default_factory=list)
    usage: list[LLMUsage] = field(default_factory=list)

    @property
    def rate(self) -> float | None:
        return self.redundant / self.units if self.units else None


async def duplicate_rate(
    gateway: LLMGateway,
    units: list[KnowledgeUnit],
    *,
    model: str | None = None,
    prompt_version: int = 1,
    options: ModelOptions | None = None,
) -> DuplicateScore:
    if len(units) < 2:
        return DuplicateScore(units=len(units), redundant=0)
    prompt = load(PROMPT_ID, version=prompt_version, units=units)
    result = await gateway.generate_structured(
        prompt, DuplicateJudgement, model=model, options=options
    )
    known = {u.id for u in units}
    seen: set[str] = set()
    groups: list[list[str]] = []
    for g in result.output.groups:  # keep the judge honest: known ids, each once
        ids = [i for i in dict.fromkeys(g.unit_ids) if i in known and i not in seen]
        if len(ids) >= 2:
            seen.update(ids)
            groups.append(ids)
    return DuplicateScore(
        units=len(units),
        redundant=sum(len(g) - 1 for g in groups),
        groups=groups,
        usage=[result.usage],
    )
