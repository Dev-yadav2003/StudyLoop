"""Faithfulness: share of atomic claims in the final notes that the cited evidence supports.

Independent from the verify stage (different prompt, configurable judge model), so verify
is not graded by its own logic. Units the judge skips are reported, not silently counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from lecnotes.domain import KnowledgeUnit, LLMUsage, PerceptionResult
from lecnotes.llm.gateway import LLMGateway, ModelOptions
from lecnotes.pipeline.stages.verify import cited_evidence  # same evidence selection
from lecnotes.prompts.loader import load

PROMPT_ID = "judge/faithfulness"


class Claim(BaseModel):
    claim: str
    supported: bool


class UnitClaims(BaseModel):
    unit_id: str
    claims: list[Claim]


class FaithfulnessJudgement(BaseModel):
    units: list[UnitClaims]


@dataclass(frozen=True)
class FaithfulnessScore:
    supported: int
    total: int
    unsupported_claims: dict[str, list[str]] = field(default_factory=dict)
    ungraded: list[str] = field(default_factory=list)
    usage: list[LLMUsage] = field(default_factory=list)

    @property
    def score(self) -> float | None:
        return self.supported / self.total if self.total else None


async def faithfulness(
    gateway: LLMGateway,
    units: list[KnowledgeUnit],
    evidence: PerceptionResult,
    *,
    model: str | None = None,
    batch_size: int = 6,
    prompt_version: int = 1,
    options: ModelOptions | None = None,
) -> FaithfulnessScore:
    graded: dict[str, UnitClaims] = {}
    usage: list[LLMUsage] = []
    for start in range(0, len(units), batch_size):
        batch = [cited_evidence(u, evidence) for u in units[start : start + batch_size]]
        prompt = load(PROMPT_ID, version=prompt_version, units=batch)
        result = await gateway.generate_structured(
            prompt, FaithfulnessJudgement, model=model, options=options
        )
        usage.append(result.usage)
        graded |= {u.unit_id: u for u in result.output.units}

    supported = total = 0
    unsupported: dict[str, list[str]] = {}
    ungraded = []
    for unit in units:
        claims = graded.get(unit.id)
        if claims is None:
            ungraded.append(unit.id)
            continue
        total += len(claims.claims)
        supported += sum(c.supported for c in claims.claims)
        bad = [c.claim for c in claims.claims if not c.supported]
        if bad:
            unsupported[unit.id] = bad
    return FaithfulnessScore(
        supported=supported,
        total=total,
        unsupported_claims=unsupported,
        ungraded=ungraded,
        usage=usage,
    )
