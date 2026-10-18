"""Check every unit's explanation and examples against the evidence it cites (Gemini).

supported → kept; revise → explanation replaced by the verifier's supported-only rewrite and
unsupported examples removed; unsupported → unit dropped. A unit the verifier forgets is
kept and counted as `unverified`: a flaky verifier must not silently delete content.
(Code and board text were already checked deterministically in extract.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from lecnotes.domain import (
    BoardState,
    ExtractedUnits,
    KnowledgeUnit,
    PerceptionResult,
    TranscriptSegment,
    VerifySummary,
)
from lecnotes.llm.gateway import LLMGateway
from lecnotes.observability.logging import get_logger
from lecnotes.pipeline.context import RunContext
from lecnotes.pipeline.stage import Stage
from lecnotes.prompts.loader import load

log = get_logger(__name__)
PROMPT_ID = "verify"


class UnitVerdict(BaseModel):
    unit_id: str
    verdict: Literal["supported", "revise", "unsupported"]
    revised_explanation: str = Field(description="Only for 'revise'; empty otherwise")
    supported_examples: list[str]
    unsupported_claims: list[str]


class VerifyOutput(BaseModel):
    results: list[UnitVerdict]


@dataclass(frozen=True)
class UnitEvidence:
    id: str
    concept: str
    explanation: str
    examples: list[str]
    segments: list[TranscriptSegment]
    boards: list[BoardState]


class VerifyStage(Stage[ExtractedUnits, ExtractedUnits]):
    name = "verify"
    version = 1
    input_model = ExtractedUnits
    output_model = ExtractedUnits

    def __init__(
        self, gateway: LLMGateway, *, model: str, prompt_version: int = 1, batch_size: int = 6
    ) -> None:
        self._gateway = gateway
        self._model = model
        self._prompt_version = prompt_version
        self.batch_size = batch_size

    @property
    def prompt_version(self) -> int | None:
        return self._prompt_version

    @property
    def model(self) -> str | None:
        return self._model

    def config(self, ctx: RunContext) -> dict[str, Any]:
        return {"batch_size": self.batch_size}

    async def run(self, inp: ExtractedUnits, ctx: RunContext) -> ExtractedUnits:
        if inp.evidence is None:
            raise ValueError("verify needs ExtractedUnits.evidence (extract v2+)")
        video = ctx.require_video()
        verdicts: dict[str, UnitVerdict] = {}
        for start in range(0, len(inp.units), self.batch_size):
            batch = inp.units[start : start + self.batch_size]
            prompt = load(
                PROMPT_ID,
                version=self._prompt_version,
                units=[cited_evidence(u, inp.evidence) for u in batch],
            )
            result = await self._gateway.generate_structured(
                prompt,
                VerifyOutput,
                model=self._model,
                metadata={"video_id": video.video_id, "batch": start // self.batch_size},
            )
            ctx.add_usage(result.usage)
            verdicts |= {v.unit_id: v for v in result.output.results}

        kept, dropped = [], list(inp.dropped)
        counts = {"supported": 0, "revised": 0, "dropped": 0, "unverified": 0}
        for unit in inp.units:
            verdict = verdicts.get(unit.id)
            outcome, new_unit = apply_verdict(unit, verdict)
            counts[outcome] += 1
            if new_unit is None:
                claims = "; ".join(verdict.unsupported_claims) if verdict else ""
                dropped.append(f"verify: {unit.id} ({unit.concept!r}) unsupported: {claims}")
            else:
                kept.append(new_unit)
                if outcome == "revised" and verdict and verdict.unsupported_claims:
                    dropped.append(
                        f"verify: {unit.id} revised, removed: "
                        + "; ".join(verdict.unsupported_claims)
                    )
        summary = VerifySummary(checked=len(inp.units), **counts)
        log.info("verify.done", **summary.model_dump())
        return inp.model_copy(update={"units": kept, "dropped": dropped, "verify": summary})


def apply_verdict(
    unit: KnowledgeUnit, verdict: UnitVerdict | None
) -> tuple[str, KnowledgeUnit | None]:
    if verdict is None:
        return "unverified", unit
    if verdict.verdict == "unsupported":
        return "dropped", None
    supported_examples = set(verdict.supported_examples)
    examples = [e for e in unit.examples if e in supported_examples]
    if verdict.verdict == "supported":
        return "supported", unit.model_copy(update={"examples": examples})
    revised = verdict.revised_explanation.strip()
    if not revised:  # "revise" with nothing left = nothing supported
        return "dropped", None
    return "revised", unit.model_copy(update={"explanation": revised, "examples": examples})


def cited_evidence(unit: KnowledgeUnit, evidence: PerceptionResult) -> UnitEvidence:
    """The unit plus only the evidence it cites (what verifiers and judges may look at)."""
    cited_segments = {i for e in unit.evidence for i in e.segment_ids}
    cited_boards = {i for e in unit.evidence for i in e.board_state_ids}
    return UnitEvidence(
        id=unit.id,
        concept=unit.concept,
        explanation=unit.explanation,
        examples=unit.examples,
        segments=[s for s in evidence.segments if s.id in cited_segments],
        boards=[b for b in evidence.board_states if b.id in cited_boards],
    )
