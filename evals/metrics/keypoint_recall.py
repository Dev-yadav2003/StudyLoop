"""Keypoint recall: share of the golden key points that the final notes actually state.

LLM-as-judge, with leniency guards in code: a "covered" verdict must quote the notes, and the
quote must really be in the notes (fuzzy). Key points the judge skips count as not covered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from evals.golden_set import Keypoint
from lecnotes.domain import LLMUsage
from lecnotes.llm.gateway import LLMGateway, ModelOptions
from lecnotes.prompts.loader import load

PROMPT_ID = "judge/keypoint_recall"
QUOTE_RATIO = 85.0


class KeypointVerdict(BaseModel):
    id: str
    covered: bool
    quote: str = Field(description="Sentence(s) copied from the notes; empty if not covered")


class KeypointJudgement(BaseModel):
    verdicts: list[KeypointVerdict]


@dataclass(frozen=True)
class KeypointScore:
    covered: list[str]
    missed: list[str]
    unsupported: list[str] = field(default_factory=list)
    """Judge said covered but its quote isn't in the notes → counted as missed."""
    ungraded: list[str] = field(default_factory=list)
    """Judge skipped these ids → counted as missed."""
    usage: LLMUsage | None = None

    @property
    def recall(self) -> float:
        total = len(self.covered) + len(self.missed)
        return len(self.covered) / total if total else 0.0


def _in_notes(quote: str, notes: str) -> bool:
    q, n = " ".join(quote.split()).lower(), " ".join(notes.split()).lower()
    return bool(q) and (q in n or fuzz.partial_ratio(q, n) >= QUOTE_RATIO)


async def keypoint_recall(
    gateway: LLMGateway,
    notes_markdown: str,
    keypoints: list[Keypoint],
    *,
    prompt_version: int = 1,
    model: str | None = None,
    options: ModelOptions | None = None,
) -> KeypointScore:
    prompt = load(PROMPT_ID, version=prompt_version, keypoints=keypoints, notes=notes_markdown)
    result = await gateway.generate_structured(
        prompt, KeypointJudgement, model=model, options=options
    )

    verdicts = {v.id: v for v in result.output.verdicts}
    covered, missed, unsupported, ungraded = [], [], [], []
    for kp in keypoints:
        verdict = verdicts.get(kp.id)
        if verdict is None:
            ungraded.append(kp.id)
            missed.append(kp.id)
        elif verdict.covered and _in_notes(verdict.quote, notes_markdown):
            covered.append(kp.id)
        else:
            if verdict.covered:
                unsupported.append(kp.id)
            missed.append(kp.id)
    return KeypointScore(
        covered=covered,
        missed=missed,
        unsupported=unsupported,
        ungraded=ungraded,
        usage=result.usage,
    )
