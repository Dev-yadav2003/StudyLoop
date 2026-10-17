"""QA correctness: share of practice questions whose answer is correct AND answerable from the
notes they cite alone (step 9.1).

The judge sees each question with only its cited notes, so a question that leans on outside
knowledge or on a note it didn't cite fails, even if the answer is true in general.
Questions the judge skips are reported, not silently counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from lecnotes.domain import KnowledgeUnit, LLMUsage, Question
from lecnotes.llm.gateway import LLMGateway, ModelOptions
from lecnotes.prompts.loader import load

PROMPT_ID = "judge/qa_correctness"


class QuestionVerdict(BaseModel):
    question_id: str
    grounded: bool
    correct: bool
    reason: str


class QaJudgement(BaseModel):
    verdicts: list[QuestionVerdict]


@dataclass(frozen=True)
class QaCorrectnessScore:
    passed: int
    total: int
    failures: dict[str, str] = field(default_factory=dict)
    ungraded: list[str] = field(default_factory=list)
    usage: list[LLMUsage] = field(default_factory=list)

    @property
    def score(self) -> float | None:
        return self.passed / self.total if self.total else None


async def qa_correctness(
    gateway: LLMGateway,
    questions: list[Question],
    units: list[KnowledgeUnit],
    *,
    model: str | None = None,
    prompt_version: int = 1,
    options: ModelOptions | None = None,
) -> QaCorrectnessScore:
    if not questions:
        return QaCorrectnessScore(passed=0, total=0)
    by_id = {u.id: u for u in units}
    items = [
        {"question": q, "notes": [by_id[u] for u in q.unit_ids if u in by_id]} for q in questions
    ]
    prompt = load(PROMPT_ID, version=prompt_version, items=items)
    result = await gateway.generate_structured(prompt, QaJudgement, model=model, options=options)
    verdicts = {v.question_id: v for v in result.output.verdicts}

    passed = total = 0
    failures: dict[str, str] = {}
    ungraded = []
    for q in questions:
        v = verdicts.get(q.id)
        if v is None:
            ungraded.append(q.id)
            continue
        total += 1
        if v.grounded and v.correct:
            passed += 1
        else:
            failures[q.id] = v.reason
    return QaCorrectnessScore(
        passed=passed, total=total, failures=failures, ungraded=ungraded, usage=[result.usage]
    )
