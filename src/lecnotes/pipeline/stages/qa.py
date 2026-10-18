"""Practice questions from finished notes (Gemini, text only), grounded in code:

- every question must cite note units that exist; a question citing none is dropped;
- timestamps are the cited units' primary teaching moments — never written by the model —
  so each question links to where its answer is taught;
- duplicates are dropped and the requested level is enforced.

Runs on demand (not part of the job chain) through the orchestrator, so a repeated request
for the same notes/level/count is a cache hit, and every generation is traced and costed.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from lecnotes.domain import (
    QaLevel,
    QaRequest,
    Question,
    QuestionLevel,
    QuestionSet,
    VideoSpan,
)
from lecnotes.llm.gateway import LLMGateway
from lecnotes.observability.logging import get_logger
from lecnotes.pipeline.context import RunContext
from lecnotes.pipeline.stage import Stage
from lecnotes.prompts.loader import load

log = get_logger(__name__)
PROMPT_ID = "qa"


class QaItem(BaseModel):
    level: Literal["recall", "application"]
    question: str
    answer: str
    unit_ids: list[str] = Field(description="Ids of the notes the answer relies on")


class QaOutput(BaseModel):
    questions: list[QaItem]


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


class QaStage(Stage[QaRequest, QuestionSet]):
    name = "qa"
    version = 2  # v2: one timestamp per distinct link (cached v1 sets repeated 00:00)
    input_model = QaRequest
    output_model = QuestionSet

    def __init__(self, gateway: LLMGateway, *, model: str, prompt_version: int = 1) -> None:
        self._gateway = gateway
        self._model = model
        self._prompt_version = prompt_version

    @property
    def prompt_version(self) -> int | None:
        return self._prompt_version

    @property
    def model(self) -> str | None:
        return self._model

    async def run(self, inp: QaRequest, ctx: RunContext) -> QuestionSet:
        prompt = load(
            PROMPT_ID,
            version=self._prompt_version,
            units=inp.units,
            level=inp.level.value,
            count=inp.count,
        )
        result = await self._gateway.generate_structured(
            prompt,
            QaOutput,
            model=self._model,
            metadata={"video_id": ctx.video.video_id if ctx.video else None},
        )
        ctx.add_usage(result.usage)
        out = to_question_set(result.output, inp)
        log.info("qa.done", questions=len(out.questions), dropped=len(out.dropped))
        return out


def to_question_set(raw: QaOutput, request: QaRequest) -> QuestionSet:
    units = {u.id: u for u in request.units}
    questions: list[Question] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for i, item in enumerate(raw.questions, start=1):
        where = f"question {i} ({item.question[:60]!r})"
        cited = [u for u in dict.fromkeys(item.unit_ids) if u in units]
        if not cited:
            dropped.append(f"{where}: cites no existing note")
            continue
        if not item.question.strip() or not item.answer.strip():
            dropped.append(f"{where}: empty question or answer")
            continue
        if _norm(item.question) in seen:
            dropped.append(f"{where}: duplicate")
            continue
        seen.add(_norm(item.question))
        level = (
            QuestionLevel(item.level)
            if request.level is QaLevel.MIXED
            else QuestionLevel(request.level.value)
        )
        spans = _primary_spans([units[u].spans[0] for u in cited])
        questions.append(
            Question(
                id=f"q{len(questions) + 1:02d}",
                level=level,
                question=item.question,
                answer=item.answer,
                unit_ids=cited,
                spans=spans,
            )
        )
        if len(questions) == request.count:
            break
    return QuestionSet(questions=questions, dropped=dropped)


def _primary_spans(spans: list[VideoSpan]) -> list[VideoSpan]:
    """One per distinct link: two notes taught from the same second link to the same place
    (the UI keyed chips by link, and showed `00:00` twice)."""
    unique: dict[tuple[str, int], VideoSpan] = {}
    for s in spans:
        unique.setdefault((s.video_id, int(s.start_s)), s)
    return sorted(unique.values(), key=lambda s: s.start_s)
