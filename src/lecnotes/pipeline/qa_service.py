from __future__ import annotations

from typing import cast
from uuid import UUID

from lecnotes.domain import KnowledgeUnit, QaLevel, QaRequest, QuestionSet, VideoMeta
from lecnotes.observability.tracing import traced_job
from lecnotes.pipeline.context import RunContext
from lecnotes.pipeline.orchestrator import Orchestrator
from lecnotes.pipeline.stages.qa import QaStage


class QaService:
    """On-demand practice questions for finished notes, via the orchestrator: same request →
    cache hit; every generation traced and recorded under the notes' job."""

    def __init__(self, orchestrator: Orchestrator, stage: QaStage) -> None:
        self._orch = orchestrator
        self._stage = stage

    async def generate(
        self,
        *,
        job_id: UUID,
        video: VideoMeta,
        units: list[KnowledgeUnit],
        level: QaLevel,
        count: int,
    ) -> QuestionSet:
        request = QaRequest(units=units, level=level, count=count)
        with traced_job(str(job_id), video_id=video.video_id):
            out = await self._orch.run_stage(
                self._stage, request, RunContext(job_id=job_id, video=video)
            )
        return cast(QuestionSet, out)
