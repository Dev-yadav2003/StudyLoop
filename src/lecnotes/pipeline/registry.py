"""The pipeline as a whole, for the one video a link points at (ADR 0007):

    resolve → perceive → validate → align → chunk → extract → (merge) → (verify) → render

Construction of concrete adapters lives in `lecnotes.bootstrap` (composition root).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast
from uuid import UUID

from lecnotes.domain import JobOptions, NotesDocument, VideoMeta
from lecnotes.observability.logging import bind_context, get_logger
from lecnotes.observability.tracing import traced_job
from lecnotes.pipeline.context import RunContext
from lecnotes.pipeline.orchestrator import AnyStage, Orchestrator, check_chain
from lecnotes.pipeline.stages.resolve import ResolveRequest, ResolveStage

log = get_logger(__name__)

VIDEO_STAGE_ORDER = (
    "perceive", "validate", "align", "chunk", "extract", "merge", "verify", "render",
)  # fmt: skip
OPTIONAL_STAGES = frozenset({"merge", "verify"})  # feature-flagged


class Pipeline:
    def __init__(
        self,
        orchestrator: Orchestrator,
        resolve: ResolveStage,
        video_stages: Sequence[AnyStage],
    ) -> None:
        names = tuple(s.name for s in video_stages)
        # The fixed order, with any optional stages left out.
        expected = tuple(n for n in VIDEO_STAGE_ORDER if n in names or n not in OPTIONAL_STAGES)
        if names != expected:
            raise ValueError(
                f"video stages must follow {VIDEO_STAGE_ORDER} (optional: "
                f"{sorted(OPTIONAL_STAGES)}), got {names}"
            )
        check_chain(video_stages, video_stages[0].input_model)
        if video_stages[-1].output_model is not NotesDocument:
            raise TypeError("the last video stage must produce a NotesDocument")
        self._orch = orchestrator
        self._resolve = resolve
        self._stages = list(video_stages)

    @property
    def stage_names(self) -> list[str]:
        return [self._resolve.name, *(s.name for s in self._stages)]

    def describe(self) -> dict[str, dict[str, object]]:
        """Version, prompt version and model of every stage — recorded with eval runs."""
        return {
            s.name: {"version": s.version, "prompt_version": s.prompt_version, "model": s.model}
            for s in [self._resolve, *self._stages]
        }

    async def run(
        self,
        url: str,
        *,
        job_id: UUID,
        options: JobOptions | None = None,
        rerun_from: str | None = None,
    ) -> NotesDocument:
        opts = options or JobOptions()
        with traced_job(str(job_id)):
            video = cast(
                VideoMeta,
                await self._orch.run_stage(
                    self._resolve, ResolveRequest(url=url), RunContext(job_id=job_id, options=opts)
                ),
            )
            ctx = RunContext(job_id=job_id, options=opts, video=video)
            with bind_context(video_id=video.video_id):
                doc = cast(
                    NotesDocument,
                    await self._orch.run_chain(self._stages, video, ctx, rerun_from=rerun_from),
                )
            log.info("pipeline.done", video_id=video.video_id, units=len(doc.units))
            return doc
