from __future__ import annotations

from typing import cast
from uuid import UUID

from lecnotes.domain import ExtractedUnits, NotesDocument, UnitEdit
from lecnotes.observability.tracing import traced_job
from lecnotes.pipeline.context import RunContext
from lecnotes.pipeline.orchestrator import Orchestrator
from lecnotes.pipeline.stages.render import RenderStage


class UnitNotFound(LookupError):
    pass


class NotesEditor:
    """Apply a user's edit to one unit and re-render — render only, nothing upstream is
    reprocessed and no model is called. Runs through the orchestrator, so the re-render is a
    recorded, traced stage run under the notes' job."""

    def __init__(self, orchestrator: Orchestrator, render: RenderStage) -> None:
        self._orch = orchestrator
        self._render = render

    async def edit(
        self, *, job_id: UUID, doc: NotesDocument, unit_id: str, edit: UnitEdit
    ) -> NotesDocument:
        if not any(u.id == unit_id for u in doc.units):
            raise UnitNotFound(unit_id)
        units = [edit.apply(u) if u.id == unit_id else u for u in doc.units]
        inp = ExtractedUnits(units=units, evidence=doc.evidence, verify=doc.verify)
        with traced_job(str(job_id), video_id=doc.video.video_id) as span:
            span.update(metadata={"edited_unit": unit_id})
            out = await self._orch.run_stage(
                self._render, inp, RunContext(job_id=job_id, video=doc.video)
            )
        return cast(NotesDocument, out)
