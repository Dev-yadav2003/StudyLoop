from __future__ import annotations

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

from apps.api.deps import Container
from apps.api.schemas import (
    ErrorOut,
    NotesOut,
    QaCreate,
    QuestionOut,
    QuestionSetOut,
    TimestampLink,
    UnitRef,
)
from lecnotes.domain import UnitEdit
from lecnotes.pipeline.notes_editor import UnitNotFound
from lecnotes.storage.repositories.notes import StoredNotes
from lecnotes.timefmt import format_timestamp

router = APIRouter(prefix="/v1/notes", tags=["notes"])


async def _load(c: Container, notes_id: UUID) -> StoredNotes:
    stored = await c.notes.get(notes_id)
    if stored is None:
        raise HTTPException(404, "Notes not found.")
    return stored


def _out(stored: StoredNotes) -> NotesOut:
    doc = stored.document
    return NotesOut(
        id=stored.id,
        job_id=stored.job_id,
        title=doc.title,
        video=doc.video,
        markdown=doc.markdown,
        units=doc.units,
        edited_units=sorted(stored.edited_units),
        revision=stored.revision,
    )


@router.get("/{notes_id}", response_model=NotesOut, responses={404: {"model": ErrorOut}})
async def get_notes(notes_id: UUID, c: Container) -> NotesOut:
    return _out(await _load(c, notes_id))


@router.patch(
    "/{notes_id}/units/{unit_id}",
    response_model=NotesOut,
    responses={404: {"model": ErrorOut}, 409: {"model": ErrorOut}, 422: {"model": ErrorOut}},
)
async def edit_unit(notes_id: UUID, unit_id: str, edit: UnitEdit, c: Container) -> NotesOut:
    """Correct one unit's content; the notes are re-rendered (no reprocessing, no AI call).
    Where it was taught and what it cites can't be edited. Note: an admin rerun of the job
    replaces the notes, edits included."""
    stored = await _load(c, notes_id)
    try:
        doc = await c.services.editor.edit(
            job_id=stored.job_id, doc=stored.document, unit_id=unit_id, edit=edit
        )
    except UnitNotFound:
        raise HTTPException(404, "Unit not found in these notes.") from None
    saved = await c.notes.save_edit(stored, doc, unit_id)
    if saved is None:
        raise HTTPException(409, "These notes were changed meanwhile. Reload and try again.")
    return _out(saved)


@router.get(
    "/{notes_id}/export",
    response_class=PlainTextResponse,
    responses={200: {"content": {"text/markdown": {}}}, 404: {"model": ErrorOut}},
)
async def export_notes(
    notes_id: UUID, c: Container, format: Literal["md"] = Query("md")
) -> PlainTextResponse:
    stored = await _load(c, notes_id)
    filename = f"{stored.document.video.video_id}.md"
    return PlainTextResponse(
        stored.document.markdown,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post(
    "/{notes_id}/qa",
    response_model=QuestionSetOut,
    responses={404: {"model": ErrorOut}, 422: {"model": ErrorOut}, 503: {"model": ErrorOut}},
)
async def practice_questions(notes_id: UUID, body: QaCreate, c: Container) -> QuestionSetOut:
    """Grounded practice questions. Each cites the notes its answer comes from and links to
    where they are taught. The same request again is served from the cache."""
    stored = await _load(c, notes_id)
    doc = stored.document
    if not doc.units:
        raise HTTPException(422, "These notes have no content to ask questions about.")
    qset = await c.services.qa.generate(
        job_id=stored.job_id, video=doc.video, units=doc.units, level=body.level, count=body.count
    )
    concepts = {u.id: u.concept for u in doc.units}
    return QuestionSetOut(
        notes_id=stored.id,
        questions=[
            QuestionOut(
                id=q.id,
                level=q.level,
                question=q.question,
                answer=q.answer,
                units=[UnitRef(id=u, concept=concepts[u]) for u in q.unit_ids],
                timestamps=[
                    TimestampLink(
                        start_s=s.start_s,
                        label=format_timestamp(s.start_s),
                        url=f"https://youtu.be/{s.video_id}?t={int(s.start_s)}",
                    )
                    for s in q.spans
                ],
            )
            for q in qset.questions
        ],
    )
