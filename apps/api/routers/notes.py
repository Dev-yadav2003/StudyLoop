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
from lecnotes.storage.repositories.notes import StoredNotes
from lecnotes.timefmt import format_timestamp

router = APIRouter(prefix="/v1/notes", tags=["notes"])


async def _load(c: Container, notes_id: UUID) -> StoredNotes:
    stored = await c.notes.get(notes_id)
    if stored is None:
        raise HTTPException(404, "Notes not found.")
    return stored


@router.get("/{notes_id}", response_model=NotesOut, responses={404: {"model": ErrorOut}})
async def get_notes(notes_id: UUID, c: Container) -> NotesOut:
    stored = await _load(c, notes_id)
    doc = stored.document
    return NotesOut(
        id=stored.id,
        job_id=stored.job_id,
        title=doc.title,
        video=doc.video,
        markdown=doc.markdown,
        units=doc.units,
    )


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
    qset = await c.qa.generate(
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
