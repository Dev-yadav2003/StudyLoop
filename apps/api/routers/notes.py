from __future__ import annotations

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

from apps.api.deps import Container
from apps.api.schemas import ErrorOut, NotesOut
from lecnotes.storage.repositories.notes import StoredNotes

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
