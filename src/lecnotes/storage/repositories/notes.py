"""Finished notes: one `notes` row per job + one `units` row per knowledge unit.

Units are editable (step 9.2): an edit replaces one unit and the rendered Markdown together,
guarded by `notes.revision` so two concurrent edits can't silently drop one another. A rerun
replaces the job's notes — edits included (units go with them via ON DELETE CASCADE)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import ColumnElement, delete, func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncEngine

from lecnotes.domain import (
    KnowledgeUnit,
    NotesDocument,
    PerceptionResult,
    VerifySummary,
    VideoMeta,
)
from lecnotes.storage.tables import NotesRow, UnitRow


@dataclass(frozen=True)
class StoredNotes:
    id: uuid.UUID
    job_id: uuid.UUID
    document: NotesDocument
    revision: int = 0
    edited_units: frozenset[str] = frozenset()


class NotesRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def save(self, job_id: uuid.UUID, doc: NotesDocument) -> uuid.UUID:
        notes_id = uuid.uuid4()
        async with self._engine.begin() as conn:
            await conn.execute(delete(NotesRow).where(NotesRow.job_id == job_id))
            await conn.execute(
                insert(NotesRow).values(
                    id=notes_id,
                    job_id=job_id,
                    title=doc.title,
                    markdown=doc.markdown,
                    video=doc.video.model_dump(mode="json"),
                    evidence=doc.evidence.model_dump(mode="json") if doc.evidence else None,
                    verify=doc.verify.model_dump(mode="json") if doc.verify else None,
                )
            )
            if doc.units:
                await conn.execute(
                    insert(UnitRow),
                    [
                        {
                            "id": uuid.uuid4(),
                            "notes_id": notes_id,
                            "unit_key": unit.id,
                            "position": position,
                            "data": unit.model_dump(mode="json"),
                        }
                        for position, unit in enumerate(doc.units)
                    ],
                )
        return notes_id

    async def save_edit(
        self, stored: StoredNotes, doc: NotesDocument, unit_id: str
    ) -> StoredNotes | None:
        """Persist `doc` (with `unit_id` edited) if the notes are still at `stored.revision`.
        Returns the new state, or None if someone else edited them first."""
        unit = next(u for u in doc.units if u.id == unit_id)
        async with self._engine.begin() as conn:
            bumped = await conn.execute(
                update(NotesRow)
                .where(NotesRow.id == stored.id, NotesRow.revision == stored.revision)
                .values(markdown=doc.markdown, revision=NotesRow.revision + 1)
            )
            if bumped.rowcount == 0:
                return None
            await conn.execute(
                update(UnitRow)
                .where(UnitRow.notes_id == stored.id, UnitRow.unit_key == unit_id)
                .values(data=unit.model_dump(mode="json"), edited_at=func.now())
            )
        return StoredNotes(
            id=stored.id,
            job_id=stored.job_id,
            document=doc,
            revision=stored.revision + 1,
            edited_units=stored.edited_units | {unit_id},
        )

    async def get(self, notes_id: uuid.UUID) -> StoredNotes | None:
        return await self._load(NotesRow.id == notes_id)

    async def get_by_job(self, job_id: uuid.UUID) -> StoredNotes | None:
        return await self._load(NotesRow.job_id == job_id)

    async def _load(self, where: ColumnElement[bool]) -> StoredNotes | None:
        async with self._engine.connect() as conn:
            row = (await conn.execute(select(NotesRow).where(where))).one_or_none()
            if row is None:
                return None
            units = (
                await conn.execute(
                    select(UnitRow.data, UnitRow.edited_at)
                    .where(UnitRow.notes_id == row.id)
                    .order_by(UnitRow.position)
                )
            ).all()
            document = NotesDocument(
                title=row.title,
                video=VideoMeta.model_validate(row.video),
                units=[KnowledgeUnit.model_validate(u.data) for u in units],
                markdown=row.markdown,
                evidence=PerceptionResult.model_validate(row.evidence) if row.evidence else None,
                verify=VerifySummary.model_validate(row.verify) if row.verify else None,
            )
        edited = frozenset(u.data["id"] for u in units if u.edited_at is not None)
        return StoredNotes(
            id=row.id,
            job_id=row.job_id,
            document=document,
            revision=row.revision,
            edited_units=edited,
        )
