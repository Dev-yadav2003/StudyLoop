"""The user's library: chapters (e.g. "Python") holding lessons (one lecture each).

A lesson points at the job that makes its notes. Jobs are shared — the same video + options
is one job — so renaming, moving or deleting a lesson never touches notes or the cache.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from lecnotes.domain import JobStatus
from lecnotes.storage.tables import ChapterRow, JobRow, LessonRow, NotesRow


@dataclass(frozen=True)
class LessonView:
    id: uuid.UUID
    chapter_id: uuid.UUID
    job_id: uuid.UUID
    custom_title: str | None
    url: str
    video_id: str | None
    status: JobStatus
    notes_id: uuid.UUID | None
    video_title: str | None
    created_at: datetime

    @property
    def title(self) -> str:
        """What the user sees: their name for it, else the video's title, else the link."""
        return self.custom_title or self.video_title or self.url


@dataclass(frozen=True)
class ChapterView:
    id: uuid.UUID
    name: str
    created_at: datetime
    lessons: list[LessonView]


class LibraryRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    # --- chapters -------------------------------------------------------------------------

    async def chapters(self) -> list[ChapterView]:
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    select(ChapterRow).order_by(ChapterRow.position, ChapterRow.created_at)
                )
            ).all()
            lessons = await _lessons(conn)
        by_chapter: dict[uuid.UUID, list[LessonView]] = {}
        for lesson in lessons:
            by_chapter.setdefault(lesson.chapter_id, []).append(lesson)
        return [
            ChapterView(
                id=r.id, name=r.name, created_at=r.created_at, lessons=by_chapter.get(r.id, [])
            )
            for r in rows
        ]

    async def chapter(self, chapter_id: uuid.UUID) -> ChapterView | None:
        return next((c for c in await self.chapters() if c.id == chapter_id), None)

    async def create_chapter(self, name: str) -> ChapterView:
        async with self._engine.begin() as conn:
            position = await _next_position(conn, select(func.max(ChapterRow.position)))
            row = (
                await conn.execute(
                    insert(ChapterRow)
                    .values(id=uuid.uuid4(), name=name, position=position)
                    .returning(ChapterRow)
                )
            ).one()
        return ChapterView(id=row.id, name=row.name, created_at=row.created_at, lessons=[])

    async def rename_chapter(self, chapter_id: uuid.UUID, name: str) -> bool:
        async with self._engine.begin() as conn:
            done = await conn.execute(
                update(ChapterRow).where(ChapterRow.id == chapter_id).values(name=name)
            )
        return done.rowcount > 0

    async def delete_chapter(self, chapter_id: uuid.UUID) -> bool:
        """Deletes the chapter and its lessons (not their notes)."""
        async with self._engine.begin() as conn:
            done = await conn.execute(delete(ChapterRow).where(ChapterRow.id == chapter_id))
        return done.rowcount > 0

    # --- lessons --------------------------------------------------------------------------

    async def add_lesson(
        self, chapter_id: uuid.UUID, job_id: uuid.UUID, title: str | None = None
    ) -> LessonView:
        lesson_id = uuid.uuid4()
        async with self._engine.begin() as conn:
            position = await _next_position(
                conn,
                select(func.max(LessonRow.position)).where(LessonRow.chapter_id == chapter_id),
            )
            await conn.execute(
                insert(LessonRow).values(
                    id=lesson_id,
                    chapter_id=chapter_id,
                    job_id=job_id,
                    title=title,
                    position=position,
                )
            )
        lesson = await self.lesson(lesson_id)
        assert lesson is not None
        return lesson

    async def lesson(self, lesson_id: uuid.UUID) -> LessonView | None:
        async with self._engine.connect() as conn:
            found = await _lessons(conn, LessonRow.id == lesson_id)
        return found[0] if found else None

    async def update_lesson(
        self,
        lesson_id: uuid.UUID,
        *,
        title: str | None = None,
        clear_title: bool = False,
        chapter_id: uuid.UUID | None = None,
        job_id: uuid.UUID | None = None,
    ) -> LessonView | None:
        values: dict[str, object] = {}
        if title is not None or clear_title:
            values["title"] = title
        if chapter_id is not None:
            values["chapter_id"] = chapter_id
        if job_id is not None:
            values["job_id"] = job_id
        if values:
            async with self._engine.begin() as conn:
                await conn.execute(
                    update(LessonRow).where(LessonRow.id == lesson_id).values(**values)
                )
        return await self.lesson(lesson_id)

    async def delete_lesson(self, lesson_id: uuid.UUID) -> bool:
        async with self._engine.begin() as conn:
            done = await conn.execute(delete(LessonRow).where(LessonRow.id == lesson_id))
        return done.rowcount > 0


async def _next_position(conn: AsyncConnection, max_query: object) -> int:
    current = (await conn.execute(max_query)).scalar()  # type: ignore[call-overload]
    return 0 if current is None else int(current) + 1


async def _lessons(conn: AsyncConnection, *where: object) -> list[LessonView]:
    query = (
        select(
            LessonRow.id,
            LessonRow.chapter_id,
            LessonRow.job_id,
            LessonRow.title,
            LessonRow.created_at,
            JobRow.url,
            JobRow.video_id,
            JobRow.status,
            NotesRow.id.label("notes_id"),
            NotesRow.title.label("video_title"),
        )
        .join(JobRow, JobRow.id == LessonRow.job_id)
        .outerjoin(NotesRow, NotesRow.job_id == LessonRow.job_id)
        .where(*where)  # type: ignore[arg-type]
        .order_by(LessonRow.position, LessonRow.created_at)
    )
    return [
        LessonView(
            id=r.id,
            chapter_id=r.chapter_id,
            job_id=r.job_id,
            custom_title=r.title,
            url=r.url,
            video_id=r.video_id,
            status=JobStatus(r.status),
            notes_id=r.notes_id,
            video_title=r.video_title,
            created_at=r.created_at,
        )
        for r in (await conn.execute(query)).all()
    ]
