"""Chapters and lessons: the user's library. Adding a lesson queues its notes (same
idempotent job queue as POST /v1/jobs)."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Response

from apps.api.deps import Container
from apps.api.routers.jobs import video_id_from_link
from apps.api.schemas import (
    ChapterCreate,
    ChapterOut,
    ErrorOut,
    LessonCreate,
    LessonOut,
    LessonUpdate,
)
from lecnotes.domain import JobOptions, JobStatus
from lecnotes.storage.repositories.library import ChapterView, LessonView

router = APIRouter(prefix="/v1", tags=["library"])
ERRORS: dict[int | str, dict[str, Any]] = {404: {"model": ErrorOut}, 422: {"model": ErrorOut}}


def lesson_out(lesson: LessonView) -> LessonOut:
    return LessonOut(
        id=lesson.id,
        chapter_id=lesson.chapter_id,
        title=lesson.title,
        custom_title=lesson.custom_title,
        url=lesson.url,
        video_id=lesson.video_id,
        job_id=lesson.job_id,
        status=lesson.status,
        notes_id=lesson.notes_id,
        created_at=lesson.created_at,
    )


def chapter_out(chapter: ChapterView) -> ChapterOut:
    return ChapterOut(
        id=chapter.id,
        name=chapter.name,
        created_at=chapter.created_at,
        lessons=[lesson_out(lesson) for lesson in chapter.lessons],
    )


async def _lesson(c: Container, lesson_id: UUID) -> LessonView:
    lesson = await c.library.lesson(lesson_id)
    if lesson is None:
        raise HTTPException(404, "Lesson not found.")
    return lesson


@router.get("/chapters", response_model=list[ChapterOut])
async def list_chapters(c: Container) -> list[ChapterOut]:
    """Every chapter with its lessons, in order."""
    return [chapter_out(ch) for ch in await c.library.chapters()]


@router.post("/chapters", response_model=ChapterOut, status_code=201, responses=ERRORS)
async def create_chapter(body: ChapterCreate, c: Container) -> ChapterOut:
    return chapter_out(await c.library.create_chapter(body.name))


@router.patch("/chapters/{chapter_id}", response_model=ChapterOut, responses=ERRORS)
async def rename_chapter(chapter_id: UUID, body: ChapterCreate, c: Container) -> ChapterOut:
    if not await c.library.rename_chapter(chapter_id, body.name):
        raise HTTPException(404, "Chapter not found.")
    chapter = await c.library.chapter(chapter_id)
    assert chapter is not None
    return chapter_out(chapter)


@router.delete("/chapters/{chapter_id}", status_code=204, responses=ERRORS)
async def delete_chapter(chapter_id: UUID, c: Container) -> Response:
    """Deletes the chapter and its lessons. Notes stay (other lessons may share them)."""
    if not await c.library.delete_chapter(chapter_id):
        raise HTTPException(404, "Chapter not found.")
    return Response(status_code=204)


@router.post(
    "/chapters/{chapter_id}/lessons", response_model=LessonOut, status_code=201, responses=ERRORS
)
async def add_lesson(chapter_id: UUID, body: LessonCreate, c: Container) -> LessonOut:
    """Add a lecture to a chapter and queue its notes. A video already processed with the
    same options reuses its notes at once."""
    if await c.library.chapter(chapter_id) is None:
        raise HTTPException(404, "Chapter not found.")
    video_id = video_id_from_link(body.url)
    job, _ = await c.queue.enqueue(
        body.url, video_id, JobOptions(subject_preset=body.subject_preset)
    )
    return lesson_out(await c.library.add_lesson(chapter_id, job.id, body.title))


@router.get("/lessons/{lesson_id}", response_model=LessonOut, responses=ERRORS)
async def get_lesson(lesson_id: UUID, c: Container) -> LessonOut:
    return lesson_out(await _lesson(c, lesson_id))


@router.patch("/lessons/{lesson_id}", response_model=LessonOut, responses=ERRORS)
async def update_lesson(lesson_id: UUID, body: LessonUpdate, c: Container) -> LessonOut:
    """Rename (`title`; null resets to the video title) and/or move (`chapter_id`)."""
    await _lesson(c, lesson_id)
    if body.chapter_id is not None and await c.library.chapter(body.chapter_id) is None:
        raise HTTPException(404, "Chapter not found.")
    updated = await c.library.update_lesson(
        lesson_id,
        title=body.title,
        clear_title="title" in body.model_fields_set and body.title is None,
        chapter_id=body.chapter_id,
    )
    assert updated is not None
    return lesson_out(updated)


@router.delete("/lessons/{lesson_id}", status_code=204, responses=ERRORS)
async def delete_lesson(lesson_id: UUID, c: Container) -> Response:
    if not await c.library.delete_lesson(lesson_id):
        raise HTTPException(404, "Lesson not found.")
    return Response(status_code=204)


@router.post(
    "/lessons/{lesson_id}/retry",
    response_model=LessonOut,
    responses=ERRORS | {409: {"model": ErrorOut}},
)
async def retry_lesson(lesson_id: UUID, c: Container) -> LessonOut:
    """Queue a failed lesson's video again (a failed job is never reused)."""
    lesson = await _lesson(c, lesson_id)
    if lesson.status is not JobStatus.FAILED or lesson.video_id is None:
        raise HTTPException(409, "Only a failed lesson can be retried.")
    job = await c.queue.get(lesson.job_id)
    assert job is not None
    new_job, _ = await c.queue.enqueue(job.url, lesson.video_id, job.options)
    updated = await c.library.update_lesson(lesson_id, job_id=new_job.id)
    assert updated is not None
    return lesson_out(updated)
