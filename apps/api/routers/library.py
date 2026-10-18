"""Chapters and lessons: each account's own library. Adding a lesson checks the account's
allowance against the video's real length, then queues its notes (the same idempotent job
queue for everyone, so a video already processed for anyone is reused at once)."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Response

from apps.api.deps import Container, CurrentUser
from apps.api.quota import charge_video
from apps.api.routers.jobs import video_id_from_link
from apps.api.schemas import (
    ChapterCreate,
    ChapterOut,
    ErrorOut,
    LessonCreate,
    LessonOut,
    LessonUpdate,
    MeOut,
)
from lecnotes.domain import JobOptions, JobStatus
from lecnotes.storage.repositories.library import ChapterView, LessonView

router = APIRouter(prefix="/v1", tags=["library"])
ERRORS: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorOut},
    404: {"model": ErrorOut},
    422: {"model": ErrorOut},
}


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


async def _lesson(c: Container, user: CurrentUser, lesson_id: UUID) -> LessonView:
    lesson = await c.library(user.id).lesson(lesson_id)
    if lesson is None:
        raise HTTPException(404, "Lesson not found.")
    return lesson


@router.get("/me", response_model=MeOut, responses=ERRORS)
async def me(c: Container, user: CurrentUser) -> MeOut:
    """The signed-in account and what it has used of its allowance."""
    used = await c.usage.totals(user.id)
    s = c.settings
    return MeOut(
        email=user.email,
        is_owner=user.is_owner,
        video_minutes_used=round(used.video_minutes, 1),
        video_minutes_limit=None if user.is_owner else s.guest_video_minutes,
        tokens_used=used.tokens,
        tokens_limit=None if user.is_owner else s.guest_token_budget,
    )


@router.get("/chapters", response_model=list[ChapterOut], responses=ERRORS)
async def list_chapters(c: Container, user: CurrentUser) -> list[ChapterOut]:
    """Every chapter of this account with its lessons, in order."""
    library = c.library(user.id)
    if user.is_owner:
        await library.claim_unowned()  # chapters made before sign-in existed
    return [chapter_out(ch) for ch in await library.chapters()]


@router.post("/chapters", response_model=ChapterOut, status_code=201, responses=ERRORS)
async def create_chapter(body: ChapterCreate, c: Container, user: CurrentUser) -> ChapterOut:
    return chapter_out(await c.library(user.id).create_chapter(body.name))


@router.patch("/chapters/{chapter_id}", response_model=ChapterOut, responses=ERRORS)
async def rename_chapter(
    chapter_id: UUID, body: ChapterCreate, c: Container, user: CurrentUser
) -> ChapterOut:
    library = c.library(user.id)
    if not await library.rename_chapter(chapter_id, body.name):
        raise HTTPException(404, "Chapter not found.")
    chapter = await library.chapter(chapter_id)
    assert chapter is not None
    return chapter_out(chapter)


@router.delete("/chapters/{chapter_id}", status_code=204, responses=ERRORS)
async def delete_chapter(chapter_id: UUID, c: Container, user: CurrentUser) -> Response:
    """Deletes the chapter and its lessons. Notes stay (other lessons may share them)."""
    if not await c.library(user.id).delete_chapter(chapter_id):
        raise HTTPException(404, "Chapter not found.")
    return Response(status_code=204)


@router.post(
    "/chapters/{chapter_id}/lessons",
    response_model=LessonOut,
    status_code=201,
    responses=ERRORS | {429: {"model": ErrorOut}},
)
async def add_lesson(
    chapter_id: UUID, body: LessonCreate, c: Container, user: CurrentUser
) -> LessonOut:
    """Add a lecture to a chapter and queue its notes. Guests are charged the video's length
    against their allowance first (429 if it doesn't fit)."""
    library = c.library(user.id)
    if await library.chapter(chapter_id) is None:
        raise HTTPException(404, "Chapter not found.")
    video_id_from_link(body.url)  # 422 for playlists/other links before any API call
    video = await c.services.catalog.resolve(body.url)
    await charge_video(c, user, video)
    job, _ = await c.queue.enqueue(
        body.url, video.video_id, JobOptions(subject_preset=body.subject_preset)
    )
    return lesson_out(await library.add_lesson(chapter_id, job.id, body.title))


@router.get("/lessons/{lesson_id}", response_model=LessonOut, responses=ERRORS)
async def get_lesson(lesson_id: UUID, c: Container, user: CurrentUser) -> LessonOut:
    return lesson_out(await _lesson(c, user, lesson_id))


@router.patch("/lessons/{lesson_id}", response_model=LessonOut, responses=ERRORS)
async def update_lesson(
    lesson_id: UUID, body: LessonUpdate, c: Container, user: CurrentUser
) -> LessonOut:
    """Rename (`title`; null resets to the video title) and/or move (`chapter_id`)."""
    await _lesson(c, user, lesson_id)
    library = c.library(user.id)
    if body.chapter_id is not None and await library.chapter(body.chapter_id) is None:
        raise HTTPException(404, "Chapter not found.")
    updated = await library.update_lesson(
        lesson_id,
        title=body.title,
        clear_title="title" in body.model_fields_set and body.title is None,
        chapter_id=body.chapter_id,
    )
    assert updated is not None
    return lesson_out(updated)


@router.delete("/lessons/{lesson_id}", status_code=204, responses=ERRORS)
async def delete_lesson(lesson_id: UUID, c: Container, user: CurrentUser) -> Response:
    if not await c.library(user.id).delete_lesson(lesson_id):
        raise HTTPException(404, "Lesson not found.")
    return Response(status_code=204)


@router.post(
    "/lessons/{lesson_id}/retry",
    response_model=LessonOut,
    responses=ERRORS | {409: {"model": ErrorOut}},
)
async def retry_lesson(lesson_id: UUID, c: Container, user: CurrentUser) -> LessonOut:
    """Queue a failed lesson's video again (a failed job is never reused; not charged again)."""
    lesson = await _lesson(c, user, lesson_id)
    if lesson.status is not JobStatus.FAILED or lesson.video_id is None:
        raise HTTPException(409, "Only a failed lesson can be retried.")
    job = await c.queue.get(lesson.job_id)
    assert job is not None
    new_job, _ = await c.queue.enqueue(job.url, lesson.video_id, job.options)
    updated = await c.library(user.id).update_lesson(lesson_id, job_id=new_job.id)
    assert updated is not None
    return lesson_out(updated)
