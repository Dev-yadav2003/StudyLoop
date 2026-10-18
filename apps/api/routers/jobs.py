from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException

from apps.api.deps import Container
from apps.api.schemas import ErrorOut, JobOut, StageProgress, StageState
from lecnotes.bootstrap import ApiContainer
from lecnotes.domain import Job, JobStatus, StageRunStatus
from lecnotes.domain.video import is_playlist_link, parse_video_id
from lecnotes.errors import InvalidLink, PlaylistNotSupported
from lecnotes.storage.repositories.stage_runs import latest_per_stage, runs_for_job

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])
ERRORS: dict[int | str, dict[str, Any]] = {404: {"model": ErrorOut}, 422: {"model": ErrorOut}}


async def job_out(c: ApiContainer, job: Job) -> JobOut:
    latest = latest_per_stage(await runs_for_job(c.engine, job.id))
    stages = []
    for name in c.services.pipeline.stage_names:
        run = latest.get(name)
        state: StageState
        if run is None:
            state = "pending"
        elif run.status is StageRunStatus.FAILED:
            state = "failed"
        else:
            state = "cached" if run.cache_hit else "done"
        stages.append(
            StageProgress(
                stage=name,
                state=state,
                latency_ms=run.latency_ms if run else None,
                error_type=run.error_type if run else None,
            )
        )
    notes = await c.notes.get_by_job(job.id) if job.status is JobStatus.SUCCEEDED else None
    return JobOut(
        id=job.id,
        url=job.url,
        video_id=job.video_id,
        subject_preset=job.options.subject_preset,
        status=job.status,
        error=job.error,
        error_type=job.error_type,
        attempts=job.attempts,
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        stages=stages,
        notes_id=notes.id if notes else None,
    )


def video_id_from_link(url: str) -> str:
    """The video id of a single-video YouTube link; a playlist or other link is a 422."""
    video_id = parse_video_id(url)
    if video_id is None:
        if is_playlist_link(url):
            raise PlaylistNotSupported(f"playlist link: {url!r}")
        raise InvalidLink(f"not a video link: {url!r}")
    return video_id


@router.get("/{job_id}", response_model=JobOut, responses=ERRORS)
async def get_job(job_id: UUID, c: Container) -> JobOut:
    job = await c.queue.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    return await job_out(c, job)
