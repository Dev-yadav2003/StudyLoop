from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from apps.api.deps import Admin, Container
from apps.api.routers.jobs import job_out
from apps.api.schemas import (
    ErrorOut,
    FeedbackItemOut,
    FeedbackListOut,
    RerunOut,
    StageRunOut,
)
from lecnotes.pipeline.registry import VIDEO_STAGE_ORDER
from lecnotes.storage.repositories.stage_runs import runs_for_job

router = APIRouter(prefix="/v1/admin", tags=["admin"], dependencies=[Admin])
ERRORS: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorOut},
    403: {"model": ErrorOut},
    404: {"model": ErrorOut},
}


@router.get("/jobs/{job_id}/stages", response_model=list[StageRunOut], responses=ERRORS)
async def job_stages(job_id: UUID, c: Container) -> list[StageRunOut]:
    """Every stage run of the job (incl. earlier attempts and reruns), oldest first."""
    if await c.queue.get(job_id) is None:
        raise HTTPException(404, "Job not found.")
    return [
        StageRunOut(
            **run.model_dump(exclude={"job_id", "video_id", "cache_key", "cost_usd"}),
            cost_usd=str(run.cost_usd),
        )
        for run in await runs_for_job(c.engine, job_id)
    ]


@router.post(
    "/jobs/{job_id}/rerun",
    response_model=RerunOut,
    status_code=202,
    responses=ERRORS | {409: {"model": ErrorOut}, 422: {"model": ErrorOut}},
)
async def rerun(
    job_id: UUID,
    c: Container,
    from_stage: str | None = Query(None, alias="from", description="first stage to recompute"),
) -> RerunOut:
    """Queue the job again. Stages before `from` come from the cache; `from` and everything
    after it are recomputed. Without `from`, cached stages are reused as they are."""
    if from_stage is not None and from_stage not in VIDEO_STAGE_ORDER:
        raise HTTPException(422, f"Unknown stage {from_stage!r}; one of {list(VIDEO_STAGE_ORDER)}.")
    if await c.queue.get(job_id) is None:
        raise HTTPException(404, "Job not found.")
    job = await c.queue.request_rerun(job_id, from_stage)
    if job is None:
        raise HTTPException(409, "The job is running; wait for it to finish.")
    return RerunOut(job=await job_out(c, job), rerun_from=from_stage)


@router.get("/feedback", response_model=FeedbackListOut, responses=ERRORS)
async def feedback(
    c: Container,
    limit: int = Query(50, ge=1, le=500),
    rating: Literal["up", "down"] | None = Query(None, description="only 👍 or only 👎"),
) -> FeedbackListOut:
    """Recent reader feedback, newest first, with all-time totals."""
    totals = await c.feedback.summary()
    wanted = None if rating is None else 1 if rating == "up" else -1
    items = await c.feedback.recent(limit=limit, rating=wanted)
    return FeedbackListOut(
        up=totals.up,
        down=totals.down,
        items=[
            FeedbackItemOut(
                id=f.id,
                notes_id=f.notes_id,
                video_id=f.video_id,
                title=f.title,
                rating=f.feedback.rating,
                reason=f.feedback.reason,
                unit_id=f.feedback.unit_id,
                trace_id=f.trace_id,
                created_at=f.created_at,
            )
            for f in items
        ],
    )
