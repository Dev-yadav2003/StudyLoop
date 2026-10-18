import asyncio

from sqlalchemy import text, update
from sqlalchemy.ext.asyncio import AsyncEngine

from lecnotes.domain import JobOptions, JobStatus, SubjectPreset
from lecnotes.jobs.queue import JobQueue
from lecnotes.storage.tables import JobRow

VID, OTHER = "dQw4w9WgXcQ", "aaaaaaaaaaa"


async def test_enqueue_is_idempotent_per_video_and_options(engine: AsyncEngine) -> None:
    q = JobQueue(engine)
    job, created = await q.enqueue(f"https://youtu.be/{VID}", VID, JobOptions())
    same, created_again = await q.enqueue(
        f"https://www.youtube.com/watch?v={VID}", VID, JobOptions()
    )
    other_options, created_other = await q.enqueue(
        VID, VID, JobOptions(subject_preset=SubjectPreset.STEM)
    )

    assert (created, created_again, created_other) == (True, False, True)
    assert same.id == job.id  # a different URL form of the same video → same job
    assert other_options.id != job.id
    assert job.status is JobStatus.QUEUED
    assert job.video_id == VID


async def test_concurrent_duplicate_submissions_create_one_job(engine: AsyncEngine) -> None:
    q = JobQueue(engine)
    results = await asyncio.gather(*(q.enqueue(VID, VID, JobOptions()) for _ in range(5)))
    assert len({job.id for job, _ in results}) == 1
    assert sum(created for _, created in results) == 1


async def test_failed_job_can_be_resubmitted(engine: AsyncEngine) -> None:
    q = JobQueue(engine)
    job, _ = await q.enqueue(VID, VID, JobOptions())
    lease = await q.lease("w1")
    assert lease is not None
    await q.fail(lease, error_type="X", user_message="broke", retryable=False)

    again, created = await q.enqueue(VID, VID, JobOptions())
    assert created
    assert again.id != job.id


async def test_two_workers_never_lease_the_same_job(engine: AsyncEngine) -> None:
    q = JobQueue(engine)
    await q.enqueue(VID, VID, JobOptions())
    await q.enqueue(OTHER, OTHER, JobOptions())

    leases = await asyncio.gather(*(q.lease(f"w{i}") for i in range(4)))
    taken = [lease for lease in leases if lease is not None]

    assert len(taken) == 2
    assert len({lease.job.id for lease in taken}) == 2
    assert all(lease.job.status is JobStatus.RUNNING for lease in taken)
    assert all(lease.job.attempts == 1 for lease in taken)


async def test_oldest_due_job_first(engine: AsyncEngine) -> None:
    q = JobQueue(engine)
    first, _ = await q.enqueue(VID, VID, JobOptions())
    await q.enqueue(OTHER, OTHER, JobOptions())
    lease = await q.lease("w1")
    assert lease is not None
    assert lease.job.id == first.id


async def test_expired_lease_is_taken_over_and_old_owner_loses_it(engine: AsyncEngine) -> None:
    q = JobQueue(engine, lease_s=0.2)
    await q.enqueue(VID, VID, JobOptions())
    dead = await q.lease("dead-worker")
    assert dead is not None
    assert await q.lease("w2") is None  # still leased

    await asyncio.sleep(0.4)  # the worker died; its lease expires
    takeover = await q.lease("w2")

    assert takeover is not None
    assert takeover.job.id == dead.job.id
    assert takeover.job.attempts == 2
    assert await q.heartbeat(dead) is False
    assert await q.heartbeat(takeover) is True


async def test_complete(engine: AsyncEngine) -> None:
    q = JobQueue(engine)
    await q.enqueue(VID, VID, JobOptions())
    lease = await q.lease("w1")
    assert lease is not None
    await q.complete(lease)
    job = await q.get(lease.job.id)
    assert job is not None
    assert job.status is JobStatus.SUCCEEDED
    assert job.finished_at is not None
    assert await q.lease("w2") is None


async def test_retryable_failure_requeues_with_backoff(engine: AsyncEngine) -> None:
    q = JobQueue(engine)
    await q.enqueue(VID, VID, JobOptions())
    lease = await q.lease("w1")
    assert lease is not None

    job = await q.fail(
        lease, error_type="QuotaExceeded", user_message="quota", retryable=True, retry_in_s=3600
    )

    assert job.status is JobStatus.QUEUED
    assert (job.error, job.error_type) == (None, "QuotaExceeded")
    assert await q.lease("w2") is None  # not due for an hour


async def test_retries_stop_at_max_attempts(engine: AsyncEngine) -> None:
    q = JobQueue(engine, max_attempts=2)
    queued, _ = await q.enqueue(VID, VID, JobOptions())
    statuses = []
    for _ in range(2):
        lease = await q.lease("w1")
        assert lease is not None
        job = await q.fail(lease, error_type="T", user_message="busy", retryable=True, retry_in_s=0)
        statuses.append(job.status)
    assert statuses == [JobStatus.QUEUED, JobStatus.FAILED]
    final = await q.get(queued.id)
    assert final is not None
    assert final.error == "busy"


async def test_non_retryable_failure(engine: AsyncEngine) -> None:
    q = JobQueue(engine)
    await q.enqueue(VID, VID, JobOptions())
    lease = await q.lease("w1")
    assert lease is not None
    job = await q.fail(
        lease, error_type="VideoUnavailable", user_message="private", retryable=False
    )
    assert (job.status, job.error, job.error_type) == (
        JobStatus.FAILED,
        "private",
        "VideoUnavailable",
    )


async def test_rerun_requeues_finished_job_with_stage(engine: AsyncEngine) -> None:
    q = JobQueue(engine)
    await q.enqueue(VID, VID, JobOptions())
    lease = await q.lease("w1")
    assert lease is not None
    await q.complete(lease)

    job = await q.request_rerun(lease.job.id, "extract")
    assert job is not None
    assert job.status is JobStatus.QUEUED
    again = await q.lease("w2")
    assert again is not None
    assert again.rerun_from == "extract"
    assert again.job.attempts == 1  # attempts reset by the rerun


async def test_rerun_leaves_running_jobs_alone(engine: AsyncEngine) -> None:
    q = JobQueue(engine)
    await q.enqueue(VID, VID, JobOptions())
    lease = await q.lease("w1")
    assert lease is not None
    assert await q.request_rerun(lease.job.id, None) is None


async def test_unknown_status_row_is_not_leased(engine: AsyncEngine) -> None:
    """Defensive: only queued/expired-running rows are leasable."""
    q = JobQueue(engine)
    job, _ = await q.enqueue(VID, VID, JobOptions())
    async with engine.begin() as conn:
        await conn.execute(update(JobRow).where(JobRow.id == job.id).values(status="succeeded"))
        assert (await conn.execute(text("select count(*) from jobs"))).scalar_one() == 1
    assert await q.lease("w1") is None


async def test_jobs_stored_with_since_removed_options_still_load(engine: AsyncEngine) -> None:
    """Rows written before `generate_questions` was removed made every read of them a 500."""
    queue = JobQueue(engine)
    job, _ = await queue.enqueue(f"https://youtu.be/{VID}", VID, JobOptions())
    async with engine.begin() as conn:
        await conn.execute(
            update(JobRow)
            .where(JobRow.id == job.id)
            .values(options={"subject_preset": "stem", "generate_questions": False})
        )
    loaded = await queue.get(job.id)
    assert loaded is not None
    assert loaded.options == JobOptions(subject_preset=SubjectPreset.STEM)
