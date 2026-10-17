"""The HTTP API end to end on real Postgres: submit → run → poll → notes, with fakes only at
the edges (static catalog, recorded perception, scripted LLM)."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from uuid import uuid4

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncEngine

from apps.api.app import create_app
from lecnotes.adapters.cache.postgres_cache import PostgresStageCache
from lecnotes.bootstrap import ApiContainer
from lecnotes.jobs.queue import JobQueue
from lecnotes.jobs.runner import JobRunner
from lecnotes.storage.repositories.notes import NotesRepository
from lecnotes.storage.repositories.stage_runs import PostgresStageRunSink
from tests.pipeline_fakes import URL, VID, fake_services, make_settings

ADMIN = {"Authorization": "Bearer s3cret"}


@dataclass
class Api:
    client: httpx.AsyncClient
    runner: JobRunner


@pytest.fixture
async def api(engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Api]:
    settings = make_settings(monkeypatch, admin_token=SecretStr("s3cret"), run_worker=False)
    services, _ = fake_services(
        settings, PostgresStageCache(engine), sinks=[PostgresStageRunSink(engine)]
    )
    queue, notes = JobQueue(engine), NotesRepository(engine)
    runner = JobRunner(queue, services.pipeline, notes, owner="test")
    app = create_app(ApiContainer(settings, engine, queue, notes, services.pipeline, runner))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield Api(client, runner)


async def test_submit_poll_and_read_notes(api: Api) -> None:
    created = await api.client.post("/v1/jobs", json={"url": URL, "subject_preset": "stem"})
    assert created.status_code == 202
    job = created.json()
    assert created.headers["Location"] == f"/v1/jobs/{job['id']}"
    assert (job["status"], job["video_id"], job["subject_preset"]) == ("queued", VID, "stem")
    assert {s["state"] for s in job["stages"]} == {"pending"}
    assert [s["stage"] for s in job["stages"]][:2] == ["resolve", "perceive"]

    assert await api.runner.run_once()

    polled = (await api.client.get(f"/v1/jobs/{job['id']}")).json()
    assert polled["status"] == "succeeded"
    assert polled["finished_at"] is not None
    states = {s["stage"]: s["state"] for s in polled["stages"]}
    assert states["perceive"] == "done"
    assert all(state in {"done", "cached"} for state in states.values())

    notes = (await api.client.get(f"/v1/notes/{polled['notes_id']}")).json()
    assert notes["title"] == "Python Iteration Tools"
    assert notes["video"]["video_id"] == VID
    assert [u["concept"] for u in notes["units"]] == ["enumerate()"]
    assert "### enumerate()" in notes["markdown"]

    export = await api.client.get(f"/v1/notes/{polled['notes_id']}/export?format=md")
    assert export.status_code == 200
    assert export.headers["content-type"].startswith("text/markdown")
    assert f'filename="{VID}.md"' in export.headers["content-disposition"]
    assert export.text == notes["markdown"]


async def test_same_video_returns_the_existing_job(api: Api) -> None:
    first = await api.client.post("/v1/jobs", json={"url": URL})
    again = await api.client.post("/v1/jobs", json={"url": f"https://youtu.be/{VID}?t=10"})
    assert (first.status_code, again.status_code) == (202, 200)
    assert again.json()["id"] == first.json()["id"]


@pytest.mark.parametrize(
    ("url", "code", "phrase"),
    [
        ("https://vimeo.com/123", "InvalidLink", "link to a YouTube video"),
        ("https://www.youtube.com/playlist?list=PL123456789", "PlaylistNotSupported", "Playlists"),
    ],
)
async def test_bad_links_are_rejected_immediately(
    api: Api, url: str, code: str, phrase: str
) -> None:
    response = await api.client.post("/v1/jobs", json={"url": url})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == code
    assert phrase in error["message"]
    assert error["request_id"] == response.headers["X-Request-ID"]


async def test_failed_job_reports_the_user_message(api: Api) -> None:
    job = (await api.client.post("/v1/jobs", json={"url": "https://youtu.be/aaaaaaaaaaa"})).json()
    await api.runner.run_once()
    failed = (await api.client.get(f"/v1/jobs/{job['id']}")).json()
    assert (failed["status"], failed["error_type"]) == ("failed", "VideoUnavailable")
    assert "public videos" in failed["error"]
    assert failed["notes_id"] is None
    assert {s["stage"]: s["state"] for s in failed["stages"]}["resolve"] == "failed"


async def test_unknown_ids_are_404_with_json_errors(api: Api) -> None:
    for path in (f"/v1/jobs/{uuid4()}", f"/v1/notes/{uuid4()}", f"/v1/notes/{uuid4()}/export"):
        response = await api.client.get(path)
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"


async def test_admin_requires_the_token(api: Api) -> None:
    job = (await api.client.post("/v1/jobs", json={"url": URL})).json()
    path = f"/v1/admin/jobs/{job['id']}/stages"
    assert (await api.client.get(path)).status_code == 401
    wrong = await api.client.get(path, headers={"Authorization": "Bearer nope"})
    assert wrong.status_code == 401
    assert (await api.client.get(path, headers=ADMIN)).status_code == 200


async def test_admin_stage_history_and_rerun(api: Api) -> None:
    job = (await api.client.post("/v1/jobs", json={"url": URL})).json()
    await api.runner.run_once()

    rerun = await api.client.post(f"/v1/admin/jobs/{job['id']}/rerun?from=extract", headers=ADMIN)
    assert rerun.status_code == 202
    assert (rerun.json()["job"]["status"], rerun.json()["rerun_from"]) == ("queued", "extract")
    await api.runner.run_once()

    history = (await api.client.get(f"/v1/admin/jobs/{job['id']}/stages", headers=ADMIN)).json()
    extract_runs = [r for r in history if r["stage"] == "extract"]
    assert len(extract_runs) == 2  # original + rerun, both recomputed
    assert not any(r["cache_hit"] for r in extract_runs)
    perceive_runs = [r for r in history if r["stage"] == "perceive"]
    assert [r["cache_hit"] for r in perceive_runs] == [False, True]

    bad = await api.client.post(f"/v1/admin/jobs/{job['id']}/rerun?from=nope", headers=ADMIN)
    assert bad.status_code == 422


async def test_admin_disabled_without_token(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(monkeypatch, run_worker=False)
    services, _ = fake_services(settings, PostgresStageCache(engine))
    queue, notes = JobQueue(engine), NotesRepository(engine)
    runner = JobRunner(queue, services.pipeline, notes)
    app = create_app(ApiContainer(settings, engine, queue, notes, services.pipeline, runner))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(f"/v1/admin/jobs/{uuid4()}/stages", headers=ADMIN)
    assert response.status_code == 403
    assert "ADMIN_TOKEN" in response.json()["error"]["message"]


async def test_health_ready_request_id_and_openapi(api: Api) -> None:
    assert (await api.client.get("/healthz")).json() == {"status": "ok"}
    ready = await api.client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["checks"] == {"database": "ok"}  # worker disabled in tests

    echoed = await api.client.get("/healthz", headers={"X-Request-ID": "abc-123"})
    assert echoed.headers["X-Request-ID"] == "abc-123"
    unsafe = await api.client.get("/healthz", headers={"X-Request-ID": "bad id\n"})
    assert unsafe.headers["X-Request-ID"] != "bad id\n"

    spec = (await api.client.get("/openapi.json")).json()
    assert {"/v1/jobs", "/v1/jobs/{job_id}", "/v1/notes/{notes_id}"} <= set(spec["paths"])
