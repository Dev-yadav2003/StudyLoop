"""The HTTP API end to end on real Postgres: submit → run → poll → notes, with fakes only at
the edges (static catalog, recorded perception, scripted LLM)."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from uuid import UUID, uuid4

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from apps.api.app import create_app
from lecnotes.adapters.cache.postgres_cache import PostgresStageCache
from lecnotes.bootstrap import ApiContainer
from lecnotes.jobs.queue import JobQueue
from lecnotes.jobs.runner import JobRunner
from lecnotes.storage.db import code_schema_revision
from lecnotes.storage.repositories.notes import NotesRepository
from lecnotes.storage.repositories.stage_runs import PostgresStageRunSink
from tests.pipeline_fakes import UNIT_ID, URL, VID, fake_services, make_settings

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
    app = create_app(ApiContainer(settings, engine, queue, notes, services, runner))
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


async def test_practice_questions_are_grounded_and_cached(api: Api) -> None:
    job = (await api.client.post("/v1/jobs", json={"url": URL})).json()
    await api.runner.run_once()
    notes_id = (await api.client.get(f"/v1/jobs/{job['id']}")).json()["notes_id"]

    first = await api.client.post(f"/v1/notes/{notes_id}/qa", json={"level": "mixed", "count": 3})
    assert first.status_code == 200
    (q,) = first.json()["questions"]  # the one citing a made-up note was dropped
    assert (q["id"], q["level"]) == ("q01", "recall")
    assert q["units"] == [{"id": UNIT_ID, "concept": "enumerate()"}]
    (ts,) = q["timestamps"]
    assert ts["url"] == f"https://youtu.be/{VID}?t={int(ts['start_s'])}"

    again = await api.client.post(f"/v1/notes/{notes_id}/qa", json={"level": "mixed", "count": 3})
    assert again.json() == first.json()
    history = (await api.client.get(f"/v1/admin/jobs/{job['id']}/stages", headers=ADMIN)).json()
    assert [r["cache_hit"] for r in history if r["stage"] == "qa"] == [False, True]

    bad = await api.client.post(f"/v1/notes/{notes_id}/qa", json={"count": 50})
    assert bad.status_code == 422
    missing = await api.client.post(f"/v1/notes/{uuid4()}/qa", json={})
    assert missing.status_code == 404


async def _finished_notes(api: Api) -> tuple[str, dict]:
    job = (await api.client.post("/v1/jobs", json={"url": URL})).json()
    await api.runner.run_once()
    notes_id = (await api.client.get(f"/v1/jobs/{job['id']}")).json()["notes_id"]
    return job["id"], (await api.client.get(f"/v1/notes/{notes_id}")).json()


async def test_edit_unit_rerenders_and_persists(api: Api) -> None:
    job_id, notes = await _finished_notes(api)
    assert (notes["edited_units"], notes["revision"]) == ([], 0)
    path = f"/v1/notes/{notes['id']}/units/{UNIT_ID}"

    edited = await api.client.patch(
        path, json={"concept": "enumerate(iterable)", "examples": ["enumerate('ab')"]}
    )
    assert edited.status_code == 200
    body = edited.json()
    assert (body["edited_units"], body["revision"]) == ([UNIT_ID], 1)
    assert "### enumerate(iterable)\n" in body["markdown"]
    assert "**Example:** enumerate('ab')" in body["markdown"]
    unit = body["units"][0]
    assert unit["spans"] == notes["units"][0]["spans"]  # provenance untouched
    assert unit["evidence"] == notes["units"][0]["evidence"]

    reloaded = (await api.client.get(f"/v1/notes/{notes['id']}")).json()
    assert reloaded == body
    export = await api.client.get(f"/v1/notes/{notes['id']}/export")
    assert export.text == body["markdown"]

    # render only: one more render run, nothing upstream re-ran, no model call
    history = (await api.client.get(f"/v1/admin/jobs/{job_id}/stages", headers=ADMIN)).json()
    counts = {s: sum(r["stage"] == s for r in history) for s in ("extract", "render")}
    assert counts == {"extract": 1, "render": 2}
    assert history[-1]["stage"] == "render"
    assert history[-1]["tokens_in"] == 0


async def test_edit_unit_errors(api: Api) -> None:
    _, notes = await _finished_notes(api)
    base = f"/v1/notes/{notes['id']}/units"
    assert (await api.client.patch(f"{base}/nope", json={"concept": "x"})).status_code == 404
    missing = await api.client.patch(f"/v1/notes/{uuid4()}/units/{UNIT_ID}", json={"concept": "x"})
    assert missing.status_code == 404
    for bad in ({}, {"concept": ""}, {"spans": []}):
        assert (await api.client.patch(f"{base}/{UNIT_ID}", json=bad)).status_code == 422


async def test_concurrent_edit_is_rejected_not_lost(api: Api, engine: AsyncEngine) -> None:
    _, notes = await _finished_notes(api)
    repo = NotesRepository(engine)
    stale = await repo.get(UUID(notes["id"]))
    assert stale is not None
    ok = await api.client.patch(
        f"/v1/notes/{notes['id']}/units/{UNIT_ID}", json={"explanation": "First edit."}
    )
    assert ok.status_code == 200
    assert await repo.save_edit(stale, stale.document, UNIT_ID) is None  # based on revision 0
    current = await repo.get(UUID(notes["id"]))
    assert current is not None
    assert current.document.units[0].explanation == "First edit."


async def test_feedback_is_stored_and_listed_for_admins(api: Api) -> None:
    _, notes = await _finished_notes(api)
    path = f"/v1/notes/{notes['id']}/feedback"

    up = await api.client.post(path, json={"rating": 1})
    assert up.status_code == 201
    assert up.json()["scored"] is False  # tracing is off in tests
    down = await api.client.post(
        path, json={"rating": -1, "reason": "  Wrong output shown  ", "unit_id": UNIT_ID}
    )
    assert down.status_code == 201

    listed = (await api.client.get("/v1/admin/feedback", headers=ADMIN)).json()
    assert (listed["up"], listed["down"]) == (1, 1)
    newest = listed["items"][0]
    assert (newest["rating"], newest["reason"], newest["unit_id"]) == (
        -1,
        "Wrong output shown",
        UNIT_ID,
    )
    assert (newest["video_id"], newest["title"]) == (VID, "Python Iteration Tools")
    only_down = (await api.client.get("/v1/admin/feedback?rating=down", headers=ADMIN)).json()
    assert [i["rating"] for i in only_down["items"]] == [-1]
    assert (await api.client.get("/v1/admin/feedback")).status_code == 401


async def test_feedback_validation(api: Api) -> None:
    _, notes = await _finished_notes(api)
    path = f"/v1/notes/{notes['id']}/feedback"
    for bad in ({"rating": 0}, {"rating": 5}, {}, {"rating": 1, "reason": "x" * 1001}):
        assert (await api.client.post(path, json=bad)).status_code == 422
    unknown = await api.client.post(path, json={"rating": 1, "unit_id": "nope"})
    assert unknown.status_code == 404
    missing = await api.client.post(f"/v1/notes/{uuid4()}/feedback", json={"rating": 1})
    assert missing.status_code == 404


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
    app = create_app(ApiContainer(settings, engine, queue, notes, services, runner))
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
    assert ready.json()["checks"] == {"database": "ok", "schema": "ok"}  # no worker in tests

    echoed = await api.client.get("/healthz", headers={"X-Request-ID": "abc-123"})
    assert echoed.headers["X-Request-ID"] == "abc-123"
    unsafe = await api.client.get("/healthz", headers={"X-Request-ID": "bad id\n"})
    assert unsafe.headers["X-Request-ID"] != "bad id\n"

    spec = (await api.client.get("/openapi.json")).json()
    assert {"/v1/jobs", "/v1/jobs/{job_id}", "/v1/notes/{notes_id}"} <= set(spec["paths"])


async def test_readyz_fails_when_a_migration_is_missing(api: Api, engine: AsyncEngine) -> None:
    head = code_schema_revision()
    async with engine.begin() as conn:
        await conn.execute(text("update alembic_version set version_num = 'old'"))
    try:
        ready = await api.client.get("/readyz")
        assert ready.status_code == 503
        assert ready.json()["checks"]["schema"].startswith(f"at old, needs {head}")
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("update alembic_version set version_num = :v"), {"v": head})
