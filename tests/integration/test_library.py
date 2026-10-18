"""Chapters and lessons over the real API + Postgres."""

from uuid import uuid4

from tests.integration.conftest import Api
from tests.pipeline_fakes import URL, VID


async def _chapter(api: Api, name: str = "Python") -> dict:
    created = await api.client.post("/v1/chapters", json={"name": name})
    assert created.status_code == 201
    return created.json()


async def test_chapter_lifecycle(api: Api) -> None:
    python = await _chapter(api, "  Python  ")
    assert (python["name"], python["lessons"]) == ("Python", [])  # trimmed
    java = await _chapter(api, "Java")

    renamed = await api.client.patch(f"/v1/chapters/{java['id']}", json={"name": "Java basics"})
    assert renamed.json()["name"] == "Java basics"
    listed = (await api.client.get("/v1/chapters")).json()
    assert [c["name"] for c in listed] == ["Python", "Java basics"]  # creation order

    assert (await api.client.delete(f"/v1/chapters/{java['id']}")).status_code == 204
    assert [c["name"] for c in (await api.client.get("/v1/chapters")).json()] == ["Python"]
    for bad in ({"name": ""}, {"name": "   "}, {"name": "x" * 81}):
        assert (await api.client.post("/v1/chapters", json=bad)).status_code == 422


async def test_lesson_is_queued_then_shows_its_notes(api: Api) -> None:
    chapter = await _chapter(api)
    created = await api.client.post(
        f"/v1/chapters/{chapter['id']}/lessons", json={"url": URL, "subject_preset": "stem"}
    )
    assert created.status_code == 201
    lesson = created.json()
    assert (lesson["status"], lesson["video_id"], lesson["notes_id"]) == ("queued", VID, None)
    assert lesson["title"] == URL  # no video title known yet

    await api.runner.run_once()

    done = (await api.client.get(f"/v1/lessons/{lesson['id']}")).json()
    assert done["status"] == "succeeded"
    assert done["title"] == "Python Iteration Tools"  # the video's title
    notes = await api.client.get(f"/v1/notes/{done['notes_id']}")
    assert notes.status_code == 200
    (listed,) = (await api.client.get("/v1/chapters")).json()
    assert [entry["id"] for entry in listed["lessons"]] == [lesson["id"]]


async def test_rename_move_reset_and_delete_a_lesson(api: Api) -> None:
    python, other = await _chapter(api), await _chapter(api, "Other")
    lesson = (
        await api.client.post(f"/v1/chapters/{python['id']}/lessons", json={"url": URL})
    ).json()
    path = f"/v1/lessons/{lesson['id']}"

    renamed = (await api.client.patch(path, json={"title": "Lesson 1: enumerate"})).json()
    assert (renamed["title"], renamed["custom_title"]) == (
        "Lesson 1: enumerate",
        "Lesson 1: enumerate",
    )
    moved = (await api.client.patch(path, json={"chapter_id": other["id"]})).json()
    assert (moved["chapter_id"], moved["title"]) == (other["id"], "Lesson 1: enumerate")
    reset = (await api.client.patch(path, json={"title": None})).json()
    assert reset["custom_title"] is None

    missing_chapter = await api.client.patch(path, json={"chapter_id": str(uuid4())})
    assert missing_chapter.status_code == 404
    assert (await api.client.delete(path)).status_code == 204
    assert (await api.client.get(path)).status_code == 404


async def test_same_video_in_two_chapters_shares_one_job(api: Api) -> None:
    a, b = await _chapter(api, "A"), await _chapter(api, "B")
    first = (await api.client.post(f"/v1/chapters/{a['id']}/lessons", json={"url": URL})).json()
    second = (
        await api.client.post(
            f"/v1/chapters/{b['id']}/lessons", json={"url": f"https://youtu.be/{VID}"}
        )
    ).json()
    assert first["job_id"] == second["job_id"]
    # deleting the chapter removes its lesson only; the other lesson still works
    await api.client.delete(f"/v1/chapters/{a['id']}")
    assert (await api.client.get(f"/v1/lessons/{second['id']}")).status_code == 200


async def test_bad_links_and_retry(api: Api) -> None:
    chapter = await _chapter(api)
    path = f"/v1/chapters/{chapter['id']}/lessons"
    playlist = await api.client.post(path, json={"url": "https://youtube.com/playlist?list=PL1"})
    assert playlist.status_code == 422
    missing = await api.client.post(f"/v1/chapters/{uuid4()}/lessons", json={"url": URL})
    assert missing.status_code == 404

    lesson = (await api.client.post(path, json={"url": "https://youtu.be/aaaaaaaaaaa"})).json()
    not_failed = await api.client.post(f"/v1/lessons/{lesson['id']}/retry")
    assert not_failed.status_code == 409
    await api.runner.run_once()  # unknown video → fails
    failed = (await api.client.get(f"/v1/lessons/{lesson['id']}")).json()
    assert failed["status"] == "failed"

    retried = await api.client.post(f"/v1/lessons/{lesson['id']}/retry")
    assert retried.status_code == 200
    assert retried.json()["status"] == "queued"
    assert retried.json()["job_id"] != lesson["job_id"]


async def test_cors_allows_the_methods_the_ui_uses(api: Api) -> None:
    for method in ("PATCH", "DELETE"):
        preflight = await api.client.options(
            "/v1/chapters/x",
            headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": method},
        )
        assert preflight.status_code == 200, method
