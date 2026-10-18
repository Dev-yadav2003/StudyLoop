"""Sign-in, per-account libraries and the guest allowance (the owner is unlimited)."""

import time
import uuid

import jwt
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from lecnotes.storage.tables import ChapterRow
from tests.integration.conftest import GUEST_MINUTES, OWNER_EMAIL, TEST_JWT_KEY, Api, bearer
from tests.pipeline_fakes import BROKEN, BROKEN_URL, LONG_URL, URL, VIDEO

GUEST = bearer("guest@example.com")
OTHER = bearer("other@example.com")


def minutes(*videos: float) -> float:
    return round(sum(videos) / 60, 1)


async def _lesson(api: Api, url: str, headers: dict[str, str]) -> int:
    chapter = (await api.client.post("/v1/chapters", json={"name": "C"}, headers=headers)).json()
    response = await api.client.post(
        f"/v1/chapters/{chapter['id']}/lessons", json={"url": url}, headers=headers
    )
    return response.status_code


async def test_library_needs_a_valid_sign_in(api: Api) -> None:
    def token(key: str, **claims: object) -> dict[str, str]:
        payload = {"sub": str(uuid.uuid4()), "aud": "authenticated", "exp": time.time() + 60}
        return {"Authorization": f"Bearer {jwt.encode(payload | claims, key, algorithm='HS256')}"}

    for headers in (
        {"Authorization": ""},  # signed out
        token(TEST_JWT_KEY, exp=time.time() - 10),  # expired
        token("a-different-key-that-is-also-32-bytes-long"),  # forged
        token(TEST_JWT_KEY, aud="anon"),  # not a user session
        token(TEST_JWT_KEY, sub="not-a-uuid"),
    ):
        response = await api.client.get("/v1/chapters", headers=headers)
        assert response.status_code == 401, headers
        assert response.json()["error"]["code"] == "unauthorized"


async def test_me_reports_owner_and_guest_allowance(api: Api) -> None:
    owner = (await api.client.get("/v1/me")).json()
    assert (owner["email"], owner["is_owner"], owner["video_minutes_limit"]) == (
        OWNER_EMAIL,
        True,
        None,
    )
    guest = (await api.client.get("/v1/me", headers=GUEST)).json()
    assert (guest["is_owner"], guest["video_minutes_limit"], guest["video_minutes_used"]) == (
        False,
        GUEST_MINUTES,
        0,
    )


async def test_each_account_sees_only_its_own_library(api: Api) -> None:
    created = await api.client.post("/v1/chapters", json={"name": "Mine"}, headers=GUEST)
    chapter_id = created.json()["id"]

    assert (await api.client.get("/v1/chapters", headers=OTHER)).json() == []
    # a guessed id is a 404 for anyone else — never someone else's data
    for method, path, body in (
        ("PATCH", f"/v1/chapters/{chapter_id}", {"name": "hijack"}),
        ("DELETE", f"/v1/chapters/{chapter_id}", None),
        ("POST", f"/v1/chapters/{chapter_id}/lessons", {"url": URL}),
    ):
        response = await api.client.request(method, path, json=body, headers=OTHER)
        assert response.status_code == 404, (method, path)
    names = [c["name"] for c in (await api.client.get("/v1/chapters", headers=GUEST)).json()]
    assert names == ["Mine"]


async def test_guest_allowance_counts_each_video_once(api: Api) -> None:
    assert await _lesson(api, URL, GUEST) == 201
    used = minutes(VIDEO.duration_s)
    assert (await api.client.get("/v1/me", headers=GUEST)).json()["video_minutes_used"] == used

    # the same video again (another chapter) is free — already paid for
    assert await _lesson(api, f"https://youtu.be/{VIDEO.video_id}?t=5", GUEST) == 201
    assert (await api.client.get("/v1/me", headers=GUEST)).json()["video_minutes_used"] == used

    # an unavailable video is refused before anything is charged
    assert await _lesson(api, "https://youtu.be/aaaaaaaaaaa", GUEST) == 422
    assert (await api.client.get("/v1/me", headers=GUEST)).json()["video_minutes_used"] == used


async def test_guest_over_the_limit_gets_a_clear_429_and_nothing_is_queued(api: Api) -> None:
    assert await _lesson(api, URL, GUEST) == 201  # 12.6 of 20 min
    assert await _lesson(api, BROKEN_URL, GUEST) == 201  # + 4 = 16.6

    chapter = (await api.client.post("/v1/chapters", json={"name": "C"}, headers=GUEST)).json()
    refused = await api.client.post(
        f"/v1/chapters/{chapter['id']}/lessons", json={"url": LONG_URL}, headers=GUEST
    )
    assert refused.status_code == 429
    error = refused.json()["error"]
    assert error["code"] == "quota_exceeded"
    assert "30 min" in error["message"]
    assert "3 min of video left" in error["message"]
    me = (await api.client.get("/v1/me", headers=GUEST)).json()
    assert me["video_minutes_used"] == minutes(VIDEO.duration_s, BROKEN.duration_s)
    lessons = [lesson for c in (await api.client.get("/v1/chapters", headers=GUEST)).json()
               for lesson in c["lessons"]]  # fmt: skip
    assert "ccccccccccc" not in {lesson["video_id"] for lesson in lessons}


async def test_owner_is_unlimited(api: Api) -> None:
    for url in (URL, BROKEN_URL, LONG_URL):  # 46.6 min > the 20-min guest allowance
        assert await _lesson(api, url, {}) == 201
    me = (await api.client.get("/v1/me")).json()
    assert me["video_minutes_limit"] is None
    assert me["video_minutes_used"] > GUEST_MINUTES  # still recorded


async def test_only_the_owner_edits_shared_notes(api: Api) -> None:
    job = await api.submit(URL)
    await api.runner.run_once()
    notes_id = (await api.client.get(f"/v1/jobs/{job['id']}")).json()["notes_id"]
    unit_id = (await api.client.get(f"/v1/notes/{notes_id}")).json()["units"][0]["id"]
    path = f"/v1/notes/{notes_id}/units/{unit_id}"

    denied = await api.client.patch(path, json={"concept": "x"}, headers=GUEST)
    assert denied.status_code == 403
    assert (await api.client.patch(path, json={"concept": "x"})).status_code == 200


async def test_practice_questions_draw_from_the_token_budget(api: Api) -> None:
    job = await api.submit(URL)
    await api.runner.run_once()
    notes_id = (await api.client.get(f"/v1/jobs/{job['id']}")).json()["notes_id"]

    before = (await api.client.get("/v1/me", headers=GUEST)).json()["tokens_used"]
    asked = await api.client.post(f"/v1/notes/{notes_id}/qa", json={}, headers=GUEST)
    assert asked.status_code == 200
    after = (await api.client.get("/v1/me", headers=GUEST)).json()["tokens_used"]
    assert after - before == 8_000  # settings.qa_token_cost
    signed_out = await api.client.post(
        f"/v1/notes/{notes_id}/qa", json={}, headers={"Authorization": ""}
    )
    assert signed_out.status_code == 401


async def test_owner_claims_chapters_made_before_sign_in(api: Api, engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.execute(insert(ChapterRow).values(id=uuid.uuid4(), name="Old chapter"))
    assert (await api.client.get("/v1/chapters", headers=GUEST)).json() == []  # not a guest's
    names = [c["name"] for c in (await api.client.get("/v1/chapters")).json()]
    assert names == ["Old chapter"]
