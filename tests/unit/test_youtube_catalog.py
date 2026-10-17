from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import httpx
import pytest
import respx
from tenacity import wait_none

from lecnotes.adapters.catalog.youtube_data_api import (
    API,
    YouTubeDataApiCatalog,
    parse_iso8601_duration,
)
from lecnotes.domain import VideoMeta
from lecnotes.errors import (
    CatalogError,
    CatalogQuotaExceeded,
    InvalidLink,
    PlaylistNotSupported,
    VideoUnavailable,
)
from lecnotes.pipeline.context import RunContext
from lecnotes.pipeline.stages.resolve import ResolveRequest, ResolveStage

V1 = "aaaaaaaaaa1"
PLAYLIST = "PLxyz1234567890"


def video(
    vid: str,
    *,
    title: str = "Lecture",
    duration: str = "PT12M34S",
    privacy: str = "public",
    live: str = "none",
) -> dict[str, Any]:
    return {
        "id": vid,
        "snippet": {"title": title, "channelTitle": "Prof", "liveBroadcastContent": live},
        "contentDetails": {"duration": duration},
        "status": {"privacyStatus": privacy, "uploadStatus": "processed"},
    }


def videos_page(*items: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json={"items": list(items)})


def api_error(code: int, reason: str) -> httpx.Response:
    return httpx.Response(
        code, json={"error": {"code": code, "message": reason, "errors": [{"reason": reason}]}}
    )


@pytest.fixture
async def catalog() -> AsyncIterator[YouTubeDataApiCatalog]:
    async with httpx.AsyncClient() as client:
        yield YouTubeDataApiCatalog(client, "test-key", retry_wait=wait_none())


@pytest.mark.parametrize(
    ("raw", "seconds"),
    [
        ("PT12M34S", 754),
        ("PT1H2M3S", 3723),
        ("PT45S", 45),
        ("PT2H", 7200),
        ("P1DT1S", 86401),
        ("P0D", 0),
    ],
)
def test_iso8601_duration(raw: str, seconds: float) -> None:
    assert parse_iso8601_duration(raw) == seconds


@pytest.mark.parametrize("raw", ["", "PT", "12:34", "PT1X"])
def test_iso8601_duration_rejects(raw: str) -> None:
    with pytest.raises(ValueError, match="ISO 8601"):
        parse_iso8601_duration(raw)


@respx.mock
async def test_single_video(catalog: YouTubeDataApiCatalog) -> None:
    route = respx.get(f"{API}/videos").mock(return_value=videos_page(video(V1, title="Iter")))

    meta = await catalog.resolve(f"https://youtu.be/{V1}?t=30")

    assert (meta.video_id, meta.title, meta.duration_s, meta.channel) == (V1, "Iter", 754, "Prof")
    params = route.calls.last.request.url.params
    assert params["id"] == V1
    assert params["key"] == "test-key"
    assert "status" in params["part"]


@respx.mock
async def test_watch_link_inside_a_playlist_is_just_that_video(
    catalog: YouTubeDataApiCatalog,
) -> None:
    route = respx.get(f"{API}/videos").mock(return_value=videos_page(video(V1)))
    meta = await catalog.resolve(f"https://www.youtube.com/watch?v={V1}&list={PLAYLIST}&index=3")
    assert meta.video_id == V1
    assert route.call_count == 1  # one video looked up; the playlist is ignored


@respx.mock(assert_all_called=False)
async def test_playlist_link_is_rejected_without_any_request(
    catalog: YouTubeDataApiCatalog, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(url__startswith=API)
    with pytest.raises(PlaylistNotSupported) as exc_info:
        await catalog.resolve(f"https://www.youtube.com/playlist?list={PLAYLIST}")
    assert "paste its link" in exc_info.value.user_message
    assert isinstance(exc_info.value, InvalidLink)
    assert not route.called


@pytest.mark.parametrize(
    "link", ["", "https://vimeo.com/1", "https://www.youtube.com/channel/UC123", "hello"]
)
@respx.mock(assert_all_called=False)
async def test_invalid_link_makes_no_request(
    catalog: YouTubeDataApiCatalog, link: str, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(url__startswith=API)
    with pytest.raises(InvalidLink) as exc_info:
        await catalog.resolve(link)
    assert not isinstance(exc_info.value, PlaylistNotSupported)
    assert not route.called


@pytest.mark.parametrize(
    ("items", "reason"),
    [
        ([], "private or deleted"),
        ([video(V1, privacy="private")], "private video"),
        ([video(V1, privacy="unlisted")], "unlisted video"),
        ([video(V1, live="live", duration="P0D")], "live broadcast"),
        ([video(V1, live="upcoming", duration="P0D")], "upcoming broadcast"),
    ],
)
@respx.mock
async def test_unusable_video(
    catalog: YouTubeDataApiCatalog, items: list[dict[str, Any]], reason: str
) -> None:
    respx.get(f"{API}/videos").mock(return_value=videos_page(*items))
    with pytest.raises(VideoUnavailable, match=reason):
        await catalog.resolve(V1)


@respx.mock
async def test_quota_exceeded_is_typed_and_not_retried(catalog: YouTubeDataApiCatalog) -> None:
    route = respx.get(f"{API}/videos").mock(return_value=api_error(403, "quotaExceeded"))
    with pytest.raises(CatalogQuotaExceeded) as exc_info:
        await catalog.resolve(V1)
    assert exc_info.value.retryable is True
    assert route.call_count == 1


@respx.mock
async def test_bad_api_key_is_not_retried(catalog: YouTubeDataApiCatalog) -> None:
    route = respx.get(f"{API}/videos").mock(return_value=api_error(400, "keyInvalid"))
    with pytest.raises(CatalogError) as exc_info:
        await catalog.resolve(V1)
    assert exc_info.value.retryable is False
    assert route.call_count == 1


@respx.mock
async def test_server_errors_and_timeouts_are_retried(catalog: YouTubeDataApiCatalog) -> None:
    route = respx.get(f"{API}/videos").mock(
        side_effect=[httpx.Response(503), httpx.ConnectTimeout("slow"), videos_page(video(V1))]
    )
    meta = await catalog.resolve(V1)
    assert meta.video_id == V1
    assert route.call_count == 3


@respx.mock
async def test_persistent_server_error_becomes_retryable_catalog_error(
    catalog: YouTubeDataApiCatalog,
) -> None:
    route = respx.get(f"{API}/videos").mock(return_value=httpx.Response(500))
    with pytest.raises(CatalogError) as exc_info:
        await catalog.resolve(V1)
    assert exc_info.value.retryable is True
    assert route.call_count == 3


@respx.mock
async def test_resolve_stage(catalog: YouTubeDataApiCatalog) -> None:
    respx.get(f"{API}/videos").mock(return_value=videos_page(video(V1)))
    stage = ResolveStage(catalog)
    out = await stage.run(ResolveRequest(url=V1), RunContext(job_id=uuid4()))
    assert isinstance(out, VideoMeta)
    assert stage.cacheable is False
