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
from lecnotes.domain import ResolvedVideos
from lecnotes.errors import CatalogError, CatalogQuotaExceeded, InvalidLink, VideoUnavailable
from lecnotes.pipeline.context import RunContext
from lecnotes.pipeline.stages.resolve import ResolveRequest, ResolveStage

V1, V2, V3, V4 = "aaaaaaaaaa1", "bbbbbbbbbb2", "cccccccccc3", "dddddddddd4"
PLAYLIST = "PLxyz1234567890"
PLAYLIST_URL = f"https://www.youtube.com/playlist?list={PLAYLIST}"


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


def items_page(ids: list[str], next_token: str | None = None) -> httpx.Response:
    body: dict[str, Any] = {"items": [{"contentDetails": {"videoId": v}} for v in ids]}
    if next_token:
        body["nextPageToken"] = next_token
    return httpx.Response(200, json=body)


def videos_page(*items: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json={"items": list(items)})


def api_error(code: int, reason: str) -> httpx.Response:
    return httpx.Response(
        code, json={"error": {"code": code, "message": reason, "errors": [{"reason": reason}]}}
    )


@pytest.fixture
async def catalog() -> AsyncIterator[YouTubeDataApiCatalog]:
    async with httpx.AsyncClient() as client:
        yield YouTubeDataApiCatalog(
            client, "test-key", max_playlist_videos=3, retry_wait=wait_none()
        )


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

    resolved = await catalog.resolve(f"https://youtu.be/{V1}?t=30")

    [meta] = resolved.videos
    assert (meta.video_id, meta.title, meta.duration_s, meta.channel) == (V1, "Iter", 754, "Prof")
    assert resolved.playlist_id is None
    params = route.calls.last.request.url.params
    assert params["id"] == V1
    assert params["key"] == "test-key"
    assert "status" in params["part"]


@respx.mock(assert_all_called=False)
async def test_watch_link_with_list_is_the_video_not_the_playlist(
    catalog: YouTubeDataApiCatalog, respx_mock: respx.MockRouter
) -> None:
    # With arguments, @respx.mock makes its own router: register routes on the injected one.
    respx_mock.get(f"{API}/videos").mock(return_value=videos_page(video(V1)))
    playlist_route = respx_mock.get(f"{API}/playlistItems")

    resolved = await catalog.resolve(f"https://www.youtube.com/watch?v={V1}&list={PLAYLIST}")

    assert [v.video_id for v in resolved.videos] == [V1]
    assert not playlist_route.called


@respx.mock
async def test_playlist_paginates_keeps_order_and_skips_private(
    catalog: YouTubeDataApiCatalog,
) -> None:
    pages = respx.get(f"{API}/playlistItems").mock(
        side_effect=[items_page([V1, V2], next_token="page2"), items_page([V3])]
    )
    # The API returns items in arbitrary order; playlist order must be preserved.
    respx.get(f"{API}/videos").mock(
        return_value=videos_page(video(V3, title="Third"), video(V2, privacy="private"), video(V1))
    )

    resolved = await catalog.resolve(PLAYLIST_URL)

    assert [(v.video_id, v.position, v.playlist_id) for v in resolved.videos] == [
        (V1, 0, PLAYLIST),
        (V3, 1, PLAYLIST),
    ]
    assert [(s.video_id, s.reason.split()[0]) for s in resolved.skipped] == [(V2, "private")]
    assert pages.calls[1].request.url.params["pageToken"] == "page2"


@respx.mock
async def test_playlist_limit(catalog: YouTubeDataApiCatalog) -> None:
    ids = [V1, V2, V3, V4]
    respx.get(f"{API}/playlistItems").mock(return_value=items_page(ids))
    respx.get(f"{API}/videos").mock(return_value=videos_page(*(video(v) for v in ids)))

    resolved = await catalog.resolve(PLAYLIST_URL)

    assert len(resolved.videos) == 3
    assert [s.video_id for s in resolved.skipped] == [V4]
    assert "limit" in resolved.skipped[0].reason


@pytest.mark.parametrize(
    "link", ["", "https://vimeo.com/1", "https://www.youtube.com/channel/UC123", "hello"]
)
@respx.mock(assert_all_called=False)
async def test_invalid_link_makes_no_request(
    catalog: YouTubeDataApiCatalog, link: str, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(url__startswith=API)
    with pytest.raises(InvalidLink):
        await catalog.resolve(link)
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
async def test_playlist_with_no_public_videos(catalog: YouTubeDataApiCatalog) -> None:
    respx.get(f"{API}/playlistItems").mock(return_value=items_page([V1]))
    respx.get(f"{API}/videos").mock(return_value=videos_page())
    with pytest.raises(VideoUnavailable, match="no public videos"):
        await catalog.resolve(PLAYLIST_URL)


@respx.mock
async def test_missing_playlist(catalog: YouTubeDataApiCatalog) -> None:
    respx.get(f"{API}/playlistItems").mock(return_value=api_error(404, "playlistNotFound"))
    with pytest.raises(VideoUnavailable):
        await catalog.resolve(PLAYLIST_URL)


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
    resolved = await catalog.resolve(V1)
    assert resolved.videos[0].video_id == V1
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
    assert isinstance(out, ResolvedVideos)
    assert stage.cacheable is False
