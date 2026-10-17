"""YouTube Data API v3 catalog: official, key-authenticated, safe from cloud IPs.

Quota cost: videos.list = 1 unit per call (up to 50 ids), playlistItems.list = 1 unit per
page of 50. A 200-video playlist costs ~8 units of the 10,000/day default.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tenacity.wait import WaitBaseT

from lecnotes.domain import ResolvedVideos, SkippedVideo, VideoMeta
from lecnotes.domain.video import parse_playlist_id, parse_video_id
from lecnotes.errors import CatalogError, CatalogQuotaExceeded, InvalidLink, VideoUnavailable

API = "https://www.googleapis.com/youtube/v3"
PAGE = 50
_DURATION = re.compile(
    r"^P(?:(?P<d>\d+)D)?(?:T(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+(?:\.\d+)?)S)?)?$"
)
_QUOTA_REASONS = {"quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded"}


class _Transient(Exception):
    """5xx or transport failure: worth retrying."""


def parse_iso8601_duration(value: str) -> float:
    """YouTube `contentDetails.duration`, e.g. PT1H2M3S → 3723.0. P0D (live) → 0."""
    match = _DURATION.match(value)
    if not match or value in ("P", "PT"):
        raise ValueError(f"not an ISO 8601 duration: {value!r}")
    parts = {k: float(v) for k, v in match.groupdict().items() if v}
    return (
        parts.get("d", 0) * 86400
        + parts.get("h", 0) * 3600
        + parts.get("m", 0) * 60
        + parts.get("s", 0)
    )


class YouTubeDataApiCatalog:
    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        *,
        max_playlist_videos: int = 50,
        retry_attempts: int = 3,
        retry_wait: WaitBaseT | None = None,
    ) -> None:
        self._client = client
        self._key = api_key
        self._max_playlist_videos = max_playlist_videos
        self._retry_attempts = retry_attempts
        self._retry_wait = retry_wait if retry_wait is not None else wait_exponential(0.5, max=8)

    async def resolve(self, link: str) -> ResolvedVideos:
        if video_id := parse_video_id(link):
            [meta], skipped = await self._videos([video_id])
            if meta is None:
                raise VideoUnavailable(f"{video_id}: {skipped[0].reason}")
            return ResolvedVideos(videos=[meta])
        if playlist_id := parse_playlist_id(link):
            return await self._playlist(playlist_id)
        raise InvalidLink(f"not a YouTube video or playlist link: {link!r}")

    async def _playlist(self, playlist_id: str) -> ResolvedVideos:
        ids: list[str] = []
        token: str | None = None
        while True:
            params = {"part": "contentDetails", "playlistId": playlist_id, "maxResults": PAGE}
            data = await self._get(
                "playlistItems", params | ({"pageToken": token} if token else {})
            )
            ids += [item["contentDetails"]["videoId"] for item in data.get("items", [])]
            token = data.get("nextPageToken")
            if not token or len(ids) > self._max_playlist_videos:
                break

        skipped = [
            SkippedVideo(video_id=v, reason=f"playlist limit of {self._max_playlist_videos}")
            for v in ids[self._max_playlist_videos :]
        ]
        metas, unavailable = await self._videos(ids[: self._max_playlist_videos])
        usable = [
            m.model_copy(update={"playlist_id": playlist_id, "position": i})
            for i, m in enumerate(m for m in metas if m is not None)
        ]
        if not usable:
            raise VideoUnavailable(f"playlist {playlist_id}: no public videos")
        return ResolvedVideos(videos=usable, playlist_id=playlist_id, skipped=unavailable + skipped)

    async def _videos(self, ids: list[str]) -> tuple[list[VideoMeta | None], list[SkippedVideo]]:
        """Metadata in the order of `ids`; None (+ a SkippedVideo) for unusable ones."""
        found: dict[str, dict[str, Any]] = {}
        for batch in _batches(ids, PAGE):
            params = {"part": "snippet,contentDetails,status", "id": ",".join(batch)}
            data = await self._get("videos", params)
            found |= {item["id"]: item for item in data.get("items", [])}

        metas: list[VideoMeta | None] = []
        skipped: list[SkippedVideo] = []
        for video_id in ids:
            item = found.get(video_id)
            reason = "private or deleted" if item is None else _unusable_reason(item)
            if reason:
                metas.append(None)
                skipped.append(SkippedVideo(video_id=video_id, reason=reason))
            else:
                assert item is not None
                metas.append(_to_meta(item))
        return metas, skipped

    async def _get(self, resource: str, params: dict[str, Any]) -> dict[str, Any]:
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self._retry_attempts),
            wait=self._retry_wait,
            retry=retry_if_exception_type(_Transient),
            reraise=True,
        )
        try:
            async for attempt in retrying:
                with attempt:
                    return await self._get_once(resource, params)
        except _Transient as exc:
            raise CatalogError(f"YouTube API {resource}: {exc}", retryable=True) from exc
        raise AssertionError("unreachable")  # pragma: no cover

    async def _get_once(self, resource: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.get(
                f"{API}/{resource}", params=params | {"key": self._key}
            )
        except httpx.TransportError as exc:
            raise _Transient(f"{type(exc).__name__}: {exc}") from exc
        if response.status_code >= 500:
            raise _Transient(f"HTTP {response.status_code}")
        if response.status_code >= 400:
            raise _client_error(resource, response)
        return response.json()


def _client_error(resource: str, response: httpx.Response) -> Exception:
    try:
        error = response.json().get("error", {})
    except ValueError:
        error = {}
    reasons = {e.get("reason") for e in error.get("errors", [])}
    message = f"YouTube API {resource}: HTTP {response.status_code} {error.get('message', '')}"
    if reasons & _QUOTA_REASONS:
        return CatalogQuotaExceeded(message)
    if response.status_code == 404 or "playlistNotFound" in reasons:
        return VideoUnavailable(message)
    return CatalogError(message)  # 400/401/403 (bad key, API disabled): not retryable


def _unusable_reason(item: dict[str, Any]) -> str | None:
    status, snippet = item.get("status", {}), item.get("snippet", {})
    if status.get("privacyStatus") != "public":
        return f"{status.get('privacyStatus', 'unknown')} video (Gemini reads public videos only)"
    if status.get("uploadStatus", "processed") != "processed":
        return f"upload status {status['uploadStatus']}"
    if snippet.get("liveBroadcastContent", "none") != "none":
        return f"{snippet['liveBroadcastContent']} broadcast"
    if parse_iso8601_duration(item.get("contentDetails", {}).get("duration", "P0D")) <= 0:
        return "no duration (live or not processed)"
    return None


def _to_meta(item: dict[str, Any]) -> VideoMeta:
    snippet = item["snippet"]
    return VideoMeta(
        video_id=item["id"],
        title=snippet.get("title") or item["id"],
        duration_s=parse_iso8601_duration(item["contentDetails"]["duration"]),
        channel=snippet.get("channelTitle"),
    )


def _batches(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
