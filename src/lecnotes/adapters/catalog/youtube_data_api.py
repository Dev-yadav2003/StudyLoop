"""YouTube Data API v3 catalog: official, key-authenticated, safe from cloud IPs.

One link = one video (ADR 0007). Quota cost: one videos.list call = 1 unit of the
10,000/day default.
"""

from __future__ import annotations

import re
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tenacity.wait import WaitBaseT

from lecnotes.domain import VideoMeta
from lecnotes.domain.video import is_playlist_link, parse_video_id
from lecnotes.errors import (
    CatalogError,
    CatalogQuotaExceeded,
    InvalidLink,
    PlaylistNotSupported,
    VideoUnavailable,
)

API = "https://www.googleapis.com/youtube/v3"
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
        retry_attempts: int = 3,
        retry_wait: WaitBaseT | None = None,
    ) -> None:
        self._client = client
        self._key = api_key
        self._retry_attempts = retry_attempts
        self._retry_wait = retry_wait if retry_wait is not None else wait_exponential(0.5, max=8)

    async def resolve(self, link: str) -> VideoMeta:
        video_id = parse_video_id(link)
        if video_id is None:
            if is_playlist_link(link):
                raise PlaylistNotSupported(f"playlist links are not supported: {link!r}")
            raise InvalidLink(f"not a YouTube video link: {link!r}")
        params = {"part": "snippet,contentDetails,status", "id": video_id}
        items = (await self._get("videos", params)).get("items", [])
        if not items:
            raise VideoUnavailable(f"{video_id}: private or deleted")
        if reason := _unusable_reason(items[0]):
            raise VideoUnavailable(f"{video_id}: {reason}")
        return _to_meta(items[0])

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
    if response.status_code == 404:
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
