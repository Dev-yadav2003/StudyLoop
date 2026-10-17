"""Videos. Scope decision (ADR 0007): one link = one video. Playlists are not supported; a
watch link that also carries `list=` means the video it points at."""

import re
from typing import Self
from urllib.parse import parse_qs, urlparse

from pydantic import Field

from lecnotes.domain.base import DomainModel, NonEmptyStr, VideoId

_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_PATH_PREFIXES = ("/shorts/", "/embed/", "/live/", "/v/")


def _parse(link: str) -> tuple[str, str, str]:
    """(host without www./m., path, query)."""
    parsed = urlparse(link if "://" in link else f"https://{link}")
    host = (parsed.hostname or "").removeprefix("www.").removeprefix("m.")
    return host, parsed.path, parsed.query


def parse_video_id(link: str) -> str | None:
    """Extract an 11-char video id from a YouTube link (or a bare id). None if not found."""
    link = link.strip()
    if _ID.match(link):
        return link
    host, path, query = _parse(link)
    candidate: str | None = None
    if host == "youtu.be":
        candidate = path.lstrip("/").split("/")[0]
    elif host in ("youtube.com", "music.youtube.com", "youtube-nocookie.com"):
        if path == "/watch":
            candidate = parse_qs(query).get("v", [None])[0]
        else:
            for prefix in _PATH_PREFIXES:
                if path.startswith(prefix):
                    candidate = path.removeprefix(prefix).split("/")[0]
    return candidate if candidate and _ID.match(candidate) else None


def is_playlist_link(link: str) -> bool:
    """A bare /playlist link (no video in it): unsupported, but worth a helpful message."""
    host, path, _ = _parse(link.strip())
    return host in ("youtube.com", "music.youtube.com") and path == "/playlist"


class VideoRef(DomainModel):
    """The video a user's link points at."""

    video_id: VideoId

    @classmethod
    def from_url(cls, link: str) -> Self:
        video_id = parse_video_id(link)
        if video_id is None:
            raise ValueError(f"not a YouTube video link: {link!r}")
        return cls(video_id=video_id)

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


class VideoMeta(DomainModel):
    video_id: VideoId
    title: NonEmptyStr
    duration_s: float = Field(gt=0)
    channel: str | None = None

    @property
    def ref(self) -> VideoRef:
        return VideoRef(video_id=self.video_id)

    @property
    def url(self) -> str:
        return self.ref.url

    def timestamp_url(self, t_s: float) -> str:
        return f"https://youtu.be/{self.video_id}?t={int(t_s)}"
