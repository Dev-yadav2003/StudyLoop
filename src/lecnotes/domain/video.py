import re
from typing import Self
from urllib.parse import parse_qs, urlparse

from pydantic import Field

from lecnotes.domain.base import DomainModel, NonEmptyStr, VideoId

_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_PATH_PREFIXES = ("/shorts/", "/embed/", "/live/", "/v/")


def parse_video_id(link: str) -> str | None:
    """Extract an 11-char video id from a YouTube link (or a bare id). None if not found."""
    link = link.strip()
    if _ID.match(link):
        return link
    parsed = urlparse(link if "://" in link else f"https://{link}")
    host = (parsed.hostname or "").removeprefix("www.").removeprefix("m.")
    candidate: str | None = None
    if host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
    elif host in ("youtube.com", "music.youtube.com", "youtube-nocookie.com"):
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [None])[0]
        else:
            for prefix in _PATH_PREFIXES:
                if parsed.path.startswith(prefix):
                    candidate = parsed.path.removeprefix(prefix).split("/")[0]
    return candidate if candidate and _ID.match(candidate) else None


_PLAYLIST_ID = re.compile(r"^[A-Za-z0-9_-]{10,64}$")


def parse_playlist_id(link: str) -> str | None:
    """`list=` id from a /playlist link. Watch links with `list=` are treated as the video."""
    parsed = urlparse(link.strip() if "://" in link else f"https://{link.strip()}")
    host = (parsed.hostname or "").removeprefix("www.").removeprefix("m.")
    if host not in ("youtube.com", "music.youtube.com") or parsed.path != "/playlist":
        return None
    candidate = parse_qs(parsed.query).get("list", [None])[0]
    return candidate if candidate and _PLAYLIST_ID.match(candidate) else None


class VideoRef(DomainModel):
    """A video to process, as resolved from the user's link."""

    video_id: VideoId
    playlist_id: str | None = None
    position: int | None = Field(default=None, ge=0, description="Index within the playlist")

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
    playlist_id: str | None = None
    position: int | None = Field(default=None, ge=0)

    @property
    def ref(self) -> VideoRef:
        return VideoRef(
            video_id=self.video_id, playlist_id=self.playlist_id, position=self.position
        )

    @property
    def url(self) -> str:
        return self.ref.url

    def timestamp_url(self, t_s: float) -> str:
        return f"https://youtu.be/{self.video_id}?t={int(t_s)}"


class SkippedVideo(DomainModel):
    video_id: str
    reason: NonEmptyStr


class ResolvedVideos(DomainModel):
    """What a user's link expands to: one video, or a playlist's usable videos in order."""

    videos: list[VideoMeta] = Field(min_length=1)
    playlist_id: str | None = None
    skipped: list[SkippedVideo] = Field(default_factory=list)
