from __future__ import annotations

from lecnotes.domain import VideoMeta
from lecnotes.domain.video import is_playlist_link, parse_video_id
from lecnotes.errors import InvalidLink, PlaylistNotSupported, VideoUnavailable


class StaticCatalog:
    """Known videos only, no network: tests and offline runs on recorded perception."""

    def __init__(self, videos: list[VideoMeta]) -> None:
        self._videos = {v.video_id: v for v in videos}

    async def resolve(self, link: str) -> VideoMeta:
        video_id = parse_video_id(link)
        if video_id is None:
            if is_playlist_link(link):
                raise PlaylistNotSupported(f"playlist links are not supported: {link!r}")
            raise InvalidLink(f"not a YouTube video link: {link!r}")
        if video_id not in self._videos:
            raise VideoUnavailable(f"{video_id}: not in the static catalog")
        return self._videos[video_id]
