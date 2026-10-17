from pydantic import Field

from lecnotes.domain.base import DomainModel, NonEmptyStr, VideoId


class VideoRef(DomainModel):
    """A video to process, as resolved from the user's link."""

    video_id: VideoId
    playlist_id: str | None = None
    position: int | None = Field(default=None, ge=0, description="Index within the playlist")

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

    def timestamp_url(self, t_s: float) -> str:
        return f"https://youtu.be/{self.video_id}?t={int(t_s)}"
