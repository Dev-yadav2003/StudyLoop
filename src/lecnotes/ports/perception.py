from typing import Protocol

from lecnotes.domain import PerceptionRun, VideoRef


class Perceiver(Protocol):
    """Turns a video into evidence: what was said and what was shown.

    The only component that looks at the video itself. Implementations must not download
    from YouTube on our servers (cloud IPs are blocked); they hand the URL to a model.
    """

    @property
    def fingerprint(self) -> dict[str, str | int | float]:
        """Everything that changes the output (prompt version, model, fps, ...).
        Becomes part of the stage cache key."""
        ...

    async def perceive(self, video: VideoRef, *, duration_s: float | None = None) -> PerceptionRun:
        """Raises PerceptionError, QuotaExceeded."""
        ...
