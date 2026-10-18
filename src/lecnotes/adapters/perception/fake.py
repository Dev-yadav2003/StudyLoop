"""Perceiver that returns recorded results: drives every downstream stage in tests offline."""

from __future__ import annotations

from collections.abc import Mapping

from lecnotes.domain import PerceptionResult, PerceptionRun, VideoRef
from lecnotes.errors import PerceptionError


class FakePerceiver:
    def __init__(
        self,
        results: Mapping[str, PerceptionResult | PerceptionRun],
        *,
        prompt_version: int = 1,
        model: str = "gemini-2.5-flash",
    ) -> None:
        self._runs = {
            video_id: r if isinstance(r, PerceptionRun) else PerceptionRun(result=r, usage=[])
            for video_id, r in results.items()
        }
        self._prompt_version = prompt_version
        self._model = model
        self.calls: list[str] = []

    @property
    def fingerprint(self) -> dict[str, str | int | float]:
        return {"adapter": "fake", "prompt_version": self._prompt_version, "model": self._model}

    async def perceive(self, video: VideoRef, *, duration_s: float | None = None) -> PerceptionRun:
        self.calls.append(video.video_id)
        try:
            return self._runs[video.video_id]
        except KeyError:
            raise PerceptionError(f"{video.video_id}: no recorded perception") from None
