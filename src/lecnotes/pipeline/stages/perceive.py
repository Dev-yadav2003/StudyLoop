from __future__ import annotations

from typing import Any

from lecnotes.domain import PerceptionRun, VideoMeta
from lecnotes.pipeline.context import RunContext
from lecnotes.pipeline.stage import Stage
from lecnotes.ports.perception import Perceiver


class PerceiveStage(Stage[VideoMeta, PerceptionRun]):
    """The only stage that looks at the video. Its output is cached until the stage version or
    anything in the perceiver's fingerprint (prompt version, model, fps, resolution) changes:
    re-perceiving costs real quota (8 h/day of YouTube video on the free tier)."""

    name = "perceive"
    version = 1
    input_model = VideoMeta
    output_model = PerceptionRun

    def __init__(self, perceiver: Perceiver) -> None:
        self._perceiver = perceiver

    @property
    def prompt_version(self) -> int | None:
        value = self._perceiver.fingerprint.get("prompt_version")
        return int(value) if value is not None else None

    @property
    def model(self) -> str | None:
        value = self._perceiver.fingerprint.get("model")
        return str(value) if value is not None else None

    def config(self, ctx: RunContext) -> dict[str, Any]:
        return dict(self._perceiver.fingerprint)

    async def run(self, inp: VideoMeta, ctx: RunContext) -> PerceptionRun:
        # The duration bounds the model's timestamps.
        run = await self._perceiver.perceive(inp.ref, duration_s=inp.duration_s)
        ctx.add_usage(*run.usage)
        return run
