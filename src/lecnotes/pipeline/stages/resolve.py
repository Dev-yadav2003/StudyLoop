from __future__ import annotations

from pydantic import BaseModel

from lecnotes.domain import VideoMeta
from lecnotes.observability.logging import get_logger
from lecnotes.pipeline.context import RunContext
from lecnotes.pipeline.stage import Stage
from lecnotes.ports.video_catalog import VideoCatalog

log = get_logger(__name__)


class ResolveRequest(BaseModel):
    url: str


class ResolveStage(Stage[ResolveRequest, VideoMeta]):
    """Link → metadata of the one video it points at, checking availability before any
    Gemini quota is spent. Not cached: availability can change, and a lookup costs 1 unit."""

    name = "resolve"
    version = 1
    input_model = ResolveRequest
    output_model = VideoMeta
    cacheable = False

    def __init__(self, catalog: VideoCatalog) -> None:
        self._catalog = catalog

    async def run(self, inp: ResolveRequest, ctx: RunContext) -> VideoMeta:
        video = await self._catalog.resolve(inp.url)
        log.info("resolve.done", video_id=video.video_id, duration_s=video.duration_s)
        return video
