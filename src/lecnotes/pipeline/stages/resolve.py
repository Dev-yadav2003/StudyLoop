from __future__ import annotations

from pydantic import BaseModel

from lecnotes.domain import ResolvedVideos
from lecnotes.observability.logging import get_logger
from lecnotes.pipeline.context import RunContext
from lecnotes.pipeline.stage import Stage
from lecnotes.ports.video_catalog import VideoCatalog

log = get_logger(__name__)


class ResolveRequest(BaseModel):
    url: str


class ResolveStage(Stage[ResolveRequest, ResolvedVideos]):
    """Link → video metadata (and availability) before any Gemini quota is spent.
    Not cached: playlists change, and a lookup costs only 1-2 API units."""

    name = "resolve"
    version = 1
    input_model = ResolveRequest
    output_model = ResolvedVideos
    cacheable = False

    def __init__(self, catalog: VideoCatalog) -> None:
        self._catalog = catalog

    async def run(self, inp: ResolveRequest, ctx: RunContext) -> ResolvedVideos:
        resolved = await self._catalog.resolve(inp.url)
        for skipped in resolved.skipped:
            log.warning(
                "resolve.video_skipped", skipped_video=skipped.video_id, reason=skipped.reason
            )
        log.info("resolve.done", videos=len(resolved.videos), playlist_id=resolved.playlist_id)
        return resolved
