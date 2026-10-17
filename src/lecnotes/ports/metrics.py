from typing import Protocol

from lecnotes.domain import StageRun


class StageRunSink(Protocol):
    """Where per-stage metrics go. Must never raise: losing one metrics row is better than
    failing a job that otherwise succeeded."""

    async def record(self, run: StageRun) -> None: ...
