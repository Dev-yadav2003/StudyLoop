"""Per-run state handed to every stage. Dependencies (gateway, perceiver, ...) are NOT here:
stages receive them in their constructors. Logging context and spans are ambient
(contextvars), so they are not here either."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from lecnotes.domain import JobOptions, LLMUsage, VideoMeta


@dataclass
class RunContext:
    job_id: UUID
    options: JobOptions = field(default_factory=JobOptions)
    video: VideoMeta | None = None
    """The video being processed by per-video stages (None for job-level stages)."""
    _usage: list[LLMUsage] = field(default_factory=list)

    def add_usage(self, *usage: LLMUsage) -> None:
        """Stages report every LLM call here; the orchestrator turns it into stage metrics."""
        self._usage.extend(usage)

    def require_video(self) -> VideoMeta:
        if self.video is None:
            raise RuntimeError("this stage runs per video, but RunContext.video is not set")
        return self.video

    def take_usage(self) -> list[LLMUsage]:
        usage, self._usage = self._usage, []
        return usage
