from pydantic import Field

from lecnotes.domain.base import DomainModel, NonEmptyStr
from lecnotes.domain.evidence import VerifySummary
from lecnotes.domain.knowledge import KnowledgeUnit
from lecnotes.domain.perception import PerceptionResult
from lecnotes.domain.video import VideoMeta


class NotesDocument(DomainModel):
    """Final output for a job: the structured units plus their rendered Markdown."""

    title: NonEmptyStr
    videos: list[VideoMeta]
    units: list[KnowledgeUnit]
    markdown: str
    evidence: PerceptionResult | None = Field(
        default=None, description="Evidence the units cite: grounds evals and Q&A"
    )
    verify: VerifySummary | None = None
