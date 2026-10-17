from lecnotes.domain.base import DomainModel, NonEmptyStr
from lecnotes.domain.knowledge import KnowledgeUnit
from lecnotes.domain.video import VideoMeta


class NotesDocument(DomainModel):
    """Final output for a job: the structured units plus their rendered Markdown."""

    title: NonEmptyStr
    videos: list[VideoMeta]
    units: list[KnowledgeUnit]
    markdown: str
