"""Pure data contracts shared by every stage. Must not import anything else from lecnotes."""

from lecnotes.domain.alignment import AlignedChunk, AlignmentLink, AlignmentMethod
from lecnotes.domain.job import (
    Job,
    JobOptions,
    JobStatus,
    StageRun,
    StageRunStatus,
    SubjectPreset,
)
from lecnotes.domain.knowledge import CodeBlock, Evidence, KnowledgeUnit, VideoSpan
from lecnotes.domain.notes import NotesDocument
from lecnotes.domain.perception import PerceptionResult, PerceptionRun
from lecnotes.domain.qa import Question, QuestionLevel
from lecnotes.domain.transcript import TranscriptSegment, TranscriptSource
from lecnotes.domain.usage import LLMUsage
from lecnotes.domain.video import VideoMeta, VideoRef
from lecnotes.domain.visual import UNCLEAR_MARKER, BoardState, BoardType

__all__ = [
    "UNCLEAR_MARKER",
    "AlignedChunk",
    "AlignmentLink",
    "AlignmentMethod",
    "BoardState",
    "BoardType",
    "CodeBlock",
    "Evidence",
    "Job",
    "JobOptions",
    "JobStatus",
    "KnowledgeUnit",
    "LLMUsage",
    "NotesDocument",
    "PerceptionResult",
    "PerceptionRun",
    "Question",
    "QuestionLevel",
    "StageRun",
    "StageRunStatus",
    "SubjectPreset",
    "TranscriptSegment",
    "TranscriptSource",
    "VideoMeta",
    "VideoRef",
    "VideoSpan",
]
