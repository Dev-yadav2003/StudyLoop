"""Pure data contracts shared by every stage. Must not import anything else from lecnotes."""

from lecnotes.domain.alignment import AlignedChunk, AlignmentLink, AlignmentMethod
from lecnotes.domain.evidence import (
    Alignment,
    Chunks,
    ExtractedUnits,
    ValidatedEvidence,
    ValidationIssue,
)
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
from lecnotes.domain.video import ResolvedVideos, SkippedVideo, VideoMeta, VideoRef
from lecnotes.domain.visual import UNCLEAR_MARKER, BoardState, BoardType

__all__ = [
    "UNCLEAR_MARKER",
    "AlignedChunk",
    "Alignment",
    "AlignmentLink",
    "AlignmentMethod",
    "BoardState",
    "BoardType",
    "Chunks",
    "CodeBlock",
    "Evidence",
    "ExtractedUnits",
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
    "ResolvedVideos",
    "SkippedVideo",
    "StageRun",
    "StageRunStatus",
    "SubjectPreset",
    "TranscriptSegment",
    "TranscriptSource",
    "ValidatedEvidence",
    "ValidationIssue",
    "VideoMeta",
    "VideoRef",
    "VideoSpan",
]
