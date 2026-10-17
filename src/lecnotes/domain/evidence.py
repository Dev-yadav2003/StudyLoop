"""Stage outputs between perception and extraction."""

from typing import Literal

from pydantic import Field

from lecnotes.domain.alignment import AlignedChunk, AlignmentLink
from lecnotes.domain.base import DomainModel, NonEmptyStr, Score
from lecnotes.domain.knowledge import KnowledgeUnit
from lecnotes.domain.perception import PerceptionResult


class ValidationIssue(DomainModel):
    action: Literal["repaired", "dropped", "merged", "flagged"]
    item_id: NonEmptyStr
    reason: NonEmptyStr


class ValidatedEvidence(DomainModel):
    """Perception after sanity checks: sorted, clipped, de-duplicated, every change explained."""

    evidence: PerceptionResult
    issues: list[ValidationIssue] = Field(default_factory=list)
    unclear_ratio: dict[str, Score] = Field(
        default_factory=dict, description="board state id -> share of [unclear] tokens"
    )


class Alignment(DomainModel):
    evidence: PerceptionResult
    links: list[AlignmentLink]


class Chunks(DomainModel):
    chunks: list[AlignedChunk]


class ExtractedUnits(DomainModel):
    units: list[KnowledgeUnit]
    dropped: list[str] = Field(default_factory=list, description="Model units rejected, why")
