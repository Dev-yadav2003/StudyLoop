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


class VerifySummary(DomainModel):
    checked: int = Field(ge=0)
    supported: int = Field(ge=0)
    revised: int = Field(ge=0)
    dropped: int = Field(ge=0)
    unverified: int = Field(ge=0, description="Not returned by the verifier: kept, flagged")

    @property
    def dropped_rate(self) -> float:
        return self.dropped / self.checked if self.checked else 0.0


class ExtractedUnits(DomainModel):
    units: list[KnowledgeUnit]
    dropped: list[str] = Field(default_factory=list, description="Units/parts removed, why")
    evidence: PerceptionResult | None = Field(
        default=None, description="All evidence the units may cite (needed by verify)"
    )
    verify: VerifySummary | None = None
