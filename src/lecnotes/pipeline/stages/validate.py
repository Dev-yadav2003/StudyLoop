"""Sanity-check perception before anything is built on it (no AI).

Segments: drop if they start after the video ends, clip to the duration, sort, drop
(near-)duplicates, and trim overlaps — trimming, not concatenating, so sloppy model times
can't snowball into one giant segment and ruin alignment granularity.

Board states: drop empty ones, clip, sort, merge near-identical *consecutive* states (a
slide re-shown later stays separate: backward transitions are linked, not de-duplicated),
and measure the share of [unclear] text.

Every change is recorded as a ValidationIssue and logged.
"""

from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz

from lecnotes.domain import (
    UNCLEAR_MARKER,
    BoardState,
    PerceptionResult,
    PerceptionRun,
    TranscriptSegment,
    ValidatedEvidence,
    ValidationIssue,
)
from lecnotes.errors import PerceptionError
from lecnotes.observability.logging import get_logger
from lecnotes.pipeline.context import RunContext
from lecnotes.pipeline.stage import Stage

log = get_logger(__name__)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def unclear_ratio(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    return min(1.0, text.count(UNCLEAR_MARKER) / len(words))


class ValidateStage(Stage[PerceptionRun, ValidatedEvidence]):
    name = "validate"
    version = 1
    input_model = PerceptionRun
    output_model = ValidatedEvidence

    def __init__(
        self,
        *,
        duplicate_ratio: float = 95.0,
        overlap_tolerance_s: float = 0.25,
        min_segment_s: float = 0.5,
        unclear_flag_ratio: float = 0.2,
    ) -> None:
        self.duplicate_ratio = duplicate_ratio
        self.overlap_tolerance_s = overlap_tolerance_s
        self.min_segment_s = min_segment_s
        self.unclear_flag_ratio = unclear_flag_ratio

    def config(self, ctx: RunContext) -> dict[str, Any]:
        return {
            "duplicate_ratio": self.duplicate_ratio,
            "overlap_tolerance_s": self.overlap_tolerance_s,
            "min_segment_s": self.min_segment_s,
            "unclear_flag_ratio": self.unclear_flag_ratio,
        }

    async def run(self, inp: PerceptionRun, ctx: RunContext) -> ValidatedEvidence:
        duration = ctx.require_video().duration_s
        issues: list[ValidationIssue] = []
        segments = self._segments(inp.result.segments, duration, issues)
        boards = self._boards(inp.result.board_states, duration, issues)
        ratios = {b.id: unclear_ratio(b.text) for b in boards}
        for board_id, ratio in ratios.items():
            if ratio > self.unclear_flag_ratio:
                issues.append(
                    ValidationIssue(
                        action="flagged", item_id=board_id, reason=f"{ratio:.0%} [unclear]"
                    )
                )

        if not segments and not boards:
            raise PerceptionError(f"{inp.result.video_id}: nothing usable after validation")

        for issue in issues:
            log.info("validate.issue", **issue.model_dump())
        return ValidatedEvidence(
            evidence=PerceptionResult(
                video_id=inp.result.video_id, segments=segments, board_states=boards
            ),
            issues=issues,
            unclear_ratio=ratios,
        )

    def _segments(
        self, raw: list[TranscriptSegment], duration: float, issues: list[ValidationIssue]
    ) -> list[TranscriptSegment]:
        clipped: list[TranscriptSegment] = []
        for s in raw:
            if s.start_s >= duration:
                issues.append(_issue("dropped", s.id, f"starts at {s.start_s}s, after video end"))
                continue
            if s.end_s > duration:
                issues.append(_issue("repaired", s.id, f"end {s.end_s}s clipped to {duration}s"))
                s = s.model_copy(update={"end_s": duration})
            clipped.append(s)

        kept: list[TranscriptSegment] = []
        for s in sorted(clipped, key=lambda s: (s.start_s, s.end_s)):
            prev = kept[-1] if kept else None
            if prev is None or s.start_s >= prev.end_s - self.overlap_tolerance_s:
                kept.append(s)
            elif fuzz.ratio(_norm(s.text), _norm(prev.text)) >= self.duplicate_ratio:
                kept[-1] = prev.model_copy(update={"end_s": max(prev.end_s, s.end_s)})
                issues.append(_issue("merged", s.id, f"duplicate of {prev.id}"))
            elif s.end_s - prev.end_s >= self.min_segment_s:
                kept.append(s.model_copy(update={"start_s": prev.end_s}))
                issues.append(_issue("repaired", s.id, f"overlap with {prev.id} trimmed"))
            else:
                kept[-1] = prev.model_copy(
                    update={"end_s": max(prev.end_s, s.end_s), "text": f"{prev.text} {s.text}"}
                )
                issues.append(_issue("merged", s.id, f"mostly inside {prev.id}; text appended"))
        return kept

    def _boards(
        self, raw: list[BoardState], duration: float, issues: list[ValidationIssue]
    ) -> list[BoardState]:
        clipped: list[BoardState] = []
        for b in raw:
            if b.is_empty:
                issues.append(_issue("dropped", b.id, "no text and no description"))
                continue
            if b.t_first_seen >= duration:
                issues.append(_issue("dropped", b.id, f"first seen {b.t_first_seen}s, after end"))
                continue
            if b.t_last_seen > duration:
                issues.append(_issue("repaired", b.id, f"last seen clipped to {duration}s"))
                b = b.model_copy(update={"t_last_seen": duration})
            clipped.append(b)

        kept: list[BoardState] = []
        for b in sorted(clipped, key=lambda b: (b.t_first_seen, b.t_last_seen)):
            prev = kept[-1] if kept else None
            if prev is not None and prev.type == b.type and self._same_content(prev, b):
                kept[-1] = prev.model_copy(
                    update={"t_last_seen": max(prev.t_last_seen, b.t_last_seen)}
                )
                issues.append(_issue("merged", b.id, f"same content as {prev.id}"))
            else:
                kept.append(b)
        return kept

    def _same_content(self, a: BoardState, b: BoardState) -> bool:
        left = " ".join(f"{a.text} {a.description}".split())
        right = " ".join(f"{b.text} {b.description}".split())
        return fuzz.ratio(left, right) >= self.duplicate_ratio


def _issue(action: str, item_id: str, reason: str) -> ValidationIssue:
    return ValidationIssue.model_validate({"action": action, "item_id": item_id, "reason": reason})
