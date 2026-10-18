"""Chunk → knowledge units (Gemini, text only).

Faithfulness is enforced in code, not trusted to the prompt:
  - the model never outputs timestamps: a unit's span is computed from the evidence it cites;
  - cited IDs must exist in the chunk; a unit with no valid evidence is dropped;
  - code and board text must actually appear on a cited board (whitespace-insensitive fuzzy
    match), otherwise they are removed — invented code never reaches the notes.
Every removal is recorded in `ExtractedUnits.dropped`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from lecnotes.domain import (
    AlignedChunk,
    BoardState,
    Chunks,
    CodeBlock,
    Evidence,
    ExtractedUnits,
    KnowledgeUnit,
    PerceptionResult,
    TranscriptSegment,
    VideoSpan,
)
from lecnotes.domain.knowledge import primary_first
from lecnotes.llm.gateway import LLMGateway
from lecnotes.observability.logging import get_logger
from lecnotes.pipeline.context import RunContext
from lecnotes.pipeline.stage import Stage
from lecnotes.prompts.loader import load
from lecnotes.timefmt import format_timestamp

log = get_logger(__name__)
PROMPT_ID = "extract"
VERBATIM_RATIO = 90.0


class ExtractedCode(BaseModel):
    language: str = Field(description="Programming language, e.g. python; empty if unknown")
    source: str = Field(description="Code copied verbatim from a cited board state")


class ExtractedUnit(BaseModel):
    section: str = Field(description="Short topic heading this unit belongs under")
    concept: str = Field(description="Short name of the concept")
    explanation: str = Field(description="The study note for this concept, in Markdown")
    examples: list[str] = Field(description="Examples the teacher gave; empty if none")
    board_text: str = Field(description="Relevant board text copied verbatim; empty if none")
    code: list[ExtractedCode] = Field(description="Code copied verbatim from the board")
    segment_ids: list[str] = Field(description="IDs of transcript segments used")
    board_state_ids: list[str] = Field(description="IDs of board states used")


class ExtractOutput(BaseModel):
    units: list[ExtractedUnit]


def _flat(text: str) -> str:
    return " ".join(text.split())


def appears_on(text: str, boards: list[BoardState]) -> bool:
    """Whitespace-insensitive, tolerant of tiny OCR-level differences."""
    needle = _flat(text)
    if not needle:
        return False
    for board in boards:
        hay = _flat(board.text)
        if needle in hay or fuzz.partial_ratio(needle, hay) >= VERBATIM_RATIO:
            return True
    return False


class ExtractStage(Stage[Chunks, ExtractedUnits]):
    name = "extract"
    version = 3  # v2: output carries the evidence; v3: one span per moment, primary first
    input_model = Chunks
    output_model = ExtractedUnits

    def __init__(self, gateway: LLMGateway, *, model: str, prompt_version: int = 1) -> None:
        self._gateway = gateway
        self._model = model
        self._prompt_version = prompt_version

    @property
    def prompt_version(self) -> int | None:
        return self._prompt_version

    @property
    def model(self) -> str | None:
        return self._model

    def config(self, ctx: RunContext) -> dict[str, Any]:
        return {"subject_preset": ctx.options.subject_preset.value}

    async def run(self, inp: Chunks, ctx: RunContext) -> ExtractedUnits:
        video = ctx.require_video()
        units: list[KnowledgeUnit] = []
        dropped: list[str] = []
        for chunk in inp.chunks:
            prompt = load(
                PROMPT_ID,
                version=self._prompt_version,
                subject_preset=ctx.options.subject_preset.value,
                video_title=video.title,
                chunk_start=format_timestamp(chunk.start_s),
                board_states=chunk.board_states,
                segments=chunk.segments,
            )
            result = await self._gateway.generate_structured(
                prompt,
                ExtractOutput,
                model=self._model,
                metadata={"video_id": video.video_id, "chunk_id": chunk.id},
            )
            ctx.add_usage(result.usage)
            for i, raw in enumerate(result.output.units, start=1):
                unit = to_unit(raw, chunk, f"{video.video_id}-{chunk.id}-u{i:02d}", dropped)
                if unit is not None:
                    units.append(unit)
        log.info("extract.done", units=len(units), dropped=len(dropped), chunks=len(inp.chunks))
        return ExtractedUnits(
            units=units, dropped=dropped, evidence=chunk_evidence(video.video_id, inp.chunks)
        )


def chunk_evidence(video_id: str, chunks: list[AlignedChunk]) -> PerceptionResult:
    """Union of the chunks' evidence (boards may be shared by several chunks)."""
    segments = {s.id: s for c in chunks for s in c.segments}
    boards = {b.id: b for c in chunks for b in c.board_states}
    return PerceptionResult(
        video_id=video_id,
        segments=sorted(segments.values(), key=lambda s: s.start_s),
        board_states=sorted(boards.values(), key=lambda b: b.t_first_seen),
    )


def to_unit(
    raw: ExtractedUnit, chunk: AlignedChunk, unit_id: str, dropped: list[str]
) -> KnowledgeUnit | None:
    where = f"{unit_id} ({raw.concept!r})"
    segments = {s.id: s for s in chunk.segments}
    boards = {b.id: b for b in chunk.board_states}
    seg_ids = [i for i in dict.fromkeys(raw.segment_ids) if i in segments]
    board_ids = [i for i in dict.fromkeys(raw.board_state_ids) if i in boards]
    unknown = set(raw.segment_ids) - set(seg_ids) | set(raw.board_state_ids) - set(board_ids)
    if unknown:
        dropped.append(f"{where}: unknown evidence ids {sorted(unknown)} removed")
    if not seg_ids and not board_ids:
        dropped.append(f"{where}: no valid evidence, unit dropped")
        return None
    if not raw.concept.strip() or not raw.explanation.strip():
        dropped.append(f"{where}: empty concept or explanation, unit dropped")
        return None

    cited_boards = [boards[i] for i in board_ids]
    code = []
    for block in raw.code:
        if not block.source.strip():
            continue
        if appears_on(block.source, cited_boards):
            code.append(CodeBlock(source=block.source, language=block.language.strip() or None))
        else:
            dropped.append(f"{where}: code not found on cited boards, removed")
    board_text: str | None = raw.board_text if raw.board_text.strip() else None
    if board_text is not None and not appears_on(board_text, cited_boards):
        dropped.append(f"{where}: board_text not found on cited boards, removed")
        board_text = None

    return KnowledgeUnit(
        id=unit_id,
        concept=raw.concept,
        explanation=raw.explanation,
        examples=[e for e in raw.examples if e.strip()],
        board_text=board_text,
        code=code,
        section=raw.section.strip() or None,
        spans=_spans(chunk.video_id, [segments[i] for i in seg_ids], cited_boards),
        evidence=[
            Evidence(video_id=chunk.video_id, segment_ids=seg_ids, board_state_ids=board_ids)
        ],
    )


MOMENT_GAP_S = 60.0


def _spans(
    video_id: str, segments: list[TranscriptSegment], boards: list[BoardState]
) -> list[VideoSpan]:
    """From the cited evidence — never from the model. Cited speech is split into separate
    moments wherever it pauses > MOMENT_GAP_S, so one early mention (an intro saying "today:
    zip and enumerate") can't drag the link minutes before the actual teaching. The longest
    moment comes first. Without speech, the boards' visibility is used."""
    if not segments:
        start = min(b.t_first_seen for b in boards)
        end = max(b.t_last_seen for b in boards)
        return [VideoSpan(video_id=video_id, start_s=start, end_s=max(end, start + 1))]
    moments: list[list[TranscriptSegment]] = []
    for seg in sorted(segments, key=lambda s: s.start_s):
        if moments and seg.start_s - moments[-1][-1].end_s <= MOMENT_GAP_S:
            moments[-1].append(seg)
        else:
            moments.append([seg])
    spans = [
        VideoSpan(
            video_id=video_id,
            start_s=m[0].start_s,
            end_s=max(max(s.end_s for s in m), m[0].start_s + 1),
        )
        for m in moments
    ]
    return primary_first(spans)
