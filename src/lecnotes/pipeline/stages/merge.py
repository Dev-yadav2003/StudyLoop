"""Merge duplicate units within one video and give the notes one set of sections (Gemini +
deterministic assembly).

Extraction runs chunk by chunk, so one concept is often extracted several times — under the
same name (first real run: zip x2, enumerate x3) or a different one ("Define success
criteria" at 4:22 vs "Defining task, target and success criteria" at 9:06).

v1/v2 only showed the model candidate groups of fuzzy-matching names, so differently named
duplicates never reached it (17-min lecture: 29% of notes were repeats, judged). v3 shows it
the whole lecture in time order in ONE call:

1. The model declares the lecture's sections first (3-7, in lecture order, each one stretch
   of the lecture), then partitions the note ids into output notes, each with a section.
2. Code checks everything (`apply_answer`): unknown or reused ids are ignored, notes it left
   out are kept unchanged, a section outside the declared list is snapped to the closest
   declared one — a bad answer can't lose or duplicate content.
3. Assembly (code): spans (every timestamp), evidence, code blocks (deduplicated) and examples
   are unioned deterministically — the model only writes concept/explanation text for merged
   notes, which the verify stage checks afterwards. Single notes keep their original text.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from rapidfuzz import fuzz, process

from lecnotes.domain import CodeBlock, Evidence, ExtractedUnits, KnowledgeUnit, VideoSpan
from lecnotes.domain.knowledge import primary_first
from lecnotes.llm.gateway import LLMGateway
from lecnotes.observability.logging import get_logger
from lecnotes.pipeline.context import RunContext
from lecnotes.pipeline.stage import Stage
from lecnotes.prompts.loader import load
from lecnotes.timefmt import format_timestamp

log = get_logger(__name__)
PROMPT_ID = "merge"

SPAN_GAP_S = 5.0


class MergedNote(BaseModel):
    source_ids: list[str] = Field(description="Ids of the notes this output note covers")
    section: str
    concept: str = Field(default="", description="Only for merged notes (2+ source ids)")
    explanation: str = Field(default="", description="Only for merged notes (2+ source ids)")


class MergeOutput(BaseModel):
    sections: list[str] = Field(description="The lecture's sections, in lecture order")
    notes: list[MergedNote]


def merge_spans(spans: list[VideoSpan]) -> list[VideoSpan]:
    """Sorted; overlapping or near-adjacent spans of the same video joined."""
    out: list[VideoSpan] = []
    for s in sorted(spans, key=lambda s: (s.video_id, s.start_s)):
        last = out[-1] if out else None
        if last and last.video_id == s.video_id and s.start_s <= last.end_s + SPAN_GAP_S:
            out[-1] = last.model_copy(update={"end_s": max(last.end_s, s.end_s)})
        else:
            out.append(s)
    return out


def _merge_evidence(units: list[KnowledgeUnit]) -> list[Evidence]:
    by_video: dict[str, tuple[dict[str, None], dict[str, None]]] = {}
    for u in units:
        for e in u.evidence:
            segs, boards = by_video.setdefault(e.video_id, ({}, {}))
            segs.update(dict.fromkeys(e.segment_ids))
            boards.update(dict.fromkeys(e.board_state_ids))
    return [
        Evidence(video_id=v, segment_ids=list(s), board_state_ids=list(b))
        for v, (s, b) in by_video.items()
    ]


def assemble(sources: list[KnowledgeUnit], note: MergedNote) -> KnowledgeUnit:
    ordered = sorted(sources, key=lambda u: (u.start_s, u.id))
    first = ordered[0]
    if len(ordered) == 1:  # single notes keep their text; only the section may change
        return first.model_copy(update={"section": note.section.strip() or first.section})
    code: dict[str, CodeBlock] = {}
    for u in ordered:
        for c in u.code:
            code.setdefault(" ".join(c.source.split()), c)
    examples = list(dict.fromkeys(e for u in ordered for e in u.examples))
    boards = list(dict.fromkeys(u.board_text for u in ordered if u.board_text))
    return KnowledgeUnit(
        id=first.id,
        concept=note.concept.strip() or first.concept,
        explanation=note.explanation.strip() or first.explanation,
        examples=examples,
        board_text="\n\n".join(boards) if boards else None,
        code=list(code.values()),
        section=note.section.strip() or first.section,
        spans=primary_first(merge_spans([s for u in ordered for s in u.spans])),
        evidence=_merge_evidence(ordered),
    )


class MergeStage(Stage[ExtractedUnits, ExtractedUnits]):
    name = "merge"
    version = 3  # v2: merged spans ordered primary-first; v3: whole lecture + declared sections
    input_model = ExtractedUnits
    output_model = ExtractedUnits

    def __init__(self, gateway: LLMGateway, *, model: str, prompt_version: int = 2) -> None:
        if prompt_version < 2:
            raise ValueError("merge stage v3 needs merge prompt v2+ (v1 answers per group)")
        self._gateway = gateway
        self._model = model
        self._prompt_version = prompt_version

    @property
    def prompt_version(self) -> int | None:
        return self._prompt_version

    @property
    def model(self) -> str | None:
        return self._model

    async def run(self, inp: ExtractedUnits, ctx: RunContext) -> ExtractedUnits:
        if len(inp.units) < 2:
            return inp
        video = ctx.require_video()
        units = sorted(inp.units, key=lambda u: (u.start_s, u.id))
        prompt = load(
            PROMPT_ID,
            version=self._prompt_version,
            notes=[{"at": format_timestamp(u.start_s), "unit": u} for u in units],
        )
        result = await self._gateway.generate_structured(
            prompt, MergeOutput, model=self._model, metadata={"video_id": video.video_id}
        )
        ctx.add_usage(result.usage)
        merged, notes = apply_answer(units, result.output)
        log.info(
            "merge.done",
            units_in=len(inp.units),
            units_out=len(merged),
            sections=len({u.section for u in merged}),
        )
        return inp.model_copy(update={"units": merged, "dropped": [*inp.dropped, *notes]})


def apply_answer(
    units: list[KnowledgeUnit], answer: MergeOutput
) -> tuple[list[KnowledgeUnit], list[str]]:
    """The merged units (in time order) and what happened, from a possibly bad answer."""
    by_id = {u.id: u for u in units}
    declared = list(dict.fromkeys(s.strip() for s in answer.sections if s.strip()))
    log_: list[str] = []
    used: set[str] = set()
    out: list[KnowledgeUnit] = []
    for n, note in enumerate(answer.notes, start=1):
        ids = [i for i in dict.fromkeys(note.source_ids) if i in by_id and i not in used]
        if len(ids) < len(note.source_ids):
            bad = [i for i in note.source_ids if i not in ids]
            log_.append(f"merge: note {n} ignored unknown/reused ids {bad}")
        if not ids:
            continue
        used.update(ids)
        section = _snap(note.section, declared)
        unit = assemble([by_id[i] for i in ids], note.model_copy(update={"section": section}))
        out.append(unit)
        if len(ids) > 1:
            log_.append(f"merge: {', '.join(ids)} -> {unit.id} ({unit.concept!r})")
    missing = [u for u in units if u.id not in used]
    if missing:
        log_.append(f"merge: kept unchanged (not in answer): {[u.id for u in missing]}")
        out += [_nearest_section(u, out) for u in missing]
    out.sort(key=lambda u: (u.start_s, u.id))
    return out, log_


def _snap(section: str, declared: list[str]) -> str:
    """A section name from the declared list: exact, else the closest one."""
    section = section.strip()
    if not declared or section in declared:
        return section
    best = process.extractOne(section, declared, scorer=fuzz.token_set_ratio)
    return best[0] if best else section


def _nearest_section(unit: KnowledgeUnit, placed: list[KnowledgeUnit]) -> KnowledgeUnit:
    """A note the answer left out joins the section of the closest note in time, so it
    doesn't bring back an extraction-time section name."""
    if not placed:
        return unit
    nearest = min(placed, key=lambda u: abs(u.start_s - unit.start_s))
    return unit.model_copy(update={"section": nearest.section})
