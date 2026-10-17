"""Merge duplicate units within one video (Gemini + deterministic assembly).

Extraction runs chunk by chunk, so one concept is often extracted several times (the first
real run: zip x2, enumerate x3, manual indexing x2). Steps:

1. Candidate groups (code): union-find over units whose *core* concept names match
   (generic words like "function" removed; rapidfuzz token-set ≥ threshold) or that share an
   identical code block.
2. One LLM call per video: for each group, which units are really the same concept (it may
   split a group, never merge across groups), a combined concept + explanation for merged
   units, and one consistent set of section names for the whole video.
3. Assembly (code): spans (every timestamp), evidence, code blocks (deduplicated) and examples
   are unioned deterministically — the LLM only writes concept/explanation text, which the
   verify stage checks afterwards. Singletons keep their original text.

An LLM answer that doesn't partition a group exactly is ignored for that group (units kept
as they were) and recorded — a bad answer can't lose or duplicate content.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from lecnotes.domain import CodeBlock, Evidence, ExtractedUnits, KnowledgeUnit, VideoSpan
from lecnotes.domain.knowledge import primary_first
from lecnotes.llm.gateway import LLMGateway
from lecnotes.observability.logging import get_logger
from lecnotes.pipeline.context import RunContext
from lecnotes.pipeline.stage import Stage
from lecnotes.prompts.loader import load

log = get_logger(__name__)
PROMPT_ID = "merge"

_GENERIC = frozenset({
    "a", "an", "the", "of", "in", "for", "to", "and", "or", "with", "using", "use",
    "python", "function", "functions", "method", "methods", "basic", "introduction",
    "intro", "example", "examples", "concept", "concepts", "overview",
})  # fmt: skip
_WORD = re.compile(r"[a-z0-9_]+")
SPAN_GAP_S = 5.0


class MergedNote(BaseModel):
    source_ids: list[str] = Field(description="Ids of the notes this output note covers")
    concept: str
    explanation: str
    section: str


class GroupResult(BaseModel):
    group_id: str
    notes: list[MergedNote]


class MergeOutput(BaseModel):
    groups: list[GroupResult]


@dataclass(frozen=True)
class _Group:
    id: str
    units: list[KnowledgeUnit]


def core_concept(concept: str) -> str:
    """'`enumerate()` Function (Direct Unpacking)' → 'enumerate direct unpacking'."""
    return " ".join(w for w in _WORD.findall(concept.lower()) if w not in _GENERIC)


def _code_keys(unit: KnowledgeUnit) -> set[str]:
    return {" ".join(c.source.split()) for c in unit.code}


def candidate_groups(units: list[KnowledgeUnit], threshold: float = 90.0) -> list[list[int]]:
    """Indexes of units grouped by union-find; singletons included, in first-seen order."""
    parent = list(range(len(units)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cores = [core_concept(u.concept) for u in units]
    codes = [_code_keys(u) for u in units]
    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            same_name = bool(cores[i] and cores[j]) and (
                fuzz.token_set_ratio(cores[i], cores[j]) >= threshold
            )
            if same_name or (codes[i] & codes[j]):
                parent[find(j)] = find(i)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(units)):
        groups[find(i)].append(i)
    return sorted(groups.values(), key=lambda g: g[0])


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
    if len(ordered) == 1:  # singletons keep their text; only the section may change
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
    version = 2  # v2: merged spans ordered primary-first
    input_model = ExtractedUnits
    output_model = ExtractedUnits

    def __init__(
        self,
        gateway: LLMGateway,
        *,
        model: str,
        prompt_version: int = 1,
        concept_threshold: float = 90.0,
    ) -> None:
        self._gateway = gateway
        self._model = model
        self._prompt_version = prompt_version
        self.concept_threshold = concept_threshold

    @property
    def prompt_version(self) -> int | None:
        return self._prompt_version

    @property
    def model(self) -> str | None:
        return self._model

    def config(self, ctx: RunContext) -> dict[str, Any]:
        return {"concept_threshold": self.concept_threshold}

    async def run(self, inp: ExtractedUnits, ctx: RunContext) -> ExtractedUnits:
        if len(inp.units) < 2:
            return inp
        video = ctx.require_video()
        groups = [
            _Group(id=f"g{n}", units=[inp.units[i] for i in idx])
            for n, idx in enumerate(candidate_groups(inp.units, self.concept_threshold), start=1)
        ]
        prompt = load(PROMPT_ID, version=self._prompt_version, groups=groups)
        result = await self._gateway.generate_structured(
            prompt, MergeOutput, model=self._model, metadata={"video_id": video.video_id}
        )
        ctx.add_usage(result.usage)
        answers = {g.group_id: g for g in result.output.groups}

        merged: list[KnowledgeUnit] = []
        notes = list(inp.dropped)
        for group in groups:
            outcome = _apply(group, answers.get(group.id))
            if isinstance(outcome, str):
                notes.append(f"merge: {group.id} kept as is: {outcome}")
                merged += group.units
                continue
            for unit, sources in outcome:
                merged.append(unit)
                if len(sources) > 1:
                    notes.append(f"merge: {', '.join(sources)} -> {unit.id} ({unit.concept!r})")
        merged.sort(key=lambda u: (u.start_s, u.id))
        log.info("merge.done", units_in=len(inp.units), units_out=len(merged), groups=len(groups))
        return inp.model_copy(update={"units": merged, "dropped": notes})


def _apply(
    group: _Group, answer: GroupResult | None
) -> list[tuple[KnowledgeUnit, list[str]]] | str:
    """The group's merged units, or the reason the answer was rejected."""
    if answer is None:
        return "no answer for this group"
    by_id = {u.id: u for u in group.units}
    used = [i for note in answer.notes for i in note.source_ids]
    if sorted(used) != sorted(by_id) or any(not n.source_ids for n in answer.notes):
        return f"answer does not partition {sorted(by_id)} (got {used})"
    return [
        (assemble([by_id[i] for i in note.source_ids], note), list(note.source_ids))
        for note in answer.notes
    ]
