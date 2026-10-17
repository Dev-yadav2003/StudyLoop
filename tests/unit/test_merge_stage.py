from uuid import uuid4

import pytest

from lecnotes.adapters.llm.fake import FakeLLM
from lecnotes.domain import (
    CodeBlock,
    Evidence,
    ExtractedUnits,
    KnowledgeUnit,
    VideoMeta,
    VideoSpan,
)
from lecnotes.llm.gateway import LLMGateway
from lecnotes.pipeline.context import RunContext
from lecnotes.pipeline.stages.merge import (
    GroupResult,
    MergedNote,
    MergeOutput,
    MergeStage,
    candidate_groups,
    core_concept,
    merge_spans,
)

VID = "Ek49cGiAOwo"
VIDEO = VideoMeta(video_id=VID, title="Zip and Enumerate", duration_s=347)
ZIP_CODE = "for name, age in zip(names, ages):\n    print(name, age)"
ENUM_CODE = "for i, name in enumerate(names):\n    print(name, i)"


def unit(
    id: str,
    concept: str,
    start: float,
    *,
    code: str | None = None,
    examples: list[str] | None = None,
    segs: list[str] | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=id,
        concept=concept,
        explanation=f"About {concept}.",
        examples=examples or [],
        code=[CodeBlock(source=code, language="python")] if code else [],
        section="Chunk section",
        spans=[VideoSpan(video_id=VID, start_s=start, end_s=start + 20)],
        evidence=[Evidence(video_id=VID, segment_ids=segs or [f"s{int(start)}"])],
    )


# The concept names the first real run produced for this video (devlog, Milestone B).
REAL_RUN = [
    unit("u1", "`zip()` Function", 26, code=ZIP_CODE),
    unit("u2", "`enumerate()` Function", 26, code=ENUM_CODE, examples=["enumerate(names)"]),
    unit("u3", "Basic List Iteration", 33),
    unit("u4", "Manual Indexing for Parallel Iteration", 59),
    unit("u5", "enumerate() function (Direct Unpacking)", 209, code=ENUM_CODE),
    unit("u6", "enumerate() function (Tuple Output)", 234, examples=["(0, 'tom')"]),
    unit("u7", "zip() function", 279, code=ZIP_CODE),
    unit("u8", "Manual Indexing for Parallel Iteration", 279),
]


def ids(groups: list[list[int]]) -> list[list[str]]:
    return [[REAL_RUN[i].id for i in g] for g in groups]


def test_core_concept_drops_generic_words_and_punctuation() -> None:
    assert core_concept("`enumerate()` Function (Direct Unpacking)") == "enumerate direct unpacking"
    assert core_concept("Basic List Iteration") == "list iteration"


def test_candidate_groups_on_the_real_duplicates() -> None:
    assert ids(candidate_groups(REAL_RUN)) == [
        ["u1", "u7"],
        ["u2", "u5", "u6"],
        ["u3"],
        ["u4", "u8"],
    ]


def test_merge_spans_joins_overlaps_and_keeps_separate_moments() -> None:
    spans = [
        VideoSpan(video_id=VID, start_s=209, end_s=229),
        VideoSpan(video_id=VID, start_s=26, end_s=46),
        VideoSpan(video_id=VID, start_s=230, end_s=254),  # within 5 s of the previous
    ]
    assert [(s.start_s, s.end_s) for s in merge_spans(spans)] == [(26, 46), (209, 254)]


def note(sources: list[str], concept: str, section: str) -> MergedNote:
    return MergedNote(
        source_ids=sources,
        concept=concept,
        explanation=f"Combined: {concept}.",
        section=section,
    )


async def merge(units: list[KnowledgeUnit], output: MergeOutput) -> tuple[ExtractedUnits, FakeLLM]:
    fake = FakeLLM({"merge": output})
    stage = MergeStage(
        LLMGateway(fake.factory, default_model="gemini-2.5-flash", retry_backoff=False),
        model="gemini-2.5-flash",
    )
    inp = ExtractedUnits(units=units, dropped=["extract: earlier note"])
    return await stage.run(inp, RunContext(job_id=uuid4(), video=VIDEO)), fake


REAL_ANSWER = MergeOutput(
    groups=[
        GroupResult(group_id="g1", notes=[note(["u1", "u7"], "zip()", "Iteration tools")]),
        GroupResult(
            group_id="g2", notes=[note(["u2", "u5", "u6"], "enumerate()", "Iteration tools")]
        ),
        GroupResult(group_id="g3", notes=[note(["u3"], "ignored", "Looping basics")]),
        GroupResult(group_id="g4", notes=[note(["u4", "u8"], "Manual indexing", "Looping basics")]),
    ]
)


async def test_duplicates_merged_with_every_timestamp_kept() -> None:
    out, fake = await merge(REAL_RUN, REAL_ANSWER)

    assert len(fake.calls) == 1  # one call per video
    by_id = {u.id: u for u in out.units}
    assert sorted(by_id) == ["u1", "u2", "u3", "u4"]

    enum = by_id["u2"]
    assert enum.concept == "enumerate()"
    assert enum.explanation == "Combined: enumerate()."
    # Every moment kept; the longest (where it is taught) first.
    assert [(s.start_s, s.end_s) for s in enum.spans] == [(209, 254), (26, 46)]
    assert [c.source for c in enum.code] == [ENUM_CODE]  # identical code from u2 + u5, once
    assert enum.examples == ["enumerate(names)", "(0, 'tom')"]
    assert enum.evidence[0].segment_ids == ["s26", "s209", "s234"]

    assert {u.section for u in out.units} == {"Iteration tools", "Looping basics"}
    assert by_id["u3"].explanation == "About Basic List Iteration."  # singleton text kept
    assert by_id["u3"].concept == "Basic List Iteration"
    assert out.dropped[0] == "extract: earlier note"
    assert any("u2, u5, u6 -> u2" in d for d in out.dropped)
    # ordered by primary moment: u2 (enumerate) is now taught at 209 s, after u4 (59 s)
    assert [u.id for u in out.units] == ["u1", "u3", "u4", "u2"]


async def test_llm_may_split_a_candidate_group() -> None:
    units = [unit("a", "Iteration", 10), unit("b", "List Iteration", 90)]
    assert ids_of(candidate_groups(units), units) == [["a", "b"]]  # subset → candidate pair
    answer = MergeOutput(
        groups=[
            GroupResult(
                group_id="g1",
                notes=[note(["a"], "Iteration", "Loops"), note(["b"], "List iteration", "Loops")],
            )
        ]
    )
    out, _ = await merge(units, answer)
    assert [u.id for u in out.units] == ["a", "b"]


@pytest.mark.parametrize(
    "notes",
    [
        [note(["u2", "u5"], "enumerate()", "S")],  # u6 missing
        [note(["u2", "u5", "u6"], "x", "S"), note(["u6"], "y", "S")],  # u6 twice
        [note(["u2", "u5", "u6", "u1"], "x", "S")],  # u1 from another group
    ],
)
async def test_bad_partition_keeps_the_group_unmerged(notes: list[MergedNote]) -> None:
    answer = MergeOutput(groups=[GroupResult(group_id="g2", notes=notes)])
    out, _ = await merge(REAL_RUN, answer)
    assert len(out.units) == len(REAL_RUN)  # nothing lost, nothing duplicated
    assert any("g2 kept as is" in d for d in out.dropped)
    assert any("g1 kept as is: no answer" in d for d in out.dropped)


async def test_fewer_than_two_units_makes_no_call() -> None:
    out, fake = await merge([REAL_RUN[0]], MergeOutput(groups=[]))
    assert fake.calls == []
    assert out.units == [REAL_RUN[0]]


def ids_of(groups: list[list[int]], units: list[KnowledgeUnit]) -> list[list[str]]:
    return [[units[i].id for i in g] for g in groups]
