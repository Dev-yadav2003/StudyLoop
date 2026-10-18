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
    MergedNote,
    MergeOutput,
    MergeStage,
    apply_answer,
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


def test_merge_spans_joins_overlaps_and_keeps_separate_moments() -> None:
    spans = [
        VideoSpan(video_id=VID, start_s=209, end_s=229),
        VideoSpan(video_id=VID, start_s=26, end_s=46),
        VideoSpan(video_id=VID, start_s=230, end_s=254),  # within 5 s of the previous
    ]
    assert [(s.start_s, s.end_s) for s in merge_spans(spans)] == [(26, 46), (209, 254)]


def note(sources: list[str], concept: str, section: str) -> MergedNote:
    merged = len(sources) > 1
    return MergedNote(
        source_ids=sources,
        concept=concept if merged else "",
        explanation=f"Combined: {concept}." if merged else "",
        section=section,
    )


SECTIONS = ["Looping basics", "Iteration tools"]


async def merge(units: list[KnowledgeUnit], output: MergeOutput) -> tuple[ExtractedUnits, FakeLLM]:
    fake = FakeLLM({"merge": output})
    stage = MergeStage(
        LLMGateway(fake.factory, default_model="gemini-2.5-flash", retry_backoff=False),
        model="gemini-2.5-flash",
    )
    inp = ExtractedUnits(units=units, dropped=["extract: earlier note"])
    return await stage.run(inp, RunContext(job_id=uuid4(), video=VIDEO)), fake


REAL_ANSWER = MergeOutput(
    sections=SECTIONS,
    notes=[
        note(["u1", "u7"], "zip()", "Iteration tools"),
        note(["u2", "u5", "u6"], "enumerate()", "Iteration tools"),
        note(["u3"], "", "Looping basics"),
        note(["u4", "u8"], "Manual indexing", "Looping basics"),
    ],
)


async def test_duplicates_merged_with_every_timestamp_kept() -> None:
    out, fake = await merge(REAL_RUN, REAL_ANSWER)

    assert len(fake.calls) == 1  # one call per video
    prompt = "\n".join(str(m.content) for m in fake.calls[0].messages)
    # the whole lecture in one view, with times
    assert "[u5] (03:29) enumerate() function (Direct Unpacking)" in prompt
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
    assert by_id["u3"].explanation == "About Basic List Iteration."  # single note text kept
    assert by_id["u3"].concept == "Basic List Iteration"
    assert out.dropped[0] == "extract: earlier note"
    assert any("u2, u5, u6 -> u2" in d for d in out.dropped)
    # ordered by primary moment: u2 (enumerate) is now taught at 209 s, after u4 (59 s)
    assert [u.id for u in out.units] == ["u1", "u3", "u4", "u2"]


def test_differently_named_duplicates_can_merge() -> None:
    """The case v1/v2 missed: names too different for fuzzy candidate groups."""
    units = [
        unit("a", "Define Success Criteria", 262),
        unit("b", "Running the Model", 420),
        unit("c", "Defining Task, Target, and Success Criteria", 546),
    ]
    answer = MergeOutput(
        sections=["Workflow"],
        notes=[note(["a", "c"], "Success criteria", "Workflow"), note(["b"], "", "Workflow")],
    )
    out, _ = apply_answer(units, answer)
    assert [(u.id, u.concept) for u in out] == [
        ("a", "Success criteria"),
        ("b", "Running the Model"),
    ]
    assert [s.start_s for s in out[0].spans] == [262, 546]


def test_unknown_reused_and_missing_ids_lose_nothing() -> None:
    answer = MergeOutput(
        sections=SECTIONS,
        notes=[
            note(["u2", "u5", "nope"], "enumerate()", "Iteration tools"),
            note(["u5", "u6"], "again", "Iteration tools"),  # u5 reused → only u6 left
            note(["ghost"], "", "Looping basics"),  # nothing valid → skipped
        ],
    )
    out, log = apply_answer(REAL_RUN, answer)
    ids = sorted(i for u in out for i in [u.id])
    assert ids == ["u1", "u2", "u3", "u4", "u6", "u7", "u8"]  # u5 merged into u2, rest kept
    assert "merge: note 1 ignored unknown/reused ids ['nope']" in log
    assert "merge: note 2 ignored unknown/reused ids ['u5']" in log
    assert any("not in answer" in line for line in log)
    # left-out notes join the section of the nearest placed note, not their old one
    assert {u.section for u in out} <= set(SECTIONS)


def test_sections_snap_to_the_declared_list() -> None:
    answer = MergeOutput(
        sections=["LLM Application Evaluation Workflow", "Monitoring"],
        notes=[
            note(["u1"], "", "LLM Evaluation Workflow"),  # near-duplicate name
            note(["u2"], "", "LLM Application Evaluation Workflow"),
        ],
    )
    out, _ = apply_answer(REAL_RUN[:2], answer)
    assert {u.section for u in out} == {"LLM Application Evaluation Workflow"}


def test_prompt_v1_is_rejected() -> None:
    fake = FakeLLM({})
    gateway = LLMGateway(fake.factory, default_model="gemini-2.5-flash")
    with pytest.raises(ValueError, match="prompt v2"):
        MergeStage(gateway, model="gemini-2.5-flash", prompt_version=1)


async def test_fewer_than_two_units_makes_no_call() -> None:
    out, fake = await merge([REAL_RUN[0]], MergeOutput(sections=[], notes=[]))
    assert fake.calls == []
    assert out.units == [REAL_RUN[0]]
