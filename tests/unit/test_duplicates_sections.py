from evals.metrics.duplicates import DuplicateGroup, DuplicateJudgement, duplicate_rate
from evals.metrics.sections import section_fragmentation
from lecnotes.adapters.llm.fake import FakeLLM
from lecnotes.domain import Evidence, KnowledgeUnit, VideoSpan
from lecnotes.llm.gateway import LLMGateway

VID = "dQw4w9WgXcQ"


def unit(id: str, start: float, section: str = "S") -> KnowledgeUnit:
    return KnowledgeUnit(
        id=id,
        concept=f"Concept {id}",
        explanation=f"About {id}.",
        section=section,
        spans=[VideoSpan(video_id=VID, start_s=start, end_s=start + 10)],
        evidence=[Evidence(video_id=VID, segment_ids=["s1"])],
    )


async def test_duplicate_rate_counts_k_minus_one_per_group_and_ignores_bad_ids() -> None:
    judgement = DuplicateJudgement(
        groups=[
            DuplicateGroup(unit_ids=["a", "c", "e"], concept="success criteria"),
            DuplicateGroup(unit_ids=["c", "d"], concept="reused id"),  # c already used → 1 id
            DuplicateGroup(unit_ids=["b", "ghost"], concept="unknown id"),  # → 1 id
        ]
    )
    fake = FakeLLM({"judge/duplicates": judgement})
    units = [unit(i, n * 10) for n, i in enumerate("abcde")]

    gateway = LLMGateway(fake.factory, default_model="gemini-2.5-flash")
    score = await duplicate_rate(gateway, units, samples=1)

    assert score.groups == [["a", "c", "e"]]
    assert (score.redundant, score.rate) == (2, 2 / 5)


async def test_duplicate_rate_needs_two_units() -> None:
    fake = FakeLLM({})
    gateway = LLMGateway(fake.factory, default_model="gemini-2.5-flash")
    assert (await duplicate_rate(gateway, [unit("a", 0)])).redundant == 0
    assert fake.calls == []


def test_contiguous_sections_are_not_fragmented() -> None:
    units = [unit("a", 0, "Intro"), unit("b", 10, "Intro"), unit("c", 20, "Workflow")]
    assert section_fragmentation(units).fragmentation == 0.0


def test_returning_to_a_section_is_fragmentation() -> None:
    # read in time order: Intro, Example, Intro, Workflow → 4 runs, 3 sections
    units = [
        unit("d", 30, "Workflow"),
        unit("a", 0, "Intro"),
        unit("b", 10, "Example"),
        unit("c", 20, "Intro"),
    ]
    score = section_fragmentation(units)
    assert (score.runs, score.sections) == (4, 3)
    assert score.fragmentation == 1 / 3


async def test_duplicate_rate_reports_the_median_of_noisy_judge_runs() -> None:
    units = [unit(i, n * 10) for n, i in enumerate("abcd")]
    none = DuplicateJudgement(groups=[])
    one = DuplicateJudgement(groups=[DuplicateGroup(unit_ids=["a", "b"], concept="x")])
    wild = DuplicateJudgement(groups=[DuplicateGroup(unit_ids=["a", "b", "c", "d"], concept="y")])
    fake = FakeLLM({"judge/duplicates": [wild, one, none]})
    gateway = LLMGateway(fake.factory, default_model="gemini-2.5-flash")

    score = await duplicate_rate(gateway, units)

    assert len(fake.calls) == 3
    assert (score.redundant, score.groups) == (1, [["a", "b"]])
    assert len(score.usage) == 3
