from uuid import uuid4

import pytest

from lecnotes.adapters.llm.fake import FakeLLM
from lecnotes.domain import (
    BoardState,
    BoardType,
    Evidence,
    ExtractedUnits,
    KnowledgeUnit,
    PerceptionResult,
    TranscriptSegment,
    VideoMeta,
    VideoSpan,
)
from lecnotes.llm.gateway import LLMGateway
from lecnotes.pipeline.context import RunContext
from lecnotes.pipeline.stages.verify import UnitVerdict, VerifyOutput, VerifyStage

VID = "dQw4w9WgXcQ"
VIDEO = VideoMeta(video_id=VID, title="Iteration", duration_s=600)
EVIDENCE = PerceptionResult(
    video_id=VID,
    segments=[
        TranscriptSegment(id="s1", start_s=10, end_s=15, text="zip pairs up two lists"),
        TranscriptSegment(id="s2", start_s=20, end_s=25, text="enumerate gives the index"),
    ],
    board_states=[
        BoardState(id="b1", t_first_seen=5, t_last_seen=60, type=BoardType.CODE, text="zip(a, b)")
    ],
)


def unit(id: str, concept: str, seg: str, examples: list[str] | None = None) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=id,
        concept=concept,
        explanation=f"Original explanation of {concept}.",
        examples=examples or [],
        spans=[VideoSpan(video_id=VID, start_s=10, end_s=15)],
        evidence=[Evidence(video_id=VID, segment_ids=[seg], board_state_ids=["b1"])],
    )


def verdict(unit_id: str, kind: str, **extra: object) -> UnitVerdict:
    fields: dict[str, object] = {
        "unit_id": unit_id,
        "verdict": kind,
        "revised_explanation": "",
        "supported_examples": [],
        "unsupported_claims": [],
    }
    return UnitVerdict.model_validate(fields | extra)


async def verify(
    units: list[KnowledgeUnit], *results: UnitVerdict, batch_size: int = 6
) -> tuple[ExtractedUnits, FakeLLM]:
    fake = FakeLLM({"verify": VerifyOutput(results=list(results))})
    stage = VerifyStage(
        LLMGateway(fake.factory, default_model="gemini-2.5-flash", retry_backoff=False),
        model="gemini-2.5-flash",
        batch_size=batch_size,
    )
    inp = ExtractedUnits(units=units, dropped=["extract: earlier removal"], evidence=EVIDENCE)
    return await stage.run(inp, RunContext(job_id=uuid4(), video=VIDEO)), fake


async def test_each_verdict_is_applied() -> None:
    units = [
        unit("u1", "zip", "s1", examples=["zip([1], [2])", "zip is faster than loops"]),
        unit("u2", "enumerate", "s2", examples=["enumerate(xs)"]),
        unit("u3", "invented", "s1"),
        unit("u4", "forgotten", "s2"),
    ]
    out, _ = await verify(
        units,
        verdict("u1", "supported", supported_examples=["zip([1], [2])"]),
        verdict(
            "u2",
            "revise",
            revised_explanation="enumerate gives the index.",
            supported_examples=["enumerate(xs)"],
            unsupported_claims=["it is the most professional way"],
        ),
        verdict("u3", "unsupported", unsupported_claims=["not in the lecture"]),
        # u4: the verifier forgot it
    )

    by_id = {u.id: u for u in out.units}
    assert list(by_id) == ["u1", "u2", "u4"]
    assert by_id["u1"].examples == ["zip([1], [2])"]  # unsupported example removed
    assert by_id["u2"].explanation == "enumerate gives the index."
    assert by_id["u2"].examples == ["enumerate(xs)"]
    assert by_id["u4"].explanation == "Original explanation of forgotten."  # kept untouched
    assert out.verify is not None
    assert out.verify.model_dump() == {
        "checked": 4,
        "supported": 1,
        "revised": 1,
        "dropped": 1,
        "unverified": 1,
    }
    assert out.verify.dropped_rate == pytest.approx(0.25)
    assert out.dropped[0] == "extract: earlier removal"  # history preserved
    assert any("u3" in d and "not in the lecture" in d for d in out.dropped)
    assert any("u2 revised" in d and "most professional" in d for d in out.dropped)
    assert out.evidence == EVIDENCE


async def test_revise_with_nothing_left_drops_the_unit() -> None:
    out, _ = await verify([unit("u1", "zip", "s1")], verdict("u1", "revise"))
    assert out.units == []
    assert out.verify is not None
    assert out.verify.dropped == 1


async def test_units_are_batched() -> None:
    units = [unit(f"u{i}", f"c{i}", "s1") for i in range(7)]
    _, fake = await verify(units, batch_size=3)
    assert len(fake.calls) == 3  # 3 + 3 + 1


async def test_prompt_shows_only_each_units_own_evidence() -> None:
    _, fake = await verify([unit("u1", "zip", "s1")])
    prompt = str(fake.calls[0].messages[-1].content)
    assert "===== NOTE u1 =====" in prompt
    assert "[s1] zip pairs up two lists" in prompt
    assert "[b1] zip(a, b)" in prompt
    assert "enumerate gives the index" not in prompt  # s2 is not cited by u1


async def test_requires_evidence() -> None:
    fake = FakeLLM({})
    stage = VerifyStage(
        LLMGateway(fake.factory, default_model="gemini-2.5-flash"), model="gemini-2.5-flash"
    )
    with pytest.raises(ValueError, match="evidence"):
        await stage.run(ExtractedUnits(units=[]), RunContext(job_id=uuid4(), video=VIDEO))
