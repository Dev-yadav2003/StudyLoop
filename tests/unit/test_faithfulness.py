import pytest

from evals.metrics.faithfulness import Claim, FaithfulnessJudgement, UnitClaims, faithfulness
from lecnotes.adapters.llm.fake import FakeLLM
from lecnotes.domain import (
    Evidence,
    KnowledgeUnit,
    PerceptionResult,
    TranscriptSegment,
    VideoSpan,
)
from lecnotes.llm.gateway import LLMGateway

VID = "dQw4w9WgXcQ"
EVIDENCE = PerceptionResult(
    video_id=VID,
    segments=[TranscriptSegment(id="s1", start_s=0, end_s=5, text="zip pairs two lists")],
    board_states=[],
)


def unit(id: str) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=id,
        concept="zip",
        explanation="zip pairs two lists. It is the fastest way.",
        spans=[VideoSpan(video_id=VID, start_s=0, end_s=5)],
        evidence=[Evidence(video_id=VID, segment_ids=["s1"])],
    )


async def test_claim_level_score_and_reporting() -> None:
    judgement = FaithfulnessJudgement(
        units=[
            UnitClaims(
                unit_id="u1",
                claims=[
                    Claim(claim="zip pairs two lists", supported=True),
                    Claim(claim="zip is the fastest way", supported=False),
                ],
            ),
            UnitClaims(unit_id="u2", claims=[Claim(claim="zip pairs lists", supported=True)]),
        ]
    )
    fake = FakeLLM({"judge/faithfulness": judgement})
    gateway = LLMGateway(fake.factory, default_model="gemini-2.5-flash", retry_backoff=False)

    score = await faithfulness(
        gateway, [unit("u1"), unit("u2"), unit("u3")], EVIDENCE, model="gemini-3-flash-preview"
    )

    assert (score.supported, score.total) == (2, 3)
    assert score.score == pytest.approx(2 / 3)
    assert score.unsupported_claims == {"u1": ["zip is the fastest way"]}
    assert score.ungraded == ["u3"]
    assert fake.calls[0].model == "gemini-3-flash-preview"  # judge model is configurable
    assert "[s1] zip pairs two lists" in str(fake.calls[0].messages[-1].content)


async def test_no_units_no_score() -> None:
    fake = FakeLLM({})
    gateway = LLMGateway(fake.factory, default_model="gemini-2.5-flash")
    score = await faithfulness(gateway, [], EVIDENCE)
    assert score.score is None
    assert fake.calls == []
