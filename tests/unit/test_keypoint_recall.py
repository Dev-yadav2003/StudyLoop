import pytest

from evals.golden_set import Keypoint
from evals.metrics.keypoint_recall import (
    KeypointJudgement,
    KeypointScore,
    KeypointVerdict,
    keypoint_recall,
)
from lecnotes.adapters.llm.fake import FakeLLM
from lecnotes.llm.gateway import LLMGateway

NOTES = """# Python iteration

## enumerate()
enumerate() yields (index, item) pairs. The index starts at zero unless you pass start.
"""

KEYPOINTS = [
    Keypoint(id="k1", text="enumerate yields (index, item) pairs"),
    Keypoint(id="k2", text="enumerate takes an optional start argument"),
    Keypoint(id="k3", text="zip stops at the shortest iterable"),
    Keypoint(id="k4", text="range is lazy"),
]


def judge(*verdicts: KeypointVerdict) -> FakeLLM:
    return FakeLLM({"judge/keypoint_recall": KeypointJudgement(verdicts=list(verdicts))})


async def run(fake: FakeLLM) -> KeypointScore:
    gateway = LLMGateway(fake.factory, default_model="gemini-2.5-flash", retry_backoff=False)
    return await keypoint_recall(gateway, NOTES, KEYPOINTS)


async def test_recall_with_leniency_guards() -> None:
    fake = judge(
        KeypointVerdict(id="k1", covered=True, quote="enumerate() yields (index, item) pairs."),
        KeypointVerdict(
            id="k2", covered=True, quote="The index starts at zero unless you pass start."
        ),
        # Lenient judge: claims coverage with a quote that is not in the notes.
        KeypointVerdict(id="k3", covered=True, quote="zip() stops at the shortest input."),
        # k4: not graded at all.
    )
    score = await run(fake)

    assert score.covered == ["k1", "k2"]
    assert score.missed == ["k3", "k4"]
    assert score.unsupported == ["k3"]
    assert score.ungraded == ["k4"]
    assert score.recall == pytest.approx(0.5)
    assert score.usage is not None


async def test_covered_without_quote_is_not_covered() -> None:
    score = await run(judge(KeypointVerdict(id="k1", covered=True, quote="")))
    assert "k1" in score.missed


async def test_prompt_lists_keypoints_and_notes() -> None:
    fake = judge()
    await run(fake)
    user = str(fake.calls[0].messages[-1].content)
    assert "[k3] zip stops at the shortest iterable" in user
    assert "## enumerate()" in user
