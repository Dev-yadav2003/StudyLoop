from decimal import Decimal
from uuid import uuid4

import pytest

from lecnotes.adapters.cache.memory import InMemoryStageCache
from lecnotes.adapters.perception.fake import FakePerceiver
from lecnotes.domain import (
    BoardState,
    BoardType,
    LLMUsage,
    PerceptionResult,
    PerceptionRun,
    TranscriptSegment,
    VideoMeta,
)
from lecnotes.errors import PerceptionError
from lecnotes.observability.metrics import InMemoryStageRunSink
from lecnotes.pipeline.context import RunContext
from lecnotes.pipeline.orchestrator import Orchestrator, StageFailed
from lecnotes.pipeline.stages.perceive import PerceiveStage

VIDEO = VideoMeta(video_id="dQw4w9WgXcQ", title="Iteration", duration_s=754)
RESULT = PerceptionResult(
    video_id=VIDEO.video_id,
    segments=[TranscriptSegment(id="s0001", start_s=1, end_s=6, text="Today: iteration.")],
    board_states=[
        BoardState(id="b0001", t_first_seen=0, t_last_seen=41, type=BoardType.SLIDE, text="Iter")
    ],
)
USAGE = LLMUsage(
    prompt_id="perceive",
    prompt_version=1,
    model="gemini-2.5-flash",
    tokens_in=51_234,
    tokens_out=3_210,
    cost_usd=Decimal("0.023395"),
    latency_ms=61_000,
)


def ctx() -> RunContext:
    return RunContext(job_id=uuid4(), video=VIDEO)


async def test_perceive_runs_once_then_hits_cache() -> None:
    perceiver = FakePerceiver({VIDEO.video_id: PerceptionRun(result=RESULT, usage=[USAGE])})
    sink = InMemoryStageRunSink()
    orch = Orchestrator(InMemoryStageCache(), sink)
    stage = PerceiveStage(perceiver)

    first = await orch.run_stage(stage, VIDEO, ctx())
    second = await orch.run_stage(stage, VIDEO, ctx())

    assert isinstance(first, PerceptionRun)
    assert first.result == RESULT
    assert second == first
    assert perceiver.calls == [VIDEO.video_id]
    miss, hit = sink.runs
    assert (miss.tokens_in, miss.cost_usd, miss.prompt_version, miss.model) == (
        51_234,
        Decimal("0.023395"),
        1,
        "gemini-2.5-flash",
    )
    assert hit.cache_hit
    assert hit.cost_usd == 0


async def test_changing_the_perceive_prompt_version_reperceives() -> None:
    cache = InMemoryStageCache()
    orch = Orchestrator(cache, InMemoryStageRunSink())
    v1 = FakePerceiver({VIDEO.video_id: RESULT}, prompt_version=1)
    v2 = FakePerceiver({VIDEO.video_id: RESULT}, prompt_version=2)

    await orch.run_stage(PerceiveStage(v1), VIDEO, ctx())
    await orch.run_stage(PerceiveStage(v1), VIDEO, ctx())
    await orch.run_stage(PerceiveStage(v2), VIDEO, ctx())

    assert (len(v1.calls), len(v2.calls)) == (1, 1)


async def test_perception_error_propagates_typed() -> None:
    orch = Orchestrator(InMemoryStageCache(), InMemoryStageRunSink())
    with pytest.raises(StageFailed) as exc_info:
        await orch.run_stage(PerceiveStage(FakePerceiver({})), VIDEO, ctx())
    assert isinstance(exc_info.value.cause, PerceptionError)


def test_video_meta_ref() -> None:
    assert VIDEO.ref.video_id == VIDEO.video_id
    assert VIDEO.url == f"https://www.youtube.com/watch?v={VIDEO.video_id}"
