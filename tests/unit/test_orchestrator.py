from decimal import Decimal
from typing import Any
from uuid import uuid4

import pytest
from pydantic import BaseModel

from lecnotes.adapters.cache.memory import InMemoryStageCache
from lecnotes.domain import LLMUsage, StageRunStatus, VideoMeta
from lecnotes.errors import QuotaExceeded
from lecnotes.observability.metrics import InMemoryStageRunSink
from lecnotes.observability.tracing import traced_job
from lecnotes.pipeline.context import RunContext
from lecnotes.pipeline.orchestrator import Orchestrator, StageFailed, check_chain
from lecnotes.pipeline.stage import Stage
from tests.support import SpanRecorder


class Num(BaseModel):
    value: int


class Text(BaseModel):
    text: str


class Arith(Stage[Num, Num]):
    input_model = Num
    output_model = Num

    def __init__(self, fn: Any, factor: int = 1) -> None:
        self.fn = fn
        self.factor = factor

    def config(self, ctx: RunContext) -> dict[str, Any]:
        return {"factor": self.factor}

    async def run(self, inp: Num, ctx: RunContext) -> Num:
        CALLS[self.name] = CALLS.get(self.name, 0) + 1
        return Num(value=self.fn(inp.value) * self.factor)


CALLS: dict[str, int] = {}


class AddOne(Arith):
    name = "add_one"
    version = 1

    def __init__(self, factor: int = 1) -> None:
        super().__init__(lambda v: v + 1, factor)


class Double(Arith):
    name = "double"
    version = 1

    def __init__(self) -> None:
        super().__init__(lambda v: v * 2)


class Square(Arith):
    name = "square"
    version = 1

    def __init__(self) -> None:
        super().__init__(lambda v: v * v)


class CostlyDouble(Double):
    """Reports LLM usage like an LLM stage would."""

    name = "costly"

    async def run(self, inp: Num, ctx: RunContext) -> Num:
        ctx.add_usage(
            LLMUsage(
                prompt_id="p",
                prompt_version=3,
                model="gemini-2.5-flash",
                tokens_in=1000,
                tokens_out=200,
                cost_usd=Decimal("0.0008"),
                latency_ms=5,
            )
        )
        return await super().run(inp, ctx)


class Boom(Arith):
    name = "boom"
    version = 1

    def __init__(self) -> None:
        super().__init__(lambda v: v)

    async def run(self, inp: Num, ctx: RunContext) -> Num:
        raise QuotaExceeded("429 from the model")


class ToText(Stage[Num, Text]):
    name = "to_text"
    version = 1
    input_model = Num
    output_model = Text

    async def run(self, inp: Num, ctx: RunContext) -> Text:
        return Text(text=str(inp.value))


@pytest.fixture(autouse=True)
def reset_calls() -> None:
    CALLS.clear()


@pytest.fixture
def cache() -> InMemoryStageCache:
    return InMemoryStageCache()


@pytest.fixture
def sink() -> InMemoryStageRunSink:
    return InMemoryStageRunSink()


@pytest.fixture
def orch(cache: InMemoryStageCache, sink: InMemoryStageRunSink) -> Orchestrator:
    return Orchestrator(cache, sink)


def ctx() -> RunContext:
    return RunContext(job_id=uuid4())


CHAIN = [AddOne(), Double(), Square()]


async def test_first_run_executes_all_stages(
    orch: Orchestrator, sink: InMemoryStageRunSink
) -> None:
    out = await orch.run_chain(CHAIN, Num(value=2), ctx())

    assert out == Num(value=36)  # ((2+1)*2)^2
    assert CALLS == {"add_one": 1, "double": 1, "square": 1}
    assert [(r.stage, r.cache_hit, r.status) for r in sink.runs] == [
        ("add_one", False, StageRunStatus.OK),
        ("double", False, StageRunStatus.OK),
        ("square", False, StageRunStatus.OK),
    ]


async def test_second_run_is_all_cache_hits(orch: Orchestrator, sink: InMemoryStageRunSink) -> None:
    await orch.run_chain(CHAIN, Num(value=2), ctx())
    CALLS.clear()
    sink.runs.clear()

    out = await orch.run_chain(CHAIN, Num(value=2), ctx())

    assert out == Num(value=36)
    assert CALLS == {}
    assert all(r.cache_hit for r in sink.runs)
    assert len(sink.runs) == 3


async def test_rerun_from_forces_that_stage_and_downstream_only(orch: Orchestrator) -> None:
    await orch.run_chain(CHAIN, Num(value=2), ctx())
    CALLS.clear()

    out = await orch.run_chain(CHAIN, Num(value=2), ctx(), rerun_from="double")

    assert out == Num(value=36)
    assert CALLS == {"double": 1, "square": 1}


async def test_exception_is_recorded_and_stops_the_chain(
    orch: Orchestrator, sink: InMemoryStageRunSink
) -> None:
    chain = [AddOne(), Boom(), Square()]
    with pytest.raises(StageFailed) as exc_info:
        await orch.run_chain(chain, Num(value=2), ctx())

    failed = exc_info.value
    assert failed.stage == "boom"
    assert isinstance(failed.cause, QuotaExceeded)
    assert failed.retryable is True  # inherited from the cause
    assert failed.user_message == QuotaExceeded.user_message
    assert CALLS == {"add_one": 1}  # square never ran
    assert [(r.stage, r.status, r.error_type) for r in sink.runs] == [
        ("add_one", StageRunStatus.OK, None),
        ("boom", StageRunStatus.FAILED, "QuotaExceeded"),
    ]


async def test_failed_stage_is_not_cached(orch: Orchestrator) -> None:
    with pytest.raises(StageFailed):
        await orch.run_chain([AddOne(), Boom()], Num(value=2), ctx())
    CALLS.clear()
    with pytest.raises(StageFailed):
        await orch.run_chain([AddOne(), Boom()], Num(value=2), ctx())
    assert CALLS == {}  # add_one hit the cache; boom raised again (it doesn't count calls)


async def test_stage_config_and_video_are_part_of_the_key(orch: Orchestrator) -> None:
    await orch.run_chain([AddOne(factor=1)], Num(value=2), ctx())
    await orch.run_chain([AddOne(factor=10)], Num(value=2), ctx())  # relevant config changed
    assert CALLS == {"add_one": 2}

    video = VideoMeta(video_id="dQw4w9WgXcQ", title="L1", duration_s=60)
    other = VideoMeta(video_id="aaaaaaaaaaa", title="L2", duration_s=60)
    await orch.run_chain([AddOne()], Num(value=2), RunContext(job_id=uuid4(), video=video))
    await orch.run_chain([AddOne()], Num(value=2), RunContext(job_id=uuid4(), video=other))
    assert CALLS == {"add_one": 4}


async def test_usage_becomes_stage_metrics_and_cache_hits_cost_nothing(
    orch: Orchestrator, sink: InMemoryStageRunSink
) -> None:
    await orch.run_chain([CostlyDouble()], Num(value=2), ctx())
    await orch.run_chain([CostlyDouble()], Num(value=2), ctx())

    miss, hit = sink.runs
    assert (miss.tokens_in, miss.tokens_out, miss.cost_usd, miss.model) == (
        1000,
        200,
        Decimal("0.0008"),
        "gemini-2.5-flash",
    )
    assert hit.cache_hit
    assert (hit.tokens_in, hit.cost_usd) == (0, 0)


def test_chain_type_mismatch_fails_before_running() -> None:
    with pytest.raises(TypeError, match="square expects Num, but receives Text"):
        check_chain([ToText(), Square()], Num)


async def test_bad_rerun_from_and_duplicate_names_rejected(orch: Orchestrator) -> None:
    with pytest.raises(ValueError, match="rerun_from='nope'"):
        await orch.run_chain(CHAIN, Num(value=1), ctx(), rerun_from="nope")
    with pytest.raises(ValueError, match="duplicate"):
        await orch.run_chain([AddOne(), AddOne()], Num(value=1), ctx())
    assert CALLS == {}


async def test_trace_has_one_span_per_stage(
    orch: Orchestrator, sink: InMemoryStageRunSink, span_recorder: SpanRecorder
) -> None:
    with traced_job("job-1"):
        await orch.run_chain(CHAIN, Num(value=2), ctx())

    spans = span_recorder.by_name()
    job = spans["job"]
    assert job.context is not None
    for name in ("add_one", "double", "square"):
        span = spans[f"stage:{name}"]
        assert span.parent is not None
        assert span.parent.span_id == job.context.span_id
    trace_ids = {r.trace_id for r in sink.runs}
    assert len(trace_ids) == 1
    assert None not in trace_ids
