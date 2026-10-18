from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

from lecnotes.domain import LLMUsage, StageRun, StageRunStatus
from lecnotes.observability.metrics import InMemoryStageRunSink, usage_totals
from lecnotes.ports.metrics import StageRunSink


def usage(model: str, tokens_in: int, tokens_out: int, cost: str) -> LLMUsage:
    return LLMUsage(
        prompt_id="extract",
        prompt_version=1,
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=Decimal(cost),
        latency_ms=100,
    )


def test_usage_totals_sums_exactly() -> None:
    totals = usage_totals(
        [
            usage("gemini-2.5-flash", 1000, 200, "0.0008"),
            usage("gemini-2.5-flash", 3000, 100, "0.00115"),
            usage("gemini-3.1-flash-lite", 500, 50, "0.00007"),
        ]
    )
    assert (totals.tokens_in, totals.tokens_out) == (4500, 350)
    assert totals.cost_usd == Decimal("0.00202")
    assert totals.model == "gemini-2.5-flash+gemini-3.1-flash-lite"


def test_usage_totals_of_a_stage_without_llm_calls() -> None:
    totals = usage_totals([])
    assert (totals.tokens_in, totals.tokens_out, totals.cost_usd, totals.model) == (
        0,
        0,
        Decimal(0),
        None,
    )


async def test_in_memory_sink_satisfies_port() -> None:
    sink: StageRunSink = InMemoryStageRunSink()
    run = StageRun(
        job_id=uuid4(),
        stage="validate",
        stage_version=1,
        status=StageRunStatus.OK,
        latency_ms=3,
        started_at=datetime.now(UTC),
    )
    await sink.record(run)
    assert isinstance(sink, InMemoryStageRunSink)
    assert sink.runs == [run]
