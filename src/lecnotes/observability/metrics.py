"""Stage metrics: aggregate LLM usage into a StageRun, and an in-memory sink.

The Postgres sink and the percentile queries live in storage/repositories/stage_runs.py.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal

from lecnotes.domain import LLMUsage, StageRun
from lecnotes.ports.metrics import StageRunSink


@dataclass(frozen=True)
class UsageTotals:
    tokens_in: int
    tokens_out: int
    cost_usd: Decimal
    model: str | None
    """The model that did the work; 'a+b' if a stage used several (e.g. after a fallback)."""


def usage_totals(usages: Iterable[LLMUsage]) -> UsageTotals:
    items = list(usages)
    models = list(dict.fromkeys(u.model for u in items))
    return UsageTotals(
        tokens_in=sum(u.tokens_in for u in items),
        tokens_out=sum(u.tokens_out for u in items),
        cost_usd=sum((u.cost_usd for u in items), Decimal(0)),
        model="+".join(models) if models else None,
    )


class InMemoryStageRunSink:
    """For tests and the dev CLI when no database is configured."""

    def __init__(self) -> None:
        self.runs: list[StageRun] = []

    async def record(self, run: StageRun) -> None:
        self.runs.append(run)


class TeeStageRunSink:
    """Send every run to several sinks (e.g. Postgres + an in-memory copy for a CLI summary)."""

    def __init__(self, *sinks: StageRunSink) -> None:
        self._sinks = sinks

    async def record(self, run: StageRun) -> None:
        for sink in self._sinks:
            await sink.record(run)
