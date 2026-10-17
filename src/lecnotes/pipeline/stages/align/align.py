from __future__ import annotations

from typing import Any

from lecnotes.domain import Alignment, ValidatedEvidence
from lecnotes.pipeline.context import RunContext
from lecnotes.pipeline.stage import Stage
from lecnotes.pipeline.stages.align.lexical import lexical_links
from lecnotes.pipeline.stages.align.temporal import temporal_links


class AlignStage(Stage[ValidatedEvidence, Alignment]):
    """Speech ↔ board links (no AI). v1: temporal. v2: + lexical."""

    name = "align"
    version = 2
    input_model = ValidatedEvidence
    output_model = Alignment

    def __init__(
        self, *, before_s: float = 15.0, after_s: float = 30.0, lexical_window_s: float = 180.0
    ) -> None:
        self.before_s = before_s
        self.after_s = after_s
        self.lexical_window_s = lexical_window_s

    def config(self, ctx: RunContext) -> dict[str, Any]:
        return {
            "before_s": self.before_s,
            "after_s": self.after_s,
            "lexical_window_s": self.lexical_window_s,
        }

    async def run(self, inp: ValidatedEvidence, ctx: RunContext) -> Alignment:
        ev = inp.evidence
        temporal = temporal_links(
            ev.segments,
            ev.board_states,
            ctx.require_video().duration_s,
            before_s=self.before_s,
            after_s=self.after_s,
        )
        lexical = lexical_links(
            ev.segments, ev.board_states, temporal, window_s=self.lexical_window_s
        )
        return Alignment(evidence=ev, links=temporal + lexical)
