"""Step 6.1 check with the real model: does verify remove an injected, unsupported unit?

    uv run python scripts/try_verify.py [--video Ek49cGiAOwo] [--judge-model gemini-2.5-flash]

Uses the recorded perception fixture (no YouTube/perception quota), the real gateway for
extract/verify/judge, and the Postgres stage cache so re-runs are cheap.
"""

import argparse
from pathlib import Path
from typing import cast
from uuid import uuid4

from evals.metrics.faithfulness import faithfulness
from lecnotes.adapters.cache.postgres_cache import PostgresStageCache
from lecnotes.adapters.perception.fake import FakePerceiver
from lecnotes.config import get_settings
from lecnotes.domain import (
    Evidence,
    ExtractedUnits,
    KnowledgeUnit,
    PerceptionResult,
    VideoMeta,
    VideoSpan,
)
from lecnotes.llm.gateway import LLMGateway
from lecnotes.observability.logging import configure_logging
from lecnotes.observability.metrics import InMemoryStageRunSink
from lecnotes.pipeline.context import RunContext
from lecnotes.pipeline.orchestrator import Orchestrator
from lecnotes.pipeline.stages.align.align import AlignStage
from lecnotes.pipeline.stages.align.chunk import ChunkStage
from lecnotes.pipeline.stages.extract import ExtractStage
from lecnotes.pipeline.stages.perceive import PerceiveStage
from lecnotes.pipeline.stages.validate import ValidateStage
from lecnotes.pipeline.stages.verify import VerifyStage
from lecnotes.runtime import run
from lecnotes.storage.db import create_engine

FIXTURES = Path(__file__).parents[1] / "tests" / "fixtures"
FABRICATED = (
    "zip() raises a ValueError when the lists have different lengths, so you must always "
    "check that the lists are the same length before calling it."
)


async def main(video_id: str, judge_model: str | None) -> None:
    settings = get_settings()
    configure_logging("WARNING", json=False)
    gateway = LLMGateway.from_settings(settings)
    perception = PerceptionResult.model_validate_json(
        (FIXTURES / f"perception_{video_id}.json").read_text(encoding="utf-8")
    )
    duration = max(s.end_s for s in perception.segments) + 1
    video = VideoMeta(video_id=video_id, title="verify check", duration_s=duration)
    ctx = RunContext(job_id=uuid4(), video=video)

    engine = create_engine(settings.database_url.get_secret_value())
    try:
        orch = Orchestrator(PostgresStageCache(engine), InMemoryStageRunSink())
        stages = [
            PerceiveStage(FakePerceiver({video_id: perception})),
            ValidateStage(),
            AlignStage(),
            ChunkStage(min_s=settings.chunk_min_s, max_s=settings.chunk_max_s),
            ExtractStage(gateway, model=settings.text_model),
        ]
        extracted = cast(ExtractedUnits, await orch.run_chain(stages, video, ctx))
    finally:
        await engine.dispose()
    assert extracted.evidence is not None

    zip_seg = next(s for s in extracted.evidence.segments if "zip" in s.text.lower())
    injected = KnowledgeUnit(
        id="INJECTED",
        concept="zip() with different lengths",
        explanation=FABRICATED,
        spans=[VideoSpan(video_id=video_id, start_s=zip_seg.start_s, end_s=zip_seg.end_s)],
        evidence=[Evidence(video_id=video_id, segment_ids=[zip_seg.id])],
    )
    with_fake = extracted.model_copy(update={"units": [*extracted.units, injected]})
    print(
        f"extracted {len(extracted.units)} units + 1 injected (cites {zip_seg.id}: "
        f"{zip_seg.text!r})"
    )

    before = await faithfulness(gateway, with_fake.units, extracted.evidence, model=judge_model)
    verified = await VerifyStage(gateway, model=settings.text_model).run(with_fake, ctx)
    after = await faithfulness(gateway, verified.units, extracted.evidence, model=judge_model)

    summary = verified.verify
    assert summary is not None
    survived = any(u.id == "INJECTED" for u in verified.units)
    print(f"\ninjected unit removed by verify: {not survived}")
    print(f"verify: {summary.model_dump()}  dropped rate {summary.dropped_rate:.0%}")
    print(
        f"faithfulness before verify: {before.score:.1%} ({before.supported}/{before.total} claims)"
    )
    print(f"faithfulness after verify:  {after.score:.1%} ({after.supported}/{after.total} claims)")
    print("\nverify removals:")
    for line in verified.dropped:
        if line.startswith("verify:"):
            print(" -", line[:220])
    if after.unsupported_claims:
        print("\nclaims the judge still flags after verify:")
        for unit_id, claims in after.unsupported_claims.items():
            for claim in claims:
                print(f" - {unit_id}: {claim[:160]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="Ek49cGiAOwo")
    parser.add_argument("--judge-model")
    args = parser.parse_args()
    run(main(args.video, args.judge_model))
