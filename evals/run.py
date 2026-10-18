"""Eval runner (step 6.2): run the pipeline on every recorded perception fixture and score it.

    uv run python -m evals.run --label "extract v2" [--stage extract] [--video ID ...]
                               [--judge-model gemini-3-flash-preview] [--no-db] [--qa]
                               [--code-exec off|docker|local]

Perception is replayed from `tests/fixtures/perception_<id>.json` (no YouTube access, no
perception quota); everything after it runs for real, through the stage cache, so only
changed stages cost anything. `--stage X` forces X and everything downstream.

Per video: perception metrics + keypoint recall when golden references exist, faithfulness
(LLM judge) and verify drop rate always; with `--qa`, practice-question correctness and
yield. The run record (versions, models, judge, metrics, every StageRun) goes to
evals/reports/runs/ — feed two of them to `evals.compare`.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from evals.golden_set import GOLDEN_DIR, GoldenVideo, load_references, load_videos
from evals.metrics.board_recall import board_recall
from evals.metrics.boundary_f1 import boundary_f1, predicted_boundaries
from evals.metrics.code_exec import DockerRunner, LocalRunner, Runner, check_code
from evals.metrics.duplicates import duplicate_rate
from evals.metrics.faithfulness import faithfulness
from evals.metrics.keypoint_recall import keypoint_recall
from evals.metrics.qa_correctness import qa_correctness
from evals.metrics.sections import section_fragmentation
from evals.metrics.wer_terms import term_error_rate, wer
from lecnotes.adapters.cache.memory import InMemoryStageCache
from lecnotes.adapters.cache.postgres_cache import PostgresStageCache
from lecnotes.adapters.catalog.static import StaticCatalog
from lecnotes.adapters.perception.fake import FakePerceiver
from lecnotes.bootstrap import assemble
from lecnotes.config import Settings, get_settings
from lecnotes.domain import (
    JobOptions,
    PerceptionResult,
    QaLevel,
    StageRun,
    SubjectPreset,
    VideoMeta,
)
from lecnotes.llm.gateway import LLMGateway, ModelOptions
from lecnotes.observability.logging import configure_logging
from lecnotes.ports.cache import StageCache
from lecnotes.ports.metrics import StageRunSink
from lecnotes.runtime import run as run_async
from lecnotes.storage.db import create_engine
from lecnotes.storage.repositories.stage_runs import PostgresStageRunSink

REPO = Path(__file__).parents[1]
FIXTURES = REPO / "tests" / "fixtures"
RUNS_DIR = REPO / "evals" / "reports" / "runs"
PRESET = {"code": SubjectPreset.STEM, "slides": SubjectPreset.GENERAL}

# Gemini 3's default (high) thinking took >180 s → 504 on a 7-unit claim audit; "low" took
# 7 s with the same verdicts. Judged scores are only comparable at the same settings, so
# they are recorded in every run's config.
JUDGE_OPTIONS = ModelOptions(thinking_level="low", timeout_s=300)

# Metric name -> True if higher is better (used by compare).
HIGHER_IS_BETTER = {
    "faithfulness": True,
    "keypoint_recall": True,
    "board_recall": True,
    "boundary_f1": True,
    "wer": False,
    "term_error_rate": False,
    "verify_dropped_rate": False,
    "duplicate_rate": False,
    "section_fragmentation": False,
    "qa_correctness": True,
    "qa_yield": True,
    "code_compiles": True,
    "code_runs": True,
    "code_output_matches": True,
}
QA_COUNT = 5
# Lecture code is untrusted: it only runs when asked, and in Docker unless told otherwise.
RUNNERS: dict[str, Callable[[], Runner | None]] = {
    "off": lambda: None,
    "docker": DockerRunner,
    "local": LocalRunner,
}


class VideoResult(BaseModel):
    video_id: str
    title: str
    lecture_type: str | None = None
    job_id: UUID
    units: int
    metrics: dict[str, float | None]
    stage_runs: list[StageRun]


class RunRecord(BaseModel):
    run_id: str
    label: str
    created_at: datetime
    git_sha: str | None = None
    config: dict[str, Any]
    videos: list[VideoResult] = Field(default_factory=list)


def video_meta(video: GoldenVideo, perception: PerceptionResult, meta_path: Path) -> VideoMeta:
    """Title and duration from the fixture's .meta.json (the golden entry's title wins), else
    the video id and the last evidence timestamp. The title matters: prompts see it."""
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    duration = meta.get("duration_s")
    if not duration:
        ends = [s.end_s for s in perception.segments]
        ends += [b.t_last_seen for b in perception.board_states]
        duration = max(ends, default=0) + 1
    title = video.title if video.title != video.id else meta.get("title", video.id)
    return VideoMeta(video_id=video.id, title=title, duration_s=duration)


def fixture_ids(fixtures: Path) -> list[str]:
    return sorted(
        p.stem.removeprefix("perception_")
        for p in fixtures.glob("perception_*.json")
        if not p.name.endswith(".meta.json")
    )


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


async def run_eval(
    settings: Settings,
    gateway: LLMGateway,
    *,
    label: str,
    cache: StageCache,
    sinks: Sequence[StageRunSink] = (),
    fixtures: Path = FIXTURES,
    golden: Path = GOLDEN_DIR,
    video_ids: Sequence[str] | None = None,
    rerun_from: str | None = None,
    judge_model: str | None = None,
    qa: bool = False,
    code_runner: Runner | None = None,
) -> RunRecord:
    golden_videos = {v.id: v for v in load_videos(golden)}
    ids = list(video_ids) if video_ids else fixture_ids(fixtures)
    record: RunRecord | None = None

    for video_id in ids:
        perception = PerceptionResult.model_validate_json(
            (fixtures / f"perception_{video_id}.json").read_text(encoding="utf-8")
        )
        gv = golden_videos.get(video_id) or GoldenVideo(id=video_id, title=video_id, type="mixed")
        meta = video_meta(gv, perception, fixtures / f"perception_{video_id}.meta.json")
        services = assemble(
            settings=settings,
            catalog=StaticCatalog([meta]),
            perceiver=FakePerceiver({video_id: perception}),
            gateway=gateway,
            cache=cache,
            sinks=sinks,
        )
        if record is None:
            record = RunRecord(
                run_id=f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{_slug(label)}",
                label=label,
                created_at=datetime.now(UTC),
                git_sha=_git_sha(),
                config={
                    "stages": services.pipeline.describe(),
                    "judge_model": judge_model or settings.text_model,
                    "judge_thinking": JUDGE_OPTIONS.thinking_level,
                    "features": settings.features.model_dump(),
                    "chunk_s": [settings.chunk_min_s, settings.chunk_max_s],
                },
            )

        job_id = uuid4()
        preset = PRESET.get(gv.type, SubjectPreset.GENERAL)
        doc = await services.pipeline.run(
            video_id,
            job_id=job_id,
            options=JobOptions(subject_preset=preset),
            rerun_from=rerun_from,
        )

        metrics: dict[str, float | None] = {
            "faithfulness": None,
            "verify_dropped_rate": doc.verify.dropped_rate if doc.verify else None,
            "duplicate_rate": None,
            "section_fragmentation": section_fragmentation(doc.units).fragmentation,
        }
        if doc.evidence is not None and doc.units:
            score = await faithfulness(
                gateway, doc.units, doc.evidence, model=judge_model, options=JUDGE_OPTIONS
            )
            metrics["faithfulness"] = score.score
            dups = await duplicate_rate(
                gateway, doc.units, model=judge_model, options=JUDGE_OPTIONS
            )
            metrics["duplicate_rate"] = dups.rate
        code = check_code(doc.units, code_runner)
        metrics["code_compiles"] = code.compile_rate
        metrics["code_runs"] = code.run_rate
        metrics["code_output_matches"] = code.output_match_rate
        for res in code.results:
            if res.status not in ("ok", "skipped") or res.output_matches is False:
                problem = res.error or ("printed something else" if res.status == "ok" else "")
                print(f"  code {video_id} {res.unit_id}[{res.index}]: {res.status} {problem}")
        if qa and doc.units:
            qset = await services.qa.generate(
                job_id=job_id, video=meta, units=doc.units, level=QaLevel.MIXED, count=QA_COUNT
            )
            judged = await qa_correctness(
                gateway, qset.questions, doc.units, model=judge_model, options=JUDGE_OPTIONS
            )
            metrics["qa_correctness"] = judged.score
            metrics["qa_yield"] = len(qset.questions) / QA_COUNT
        if video_id in golden_videos:
            refs = load_references(video_id, golden)
            spoken = " ".join(s.text for s in sorted(perception.segments, key=lambda s: s.start_s))
            terms = term_error_rate(refs.captions, spoken, gv.terms)
            metrics |= {
                "wer": wer(refs.captions, spoken),
                "term_error_rate": terms.error_rate if terms else None,
                "board_recall": board_recall(refs.boards, perception.board_states).recall,
                "boundary_f1": boundary_f1(
                    refs.boundaries, predicted_boundaries(perception.board_states)
                ).f1,
            }
            kp = await keypoint_recall(
                gateway, doc.markdown, refs.keypoints, model=judge_model, options=JUDGE_OPTIONS
            )
            metrics["keypoint_recall"] = kp.recall

        record.videos.append(
            VideoResult(
                video_id=video_id,
                title=gv.title,
                lecture_type=gv.type if video_id in golden_videos else None,
                job_id=job_id,
                units=len(doc.units),
                metrics=metrics,
                stage_runs=[r for r in services.runs.runs if r.job_id == job_id],
            )
        )
    if record is None:
        raise SystemExit(f"no perception fixtures found in {fixtures}")
    return record


def save(record: RunRecord, directory: Path = RUNS_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record.run_id}.json"
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return path


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "run"


async def _main(args: argparse.Namespace) -> None:
    settings = get_settings()
    configure_logging("WARNING", json=False)
    gateway = LLMGateway.from_settings(settings)
    common: dict[str, Any] = {
        "label": args.label,
        "video_ids": args.video,
        "rerun_from": args.stage,
        "judge_model": args.judge_model,
        "qa": args.qa,
        "code_runner": RUNNERS[args.code_exec](),
    }
    if args.no_db:
        record = await run_eval(settings, gateway, cache=InMemoryStageCache(), **common)
    else:
        engine = create_engine(settings.database_url.get_secret_value())
        try:
            record = await run_eval(
                settings,
                gateway,
                cache=PostgresStageCache(engine),
                sinks=[PostgresStageRunSink(engine)],
                **common,
            )
        finally:
            await engine.dispose()
    path = save(record)
    for v in record.videos:
        shown = {k: (None if m is None else round(m, 3)) for k, m in v.metrics.items()}
        print(f"{v.video_id}: {v.units} units, {shown}")
    print(f"\nrun record: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pipeline on perception fixtures.")
    parser.add_argument("--label", required=True, help="what changed, e.g. 'extract prompt v2'")
    parser.add_argument("--stage", help="force this stage and everything downstream")
    parser.add_argument("--video", action="append", help="limit to these video ids")
    parser.add_argument("--judge-model", default="gemini-3-flash-preview")
    parser.add_argument("--no-db", action="store_true")
    parser.add_argument("--qa", action="store_true", help="also generate and judge questions")
    parser.add_argument(
        "--code-exec",
        choices=sorted(RUNNERS),
        default="off",
        help="run the notes' Python code: docker (CI) or local (reviewed code only)",
    )
    run_async(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
