"""Developer CLI: run the full pipeline on one link, locally, against real services.

    uv run python -m lecnotes.cli run <youtube-url> [--preset stem] [--out notes/]
                                      [--rerun-from extract] [--no-db]

Writes notes/<video_id>.md and prints one line per stage (cache hit, tokens, cost, latency).
Needs the keys in .env; with --no-db, cache and metrics stay in memory.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from decimal import Decimal
from pathlib import Path

from lecnotes.bootstrap import Services, production_services
from lecnotes.config import ConfigError, Settings, get_settings
from lecnotes.domain import JobOptions, StageRun, SubjectPreset
from lecnotes.errors import LecNotesError
from lecnotes.observability.logging import configure_logging
from lecnotes.observability.tracing import init_tracing_from_settings, shutdown_tracing
from lecnotes.pipeline.orchestrator import StageFailed
from lecnotes.pipeline.registry import VIDEO_STAGE_ORDER
from lecnotes.runtime import run as run_async

ServicesFactory = Callable[[Settings, bool], AbstractAsyncContextManager[Services]]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lecnotes", description="Lecture-to-Notes dev CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the pipeline on a YouTube video or playlist link")
    run.add_argument("url")
    run.add_argument("--preset", choices=[p.value for p in SubjectPreset], default="general")
    run.add_argument("--out", type=Path, default=Path("notes"))
    run.add_argument("--rerun-from", choices=VIDEO_STAGE_ORDER)
    run.add_argument("--no-db", action="store_true", help="in-memory cache and metrics")
    run.add_argument("--no-trace", action="store_true", help="don't send traces to Langfuse")
    return parser


def format_runs(runs: Sequence[StageRun]) -> str:
    lines = [
        f"{'stage':<10} {'video':<12} {'cache':<5} {'tokens in/out':>15} {'cost $':>9} {'s':>7}"
    ]
    for r in runs:
        tokens = f"{r.tokens_in:,}/{r.tokens_out:,}" if r.tokens_in or r.tokens_out else "-"
        lines.append(
            f"{r.stage:<10} {r.video_id or '-':<12} {'hit' if r.cache_hit else '':<5} "
            f"{tokens:>15} {r.cost_usd:>9.4f} {r.latency_ms / 1000:>7.1f}"
            + (f"  FAILED {r.error_type}" if r.error_type else "")
        )
    total = sum((r.cost_usd for r in runs), Decimal(0))
    lines.append(f"{'total':<10} {'':<12} {'':<5} {'':>15} {total:>9.4f}")
    return "\n".join(lines)


async def _run(args: argparse.Namespace, settings: Settings, services: ServicesFactory) -> int:
    job_id = uuid.uuid4()
    async with services(settings, not args.no_db) as svc:
        try:
            docs = await svc.pipeline.run(
                args.url,
                job_id=job_id,
                options=JobOptions(subject_preset=SubjectPreset(args.preset)),
                rerun_from=args.rerun_from,
            )
        except LecNotesError as exc:
            print(format_runs(svc.runs.runs), file=sys.stderr)
            stage = f" in stage {exc.stage!r}" if isinstance(exc, StageFailed) else ""
            print(f"\nFailed{stage}: {exc.user_message}\n  ({exc})", file=sys.stderr)
            return 1
        print(format_runs(svc.runs.runs))

    args.out.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        path = args.out / f"{doc.videos[0].video_id}.md"
        path.write_text(doc.markdown, encoding="utf-8")
        print(f"\n{doc.title}: {len(doc.units)} units -> {path}")
    print(f"job {job_id}")
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: Settings | None = None,
    services: ServicesFactory | None = None,
) -> int:
    args = _parser().parse_args(argv)
    try:
        settings = settings or get_settings()
    except ConfigError as exc:
        print(f"{exc}\n\nCopy .env.example to .env and fill it in.", file=sys.stderr)
        return 2
    configure_logging(settings.log_level, json=False)
    if not args.no_trace and services is None:
        init_tracing_from_settings(settings)
    try:
        return run_async(_run(args, settings, services or production_services_factory))
    finally:
        shutdown_tracing()


def production_services_factory(
    settings: Settings, use_db: bool
) -> AbstractAsyncContextManager[Services]:
    return production_services(settings, use_db=use_db)


if __name__ == "__main__":
    sys.exit(main())
