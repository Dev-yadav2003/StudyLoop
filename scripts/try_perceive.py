"""Step 3.4: run real perception on one YouTube video and save it as a test fixture.

    uv run python scripts/try_perceive.py <youtube-url-or-id> [--duration 12:34]
        [--fps 1] [--res low|medium|high] [--prompt-version 1]

Writes tests/fixtures/perception_<id>.json (the PerceptionResult) and
perception_<id>.meta.json (settings, usage, dropped/repaired items) so baselines in the
results are reproducible. Requires GEMINI_API_KEY and LANGFUSE_* in .env.
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import get_args

from lecnotes.adapters.perception.gemini_url import GeminiUrlPerceiver
from lecnotes.config import MediaResolution, get_settings
from lecnotes.domain import VideoRef
from lecnotes.llm.gateway import LLMGateway
from lecnotes.observability.logging import configure_logging
from lecnotes.observability.tracing import (
    get_langfuse,
    init_tracing_from_settings,
    shutdown_tracing,
    traced_job,
    traced_stage,
)
from lecnotes.timefmt import parse_timestamp

FIXTURES = Path(__file__).parents[1] / "tests" / "fixtures"


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run real perception on one YouTube video.")
    parser.add_argument("video", help="YouTube URL or 11-char id")
    parser.add_argument("--duration", help="MM:SS or H:MM:SS (helps the model bound times)")
    parser.add_argument("--fps", type=float, default=settings.perceive_fps)
    parser.add_argument(
        "--res", choices=get_args(MediaResolution), default=settings.perceive_media_resolution
    )
    parser.add_argument("--prompt-version", type=int, default=settings.perceive_prompt_version)
    parser.add_argument("--out", type=Path, default=FIXTURES)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    init_tracing_from_settings(settings)

    video = VideoRef.from_url(args.video)
    perceiver = GeminiUrlPerceiver(
        LLMGateway.from_settings(settings),
        model=settings.perceive_model,
        fps=args.fps,
        media_resolution=args.res,
        prompt_version=args.prompt_version,
        timeout_s=settings.perceive_timeout_s,
    )
    duration_s = parse_timestamp(args.duration) if args.duration else None

    try:
        with (
            traced_job(f"perceive-{video.video_id}", video_id=video.video_id) as root,
            traced_stage("perceive", version=1, prompt_version=args.prompt_version),
        ):
            run = await perceiver.perceive(video, duration_s=duration_s)
            trace_url = get_langfuse().get_trace_url(trace_id=root.trace_id)
    finally:
        shutdown_tracing()

    args.out.mkdir(parents=True, exist_ok=True)
    result_path = args.out / f"perception_{video.video_id}.json"
    meta_path = args.out / f"perception_{video.video_id}.meta.json"
    result_path.write_text(run.result.model_dump_json(indent=2), encoding="utf-8")
    meta = {
        "fingerprint": perceiver.fingerprint,
        "duration_s": duration_s,
        "usage": [u.model_dump(mode="json") for u in run.usage],
        "dropped": run.dropped,
        "repaired": run.repaired,
        "trace_url": trace_url,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    by_type: dict[str, int] = {}
    for b in run.result.board_states:
        by_type[b.type] = by_type.get(b.type, 0) + 1
    usage = run.usage[0]
    print(
        f"\n{video.video_id}: {len(run.result.segments)} segments, "
        f"{len(run.result.board_states)} board states {by_type}\n"
        f"dropped {len(run.dropped)}, repaired {len(run.repaired)}\n"
        f"{usage.model}: {usage.tokens_in:,} in / {usage.tokens_out:,} out tokens, "
        f"${usage.cost_usd:.4f} list price, {usage.latency_ms / 1000:.1f}s\n"
        f"saved {result_path}\ntrace {trace_url}\n"
    )


if __name__ == "__main__":
    asyncio.run(main())
