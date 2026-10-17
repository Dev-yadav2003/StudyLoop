"""Step 6.3 helper: start a golden-set entry for one lecture.

    uv run python scripts/new_golden_video.py <youtube-url> --type code \
        [--terms "enumerate,zip,iterable"] [--tags fast-talker] [--captions]

Validates the video (public, via the YouTube Data API), creates
evals/golden/references/<id>/ with instruction-only skeletons, optionally downloads captions
with yt-dlp (your laptop only — the server never touches YouTube), and adds the video to
videos.yaml as `draft: true`. Drafts are never scored.

The skeletons are deliberately empty: ground truth must be typed by a human watching the
video. Pre-filling it from the model's output would anchor you to the model's mistakes and
make the metrics circular.
"""

from __future__ import annotations

import argparse

import httpx

from evals.golden_scaffold import create_skeleton, download_captions
from evals.golden_set import GOLDEN_DIR
from lecnotes.adapters.catalog.youtube_data_api import YouTubeDataApiCatalog
from lecnotes.config import get_settings
from lecnotes.domain import VideoMeta
from lecnotes.runtime import run
from lecnotes.timefmt import format_timestamp


async def resolve(url: str) -> VideoMeta:
    settings = get_settings()
    async with httpx.AsyncClient(timeout=20) as http:
        catalog = YouTubeDataApiCatalog(http, settings.youtube_api_key.get_secret_value())
        return (await catalog.resolve(url)).videos[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Start a golden-set entry for one lecture.")
    parser.add_argument("url")
    parser.add_argument("--type", required=True, choices=["slides", "whiteboard", "code", "mixed"])
    parser.add_argument("--terms", default="", help="comma-separated technical terms")
    parser.add_argument("--tags", default="", help="comma-separated, e.g. fast-talker")
    parser.add_argument("--captions", action="store_true", help="download captions (yt-dlp)")
    args = parser.parse_args()

    meta = run(resolve(args.url))
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]  # noqa: E731
    folder = create_skeleton(
        GOLDEN_DIR, meta, args.type, terms=split(args.terms), tags=split(args.tags)
    )
    captions = download_captions(meta.url, folder) if args.captions else None

    duration = format_timestamp(meta.duration_s)
    print(f"\nCreated {folder} for {meta.title!r} ({duration}), marked draft.\n")
    print("To finish (watch the video):")
    print(
        f"  [{'x' if captions else ' '}] captions.vtt"
        + ("" if captions else "  (re-run with --captions, or see evals/golden/README.md)")
    )
    print("  [ ] board_text.md    verbatim board/slide/code text with MM:SS")
    print("  [ ] boundaries.yaml  times where the board changed")
    print("  [ ] keypoints.yaml   10-20 facts the notes must contain")
    print(
        f"  [ ] perception fixture: uv run python scripts/try_perceive.py {meta.video_id} "
        f"--duration {duration}"
    )
    print("  [ ] set draft: false in evals/golden/videos.yaml")


if __name__ == "__main__":
    main()
