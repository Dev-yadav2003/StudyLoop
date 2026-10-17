"""Step 5.8 baseline: keypoint recall of the notes, per golden video.

    uv run python -m evals.notes_report [--fixtures tests/fixtures] [--out FILE]

Runs the real chain validate → … → render on the *recorded* perception fixtures (no YouTube
access, no perception quota), with the real Gemini gateway for extract, then grades each
video's notes against its golden keypoints with the LLM judge.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from evals.golden_set import GOLDEN_DIR, GoldenVideo, load_references, load_videos
from evals.metrics.keypoint_recall import KeypointScore, keypoint_recall
from lecnotes.adapters.cache.memory import InMemoryStageCache
from lecnotes.adapters.catalog.static import StaticCatalog
from lecnotes.adapters.perception.fake import FakePerceiver
from lecnotes.bootstrap import assemble
from lecnotes.config import Settings, get_settings
from lecnotes.domain import JobOptions, NotesDocument, PerceptionResult, SubjectPreset, VideoMeta
from lecnotes.llm.gateway import LLMGateway
from lecnotes.observability.logging import configure_logging
from lecnotes.runtime import run as run_async

REPO = Path(__file__).parents[1]
DEFAULT_FIXTURES = REPO / "tests" / "fixtures"
REPORTS = REPO / "evals" / "reports"
PRESET = {"code": SubjectPreset.STEM, "slides": SubjectPreset.GENERAL}


@dataclass(frozen=True)
class NotesScore:
    video: GoldenVideo
    doc: NotesDocument
    keypoints: KeypointScore


def video_meta(video: GoldenVideo, perception: PerceptionResult, meta_path: Path) -> VideoMeta:
    duration = None
    if meta_path.is_file():
        duration = json.loads(meta_path.read_text(encoding="utf-8")).get("duration_s")
    if not duration:  # fall back to the last evidence timestamp
        ends = [s.end_s for s in perception.segments] + [
            b.t_last_seen for b in perception.board_states
        ]
        duration = max(ends, default=0) + 1
    return VideoMeta(video_id=video.id, title=video.title, duration_s=duration)


async def score_videos(
    settings: Settings,
    gateway: LLMGateway,
    *,
    fixtures: Path = DEFAULT_FIXTURES,
    golden: Path = GOLDEN_DIR,
) -> tuple[list[NotesScore], list[GoldenVideo]]:
    scores, missing = [], []
    for video in load_videos(golden):
        path = fixtures / f"perception_{video.id}.json"
        if not path.is_file():
            missing.append(video)
            continue
        perception = PerceptionResult.model_validate_json(path.read_text(encoding="utf-8"))
        meta = video_meta(video, perception, fixtures / f"perception_{video.id}.meta.json")
        services = assemble(
            settings=settings,
            catalog=StaticCatalog([meta]),
            perceiver=FakePerceiver({video.id: perception}),
            gateway=gateway,
            cache=InMemoryStageCache(),
        )
        options = JobOptions(subject_preset=PRESET.get(video.type, SubjectPreset.GENERAL))
        doc = await services.pipeline.run(video.id, job_id=uuid4(), options=options)
        kp = await keypoint_recall(
            gateway, doc.markdown, load_references(video.id, golden).keypoints
        )
        scores.append(NotesScore(video=video, doc=doc, keypoints=kp))
    return scores, missing


def render(scores: list[NotesScore], missing: list[GoldenVideo], settings: Settings) -> str:
    lines = [
        f"# Notes report - {datetime.now(UTC):%Y-%m-%d %H:%M} UTC",
        "",
        f"extract prompt v{settings.extract_prompt_version}, model {settings.text_model}",
        "",
        "| Video | Type | Units | Keypoint recall | Unsupported | Ungraded |",
        "|---|---|---|---|---|---|",
    ]
    for s in sorted(scores, key=lambda s: (s.video.type, s.video.id)):
        k = s.keypoints
        lines.append(
            f"| {s.video.id} | {s.video.type} | {len(s.doc.units)} | "
            f"{k.recall:.0%} ({len(k.covered)}/{len(k.covered) + len(k.missed)}) | "
            f"{len(k.unsupported)} | {len(k.ungraded)} |"
        )
    if missing:
        lines += ["", "No perception fixture yet: " + ", ".join(v.id for v in missing)]
    for s in scores:
        if s.keypoints.missed:
            lines += ["", f"### {s.video.id} - missed keypoints"]
            lines += [f"- {k}" for k in s.keypoints.missed]
    return "\n".join(lines) + "\n"


async def _main(out: Path | None, fixtures: Path) -> None:
    settings = get_settings()
    configure_logging(settings.log_level, json=False)
    scores, missing = await score_videos(
        settings, LLMGateway.from_settings(settings), fixtures=fixtures
    )
    report = render(scores, missing, settings)
    path = out or REPORTS / f"notes_{datetime.now(UTC):%Y%m%d_%H%M}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keypoint recall of notes vs the golden set.")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    run_async(_main(args.out, args.fixtures))


if __name__ == "__main__":
    main()
