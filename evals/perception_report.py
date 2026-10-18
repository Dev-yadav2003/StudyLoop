"""Step 3.5: perception baseline — score saved perception fixtures against the golden set.

    uv run python -m evals.perception_report [--fixtures tests/fixtures] [--out FILE]

For every golden video with a `perception_<id>.json` fixture: WER, term error rate,
board recall (overall + by board type), boundary P/R/F1 and drift. Prints a Markdown
report and writes it to evals/reports/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from evals.golden_set import GOLDEN_DIR, GoldenVideo, load_references, load_videos
from evals.metrics.board_recall import BoardRecall, board_recall
from evals.metrics.boundary_f1 import BoundaryScore, boundary_f1, predicted_boundaries
from evals.metrics.wer_terms import TermScore, term_error_rate, wer
from lecnotes.domain import PerceptionResult

REPO = Path(__file__).parents[1]
DEFAULT_FIXTURES = REPO / "tests" / "fixtures"
REPORTS = REPO / "evals" / "reports"


@dataclass(frozen=True)
class VideoScores:
    video: GoldenVideo
    settings: str
    wer: float
    terms: TermScore | None
    board: BoardRecall
    boundaries: BoundaryScore


def score_video(video: GoldenVideo, fixtures: Path, golden: Path) -> VideoScores | None:
    path = fixtures / f"perception_{video.id}.json"
    if not path.is_file():
        return None
    result = PerceptionResult.model_validate_json(path.read_text(encoding="utf-8"))
    refs = load_references(video.id, golden)
    spoken = " ".join(s.text for s in sorted(result.segments, key=lambda s: s.start_s))
    return VideoScores(
        video=video,
        settings=_settings_label(fixtures / f"perception_{video.id}.meta.json"),
        wer=wer(refs.captions, spoken),
        terms=term_error_rate(refs.captions, spoken, video.terms),
        board=board_recall(refs.boards, result.board_states),
        boundaries=boundary_f1(refs.boundaries, predicted_boundaries(result.board_states)),
    )


def _settings_label(meta_path: Path) -> str:
    if not meta_path.is_file():
        return "?"
    fp = json.loads(meta_path.read_text(encoding="utf-8")).get("fingerprint", {})
    return (
        f"{fp.get('model', '?')} p{fp.get('prompt_version', '?')} "
        f"{fp.get('fps', '?')}fps {fp.get('media_resolution', '?')}"
    )


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def render(scores: list[VideoScores], missing: list[GoldenVideo]) -> str:
    lines = [
        f"# Perception report - {datetime.now(UTC):%Y-%m-%d %H:%M} UTC",
        "",
        "| Video | Type | Settings | WER | Term err | Board recall | Boundary P / R / F1 | Drift |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in sorted(scores, key=lambda s: (s.video.type, s.video.id)):
        b = s.boundaries
        drift = "n/a" if b.mean_drift_s is None else f"{b.mean_drift_s:.1f}s"
        lines.append(
            f"| {s.video.id} | {s.video.type} | {s.settings} | {_pct(s.wer)} | "
            f"{_pct(s.terms.error_rate if s.terms else None)} | "
            f"{_pct(s.board.recall)} ({s.board.matched}/{s.board.total}) | "
            f"{b.precision:.2f} / {b.recall:.2f} / {b.f1:.2f} | {drift} |"
        )
    if missing:
        lines += ["", "No fixture yet: " + ", ".join(v.id for v in missing)]
    for s in scores:
        detail = []
        if s.board.by_type:
            by_type = ", ".join(f"{t} {m}/{n}" for t, (m, n) in sorted(s.board.by_type.items()))
            detail.append(f"- board recall by type: {by_type}")
        if s.board.misses:
            detail.append(
                "- missed board lines: " + "; ".join(f"`{m}`" for m in s.board.misses[:8])
            )
        if s.terms and s.terms.missed:
            detail.append("- missed terms: " + ", ".join(s.terms.missed))
        if detail:
            lines += ["", f"### {s.video.id} - {s.video.title}", *detail]
    return "\n".join(lines) + "\n"


def build_report(fixtures: Path = DEFAULT_FIXTURES, golden: Path = GOLDEN_DIR) -> str:
    scores, missing = [], []
    for video in load_videos(golden):
        scored = score_video(video, fixtures, golden)
        if scored is None:
            missing.append(video)
        else:
            scores.append(scored)
    return render(scores, missing)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score perception fixtures vs the golden set.")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = build_report(args.fixtures)
    out = args.out or REPORTS / f"perception_{datetime.now(UTC):%Y%m%d_%H%M}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
