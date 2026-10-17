"""Compare two eval runs (step 6.2): metrics, per-stage changes, latency, cost, config.

    uv run python -m evals.compare BEFORE.json AFTER.json [--out FILE]
    uv run python -m evals.compare --latest          # the two most recent runs
    uv run python -m evals.compare --baseline evals/baseline.json AFTER.json
        --max-regression 0.05                         # exit 1 if a key metric drops (CI)

A stage counts as "changed" when its cache key differs for any shared video: equal keys
mean identical inputs, config and versions. So changing one prompt shows exactly that stage
and the stages downstream of it.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from evals.run import HIGHER_IS_BETTER, RUNS_DIR, RunRecord
from lecnotes.domain import StageRun

STAGE_ORDER = [
    "resolve", "perceive", "validate", "align", "chunk", "extract", "merge", "verify", "render",
]  # fmt: skip
GATED_METRICS = ("faithfulness", "keypoint_recall")
EPSILON = 1e-9


def percentile(values: list[float], q: float) -> float | None:
    """Linear interpolation, same as Postgres percentile_cont."""
    if not values:
        return None
    ordered = sorted(values)
    rank = q * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def mean_metric(record: RunRecord, metric: str, video_ids: set[str]) -> float | None:
    values = [
        value
        for v in record.videos
        if v.video_id in video_ids and (value := v.metrics.get(metric)) is not None
    ]
    return sum(values) / len(values) if values else None


@dataclass(frozen=True)
class MetricDelta:
    name: str
    before: float | None
    after: float | None

    @property
    def delta(self) -> float | None:
        if self.before is None or self.after is None:
            return None
        return self.after - self.before

    @property
    def regression(self) -> float:
        """How much worse (>0) the metric got, in the metric's own units."""
        if self.delta is None:
            return 0.0
        better_up = HIGHER_IS_BETTER.get(self.name, True)
        return max(0.0, -self.delta if better_up else self.delta)


def metric_deltas(a: RunRecord, b: RunRecord) -> list[MetricDelta]:
    shared = {v.video_id for v in a.videos} & {v.video_id for v in b.videos}
    names = sorted({m for r in (a, b) for v in r.videos for m in v.metrics})
    return [MetricDelta(n, mean_metric(a, n, shared), mean_metric(b, n, shared)) for n in names]


def _runs(record: RunRecord) -> list[StageRun]:
    return [r for v in record.videos for r in v.stage_runs]


def _keys(record: RunRecord) -> dict[tuple[str, str], str | None]:
    return {(v.video_id, r.stage): r.cache_key for v in record.videos for r in v.stage_runs}


def changed_stages(a: RunRecord, b: RunRecord) -> list[str]:
    ka, kb = _keys(a), _keys(b)
    shared = ka.keys() & kb.keys()
    return [s for s in STAGE_ORDER if any(ka[k] != kb[k] for k in shared if k[1] == s)]


def _pct(value: float | None) -> str:
    """Every metric is a fraction (rates, recall, F1, WER)."""
    return "n/a" if value is None else f"{value:.1%}"


def _stage_stats(runs: Iterable[StageRun], stage: str) -> tuple[str, str]:
    executed = [r for r in runs if r.stage == stage and not r.cache_hit]
    latencies = [r.latency_ms / 1000 for r in executed]
    p50, p95 = percentile(latencies, 0.5), percentile(latencies, 0.95)
    cost = sum((r.cost_usd for r in executed), Decimal(0))
    lat = "cached" if not executed else f"{p50:.1f}s / {p95:.1f}s"
    return lat, f"${cost:.4f}"


def render(a: RunRecord, b: RunRecord) -> str:
    lines = [
        f"# Eval compare: `{a.label}` → `{b.label}`",
        "",
        f"- before: `{a.run_id}` ({a.git_sha or '?'}), {len(a.videos)} video(s)",
        f"- after:  `{b.run_id}` ({b.git_sha or '?'}), {len(b.videos)} video(s)",
    ]
    for key in ("judge_model", "judge_thinking"):
        if a.config.get(key) != b.config.get(key):
            lines.append(
                f"- **{key} changed** ({a.config.get(key)} → {b.config.get(key)}): "
                "judged metrics are not comparable"
            )

    lines += ["", "## Metrics (mean over shared videos)", "", "| metric | before | after | Δ |",
              "|---|---|---|---|"]  # fmt: skip
    for m in metric_deltas(a, b):
        flag = " (worse)" if m.regression > EPSILON else ""
        delta = "n/a" if m.delta is None else f"{m.delta:+.1%}"
        lines.append(f"| {m.name} | {_pct(m.before)} | {_pct(m.after)} | {delta}{flag} |")

    changed = set(changed_stages(a, b))
    ra, rb = _runs(a), _runs(b)
    lines += ["", "## Stages", "",
              "| stage | changed | p50 / p95 before | p50 / p95 after | cost before | cost after |",
              "|---|---|---|---|---|---|"]  # fmt: skip
    for stage in STAGE_ORDER:
        if not any(r.stage == stage for r in ra + rb):
            continue
        lat_a, cost_a = _stage_stats(ra, stage)
        lat_b, cost_b = _stage_stats(rb, stage)
        mark = "**yes**" if stage in changed else "no"
        lines.append(f"| {stage} | {mark} | {lat_a} | {lat_b} | {cost_a} | {cost_b} |")

    if any(r.cache_hit for r in ra + rb if r.stage != "resolve"):
        lines += [
            "",
            "_Latency and cost show the work done **in each run**; `cached` = produced by an "
            "earlier run. For a fair cost comparison, run both sides with `--stage <first "
            "changed stage>`._",
        ]

    diffs = _config_diff(a.config, b.config)
    if diffs:
        lines += ["", "## Config changes", "", *(f"- `{k}`: {va} → {vb}" for k, va, vb in diffs)]
    return "\n".join(lines) + "\n"


def _config_diff(a: dict, b: dict, prefix: str = "") -> list[tuple[str, object, object]]:
    out = []
    for key in sorted(a.keys() | b.keys()):
        va, vb = a.get(key), b.get(key)
        path = f"{prefix}{key}"
        if isinstance(va, dict) and isinstance(vb, dict):
            out += _config_diff(va, vb, f"{path}.")
        elif va != vb:
            out.append((path, va, vb))
    return out


def regressions(a: RunRecord, b: RunRecord, threshold: float) -> list[MetricDelta]:
    return [m for m in metric_deltas(a, b) if m.name in GATED_METRICS and m.regression > threshold]


def load(path: Path) -> RunRecord:
    return RunRecord.model_validate_json(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two eval run records.")
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--latest", action="store_true", help="the two most recent runs")
    parser.add_argument("--baseline", type=Path, help="compare this run record to one run")
    parser.add_argument("--max-regression", type=float, help="fail if a gated metric drops more")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.latest:
        paths = sorted(RUNS_DIR.glob("*.json"))[-2:]
    elif args.baseline:
        paths = [args.baseline, *args.runs[:1]]
    else:
        paths = args.runs
    if len(paths) != 2:
        parser.error("need exactly two run records")
    a, b = load(paths[0]), load(paths[1])

    report = render(a, b)
    if args.out:
        args.out.write_text(report, encoding="utf-8")
    # Windows consoles default to cp1252, which can't encode "→" and would crash the print.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stdout.write(report)
    if args.max_regression is not None:
        bad = regressions(a, b, args.max_regression)
        for m in bad:
            print(f"REGRESSION {m.name}: {m.before:.3f} -> {m.after:.3f}", file=sys.stderr)
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
