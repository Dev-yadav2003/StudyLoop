from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import SecretStr

from evals.compare import (
    changed_stages,
    load,
    metric_deltas,
    percentile,
    regressions,
    render,
)
from evals.metrics.faithfulness import Claim, FaithfulnessJudgement, UnitClaims
from evals.run import RunRecord, VideoResult, run_eval, save
from lecnotes.adapters.cache.memory import InMemoryStageCache
from lecnotes.adapters.llm.fake import FakeLLM
from lecnotes.adapters.perception.schema import PerceiveOutput, to_perception_result
from lecnotes.config import Settings
from lecnotes.domain import StageRun, StageRunStatus
from lecnotes.llm.gateway import LLMGateway
from lecnotes.pipeline.stages.extract import ExtractedUnit, ExtractOutput
from lecnotes.pipeline.stages.verify import VerifyOutput

SAMPLE = Path(__file__).parents[1] / "fixtures" / "perceive_output_sample.json"
VID = "abcdefghijk"


def settings_with(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> Settings:
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    s = SecretStr("x")
    fields: dict[str, object] = {
        "gemini_api_key": s,
        "youtube_api_key": s,
        "database_url": s,
        "langfuse_public_key": s,
        "langfuse_secret_key": s,
    }
    return Settings(**(fields | overrides))  # type: ignore[arg-type]


def fixtures_dir(root: Path) -> Path:
    raw = PerceiveOutput.model_validate_json(SAMPLE.read_text(encoding="utf-8"))
    result = to_perception_result(raw, VID).result
    path = root / "fixtures"
    path.mkdir()
    (path / f"perception_{VID}.json").write_text(result.model_dump_json(), encoding="utf-8")
    (root / "golden").mkdir()
    (root / "golden" / "videos.yaml").write_text("videos: []\n", encoding="utf-8")
    return path


def fake_llm() -> FakeLLM:
    unit = ExtractedUnit(
        section="Iteration",
        concept="enumerate()",
        explanation="enumerate gives you the index and the item.",
        examples=[],
        board_text="",
        code=[],
        segment_ids=["s0002"],
        board_state_ids=[],
    )
    judgement = FaithfulnessJudgement(
        units=[
            UnitClaims(
                unit_id=f"{VID}-c001-u01",
                claims=[
                    Claim(claim="enumerate gives the index", supported=True),
                    Claim(claim="and the item", supported=True),
                ],
            )
        ]
    )
    return FakeLLM(
        {
            "extract": ExtractOutput(units=[unit]),
            "verify": VerifyOutput(results=[]),
            "judge/faithfulness": judgement,
        }
    )


async def run_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **changed: object
) -> tuple[RunRecord, RunRecord]:
    fixtures = fixtures_dir(tmp_path)
    cache = InMemoryStageCache()  # shared, like the Postgres cache between real runs
    gateway = LLMGateway(fake_llm().factory, default_model="gemini-2.5-flash", retry_backoff=False)
    common = {"cache": cache, "fixtures": fixtures, "golden": tmp_path / "golden"}
    before = await run_eval(settings_with(monkeypatch), gateway, label="baseline", **common)
    after = await run_eval(
        settings_with(monkeypatch, **changed), gateway, label="shorter chunks", **common
    )
    return before, after


async def test_real_chunking_change_shows_only_downstream_stages_changed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The ~3 min sample becomes several chunks instead of one.
    before, after = await run_pair(monkeypatch, tmp_path, chunk_min_s=30, chunk_max_s=60)

    [video] = before.videos
    assert video.units == 1
    assert video.metrics["faithfulness"] == 1.0
    assert {r.stage for r in video.stage_runs} >= {"perceive", "extract", "verify", "render"}
    assert all(r.cache_key for r in video.stage_runs if r.stage != "resolve")

    assert changed_stages(before, after) == ["chunk", "extract", "verify", "render"]
    report = render(before, after)
    assert "| perceive | no |" in report
    assert "| align | no |" in report
    assert "| chunk | **yes** |" in report
    assert "`chunk_s`: [120, 300] → [30, 60]" in report
    assert "`baseline` → `shorter chunks`" in report

    path = save(after, tmp_path / "runs")
    assert load(path) == after


async def test_config_change_with_identical_output_stays_cached_downstream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Keys are content-addressed: 300 → 200 s yields the same chunks for a ~3 min video,
    so extract gets byte-identical input and hits the cache."""
    before, after = await run_pair(monkeypatch, tmp_path, chunk_max_s=200)
    assert changed_stages(before, after) == ["chunk"]
    extract_after = [r for r in after.videos[0].stage_runs if r.stage == "extract"]
    assert all(r.cache_hit for r in extract_after)


def record(label: str, faithfulness: float, keypoints: float) -> RunRecord:
    run = StageRun(
        job_id=uuid4(),
        stage="extract",
        stage_version=1,
        status=StageRunStatus.OK,
        latency_ms=2000,
        cost_usd=Decimal("0.01"),
        started_at=datetime.now(UTC),
        cache_key="k",
    )
    return RunRecord(
        run_id=label,
        label=label,
        created_at=datetime.now(UTC),
        config={"judge_model": "gemini-3-flash-preview"},
        videos=[
            VideoResult(
                video_id=VID,
                title="t",
                job_id=uuid4(),
                units=3,
                metrics={
                    "faithfulness": faithfulness,
                    "keypoint_recall": keypoints,
                    "wer": 0.1,
                },
                stage_runs=[run],
            )
        ],
    )


def test_regressions_gate_only_key_metrics_beyond_threshold() -> None:
    a, b = record("a", 0.95, 0.80), record("b", 0.88, 0.79)
    assert [m.name for m in regressions(a, b, threshold=0.05)] == ["faithfulness"]
    assert regressions(b, a, threshold=0.05) == []  # improvements never fail


def test_lower_is_better_metrics() -> None:
    a, b = record("a", 0.9, 0.8), record("b", 0.9, 0.8)
    b.videos[0].metrics["wer"] = 0.2
    wer = next(m for m in metric_deltas(a, b) if m.name == "wer")
    assert wer.regression == pytest.approx(0.1)


def test_judge_change_is_called_out() -> None:
    a, b = record("a", 0.9, 0.8), record("b", 0.9, 0.8)
    b.config["judge_model"] = "gemini-2.5-flash"
    assert "judge_model changed" in render(a, b)
    assert "judged metrics are not comparable" in render(a, b)


def test_percentile_matches_postgres_percentile_cont() -> None:
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 0.5) == pytest.approx(50.5)
    assert percentile(values, 0.95) == pytest.approx(95.05)
    assert percentile([], 0.5) is None
