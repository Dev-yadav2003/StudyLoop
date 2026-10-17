import shutil
from pathlib import Path

import pytest
from pydantic import SecretStr

from evals.golden_set import GOLDEN_DIR
from evals.metrics.keypoint_recall import KeypointJudgement, KeypointVerdict
from evals.notes_report import render, score_videos
from lecnotes.adapters.llm.fake import FakeLLM
from lecnotes.adapters.perception.schema import PerceiveOutput, to_perception_result
from lecnotes.config import Settings
from lecnotes.llm.gateway import LLMGateway
from lecnotes.pipeline.stages.extract import ExtractedUnit, ExtractOutput

VID, MISSING = "abcdefghijk", "zzzzzzzzzzz"
SAMPLE = Path(__file__).parents[1] / "fixtures" / "perceive_output_sample.json"


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    s = SecretStr("x")
    return Settings(
        gemini_api_key=s,
        youtube_api_key=s,
        database_url=s,
        langfuse_public_key=s,
        langfuse_secret_key=s,
    )


def golden_and_fixtures(root: Path) -> tuple[Path, Path]:
    golden = root / "golden"
    shutil.copytree(GOLDEN_DIR / "references" / "_template", golden / "references" / VID)
    (golden / "videos.yaml").write_text(
        f"videos:\n  - {{id: {VID}, title: Iteration, type: code}}\n"
        f"  - {{id: {MISSING}, title: Later, type: slides}}\n",
        encoding="utf-8",
    )
    fixtures = root / "fixtures"
    fixtures.mkdir()
    raw = PerceiveOutput.model_validate_json(SAMPLE.read_text(encoding="utf-8"))
    result = to_perception_result(raw, VID).result
    (fixtures / f"perception_{VID}.json").write_text(result.model_dump_json(), encoding="utf-8")
    return golden, fixtures


async def test_notes_report_runs_pipeline_on_fixtures_and_judges(
    settings: Settings, tmp_path: Path
) -> None:
    golden, fixtures = golden_and_fixtures(tmp_path)
    explanation = "enumerate() yields (index, item) pairs."
    llm = FakeLLM(
        {
            "extract": ExtractOutput(
                units=[
                    ExtractedUnit(
                        section="Iteration",
                        concept="enumerate()",
                        explanation=explanation,
                        examples=[],
                        board_text="",
                        code=[],
                        segment_ids=["s0002"],
                        board_state_ids=[],
                    )
                ]
            ),
            "judge/keypoint_recall": KeypointJudgement(
                verdicts=[
                    KeypointVerdict(id="k1", covered=True, quote=explanation),
                    KeypointVerdict(id="k2", covered=False, quote=""),
                ]
            ),
        }
    )
    gateway = LLMGateway(llm.factory, default_model="gemini-2.5-flash", retry_backoff=False)

    scores, missing = await score_videos(settings, gateway, fixtures=fixtures, golden=golden)

    [score] = scores
    assert score.doc.units[0].concept == "enumerate()"
    assert score.keypoints.covered == ["k1"]
    assert score.keypoints.recall == 0.5
    assert [v.id for v in missing] == [MISSING]
    # Duration estimated from evidence when the fixture has no .meta.json.
    assert score.doc.videos[0].duration_s > 0

    report = render(scores, missing, settings)
    assert f"| {VID} | code | 1 | 50% (1/2) | 0 | 0 |" in report
    assert f"No perception fixture yet: {MISSING}" in report
    assert "- k2" in report
