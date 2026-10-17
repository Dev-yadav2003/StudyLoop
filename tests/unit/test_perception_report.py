import json
import shutil
from pathlib import Path

from evals.golden_set import GOLDEN_DIR
from evals.perception_report import build_report
from lecnotes.domain import BoardState, BoardType, PerceptionResult, TranscriptSegment

VID, MISSING = "abcdefghijk", "zzzzzzzzzzz"


def make_golden(root: Path) -> Path:
    golden = root / "golden"
    for vid in (VID, MISSING):
        shutil.copytree(GOLDEN_DIR / "references" / "_template", golden / "references" / vid)
    (golden / "videos.yaml").write_text(
        "videos:\n"
        f"  - {{id: {VID}, title: Iteration, type: code, terms: [enumerate]}}\n"
        f"  - {{id: {MISSING}, title: Not run yet, type: slides}}\n",
        encoding="utf-8",
    )
    return golden


def make_fixture(root: Path) -> Path:
    fixtures = root / "fixtures"
    fixtures.mkdir()
    result = PerceptionResult(
        video_id=VID,
        segments=[
            TranscriptSegment(id="s1", start_s=0, end_s=3, text="Today we look at a numerate."),
            TranscriptSegment(
                id="s2", start_s=3, end_s=6.5, text="It gives you the index and the item."
            ),
        ],
        board_states=[
            BoardState(
                id="b1",
                t_first_seen=0,
                t_last_seen=40,
                type=BoardType.SLIDE,
                text="Python Iteration Tools",
            ),
            BoardState(
                id="b2",
                t_first_seen=44,
                t_last_seen=90,
                type=BoardType.CODE,
                text="for i, name in enumerate(names):\n    print(i, name)",
            ),
        ],
    )
    (fixtures / f"perception_{VID}.json").write_text(result.model_dump_json(), encoding="utf-8")
    meta = {
        "fingerprint": {
            "model": "gemini-2.5-flash",
            "prompt_version": 1,
            "fps": 1.0,
            "media_resolution": "low",
        }
    }
    (fixtures / f"perception_{VID}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return fixtures


def test_report_scores_fixture_against_golden(tmp_path: Path) -> None:
    report = build_report(make_fixture(tmp_path), make_golden(tmp_path))

    [row] = [line for line in report.splitlines() if line.startswith(f"| {VID}")]
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[1] == "code"
    assert cells[2] == "gemini-2.5-flash p1 1.0fps low"
    assert cells[3] == "15%"  # 2 errors ("a numerate" vs "enumerate") / 13 reference words
    assert cells[4] == "100%"  # the only term occurrence was mangled
    assert cells[5] == "100% (3/3)"
    assert cells[6] == "1.00 / 1.00 / 1.00"  # 44 vs 42 within tolerance
    assert cells[7] == "2.0s"

    assert f"No fixture yet: {MISSING}" in report
    assert "missed terms: enumerate" in report
