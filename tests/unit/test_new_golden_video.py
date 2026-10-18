import shutil
from pathlib import Path

import pytest

from evals.golden_scaffold import create_skeleton
from evals.golden_set import GOLDEN_DIR, load_videos
from lecnotes.domain import VideoMeta

META = VideoMeta(video_id="Ek49cGiAOwo", title='Zip "and" Enumerate', duration_s=347)


@pytest.fixture
def golden(tmp_path: Path) -> Path:
    root = tmp_path / "golden"
    root.mkdir()
    shutil.copy(GOLDEN_DIR / "videos.yaml", root / "videos.yaml")  # the real, empty file
    return root


def test_skeleton_is_a_draft_with_no_ground_truth(golden: Path) -> None:
    folder = create_skeleton(golden, META, "code", terms=["zip", "enumerate"], tags=[])

    assert sorted(p.name for p in folder.iterdir()) == [
        "board_text.md",
        "boundaries.yaml",
        "keypoints.yaml",
    ]
    assert load_videos(golden) == []  # drafts are never scored
    [draft] = load_videos(golden, include_drafts=True)
    assert (draft.id, draft.type, draft.terms, draft.draft) == (
        "Ek49cGiAOwo",
        "code",
        ["zip", "enumerate"],
        True,
    )
    assert draft.title == "Zip 'and' Enumerate"
    assert "keypoints: []" in (folder / "keypoints.yaml").read_text(encoding="utf-8")


def test_second_video_appends_and_duplicates_are_refused(golden: Path) -> None:
    create_skeleton(golden, META, "code", terms=[], tags=[])
    other = META.model_copy(update={"video_id": "aaaaaaaaaaa", "title": "Other"})
    create_skeleton(golden, other, "slides", terms=[], tags=["fast-talker"])

    ids = [v.id for v in load_videos(golden, include_drafts=True)]
    assert ids == ["Ek49cGiAOwo", "aaaaaaaaaaa"]
    with pytest.raises(SystemExit, match="already"):
        create_skeleton(golden, META, "code", terms=[], tags=[])
