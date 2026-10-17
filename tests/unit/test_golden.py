from pathlib import Path
from textwrap import dedent

import pytest
from pydantic import ValidationError

from evals.golden_set import (
    GOLDEN_DIR,
    load_references,
    load_videos,
    missing_reference_files,
    parse_board_text,
)
from evals.vtt import parse_vtt
from lecnotes.domain import BoardType

# Built line by line: dedent() would turn YouTube's single-space lines into empty ones.
YOUTUBE_AUTO_VTT = "\n".join(
    [
        "WEBVTT",
        "Kind: captions",
        "Language: en",
        "",
        "00:00:00.160 --> 00:00:02.270 align:start position:0%",
        " ",
        "today<00:00:00.480><c> we</c><00:00:00.640><c> look</c>",
        "",
        "00:00:02.270 --> 00:00:02.280 align:start position:0%",
        "today we look",
        " ",
        "",
        "00:00:02.280 --> 00:00:04.990 align:start position:0%",
        "today we look",
        "at<00:00:02.560><c> enumerate</c>",
        "",
    ]
)


def test_vtt_rolling_auto_captions_are_deduplicated() -> None:
    cues = parse_vtt(YOUTUBE_AUTO_VTT)
    assert " ".join(c.text for c in cues) == "today we look at enumerate"
    assert cues[0].start_s == pytest.approx(0.16)


def test_board_text_parsing() -> None:
    boards = parse_board_text(
        dedent("""\
        # Title line is ignored
        <!-- comments are ignored
        ## 99:99 bogus -->
        ## 00:00 slide
        Iteration

        ## 01:05 code
        for i, x in enumerate(xs):
            print(i, x)
        """)
    )
    assert [(b.t_s, b.type) for b in boards] == [(0, BoardType.SLIDE), (65, BoardType.CODE)]
    assert boards[1].lines == ["for i, x in enumerate(xs):", "    print(i, x)"]


def test_template_is_valid() -> None:
    """The template documents the format, so it must itself parse."""
    refs = load_references("_template")
    assert refs.captions.startswith("Today we look at enumerate.")
    assert refs.boundaries == [42.0]
    assert refs.keypoints[0].at == 42.0
    assert [b.type for b in refs.boards] == [BoardType.SLIDE, BoardType.CODE]


def test_every_golden_video_has_all_reference_files() -> None:
    """Step 3.2 'Done when': each listed video has all 4 files, and they all parse."""
    for video in load_videos():
        assert missing_reference_files(video.id) == [], video.id
        load_references(video.id)


def test_draft_videos_are_excluded_until_finished(tmp_path: Path) -> None:
    (tmp_path / "videos.yaml").write_text(
        "videos:\n"
        "  - {id: aaaaaaaaaaa, title: Done, type: code}\n"
        "  - {id: bbbbbbbbbbb, title: WIP, type: slides, draft: true}\n",
        encoding="utf-8",
    )
    assert [v.id for v in load_videos(tmp_path)] == ["aaaaaaaaaaa"]
    assert len(load_videos(tmp_path, include_drafts=True)) == 2


def test_videos_file_rejects_bad_entries(tmp_path: Path) -> None:
    (tmp_path / "videos.yaml").write_text(
        "videos:\n  - id: short\n    title: x\n    type: podcast\n", encoding="utf-8"
    )
    with pytest.raises(ValidationError):
        load_videos(tmp_path)


def test_missing_reference_files_listed(tmp_path: Path) -> None:
    (tmp_path / "references" / "abcdefghijk").mkdir(parents=True)
    (tmp_path / "references" / "abcdefghijk" / "captions.vtt").write_text("WEBVTT\n")
    assert missing_reference_files("abcdefghijk", tmp_path) == [
        "board_text.md",
        "boundaries.yaml",
        "keypoints.yaml",
    ]
    with pytest.raises(FileNotFoundError):
        load_references("abcdefghijk", tmp_path)


def test_golden_dir_points_at_repo_data() -> None:
    assert (GOLDEN_DIR / "videos.yaml").is_file()
