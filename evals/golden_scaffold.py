"""Scaffolding for new golden-set entries (used by scripts/new_golden_video.py).

Skeletons are deliberately empty: ground truth must be typed by a human watching the video.
Pre-filling it from the model's output would anchor the annotator to the model's mistakes and
make the metrics circular. New entries are drafts, which are never scored.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from evals.golden_set import LectureType, load_videos
from lecnotes.domain import VideoMeta

BOARD_TEXT = """# Board text — {title}

<!--
Watch the video and add one section per board/slide/screen state, in time order:

## <MM:SS when it first appears> <slide|whiteboard|code|screen|other>
<the visible text, copied VERBATIM: keep code indentation and symbols>

Write [unclear] where you cannot read it yourself. Skip decorative text.
-->
"""

BOUNDARIES = """# Times (MM:SS) where visible content changed (new slide, writing, erase, scroll).
# Do not include 00:00. Example:
#   - "01:23"
boundaries: []
"""

KEYPOINTS = """# 10-20 facts the notes MUST contain, in your words. `at` (optional) = when taught.
# Example:
#   - id: k1
#     text: "enumerate() yields (index, item) pairs"
#     at: "03:29"
keypoints: []
"""


def create_skeleton(
    root: Path,
    meta: VideoMeta,
    lecture_type: LectureType,
    *,
    terms: list[str],
    tags: list[str],
) -> Path:
    if any(v.id == meta.video_id for v in load_videos(root, include_drafts=True)):
        raise SystemExit(f"{meta.video_id} is already in {root / 'videos.yaml'}")
    folder = root / "references" / meta.video_id
    if folder.exists():
        raise SystemExit(f"{folder} already exists")
    folder.mkdir(parents=True)
    (folder / "board_text.md").write_text(BOARD_TEXT.format(title=meta.title), encoding="utf-8")
    (folder / "boundaries.yaml").write_text(BOUNDARIES, encoding="utf-8")
    (folder / "keypoints.yaml").write_text(KEYPOINTS, encoding="utf-8")

    title = meta.title.replace('"', "'")
    entry = (
        f'  - id: {meta.video_id}\n    title: "{title}"\n    type: {lecture_type}\n'
        f"    tags: [{', '.join(tags)}]\n    terms: [{', '.join(terms)}]\n"
        f"    draft: true  # flip to false once all 4 reference files are written\n"
    )
    videos = root / "videos.yaml"
    text = videos.read_text(encoding="utf-8")
    if "\nvideos: []" in text:
        text = text.replace("\nvideos: []", "\nvideos:\n" + entry.rstrip("\n"))
    else:
        text = text.rstrip("\n") + "\n" + entry
    videos.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    return folder


def download_captions(url: str, folder: Path) -> Path | None:
    """yt-dlp via uvx: prefers manual subtitles, falls back to auto-generated."""
    if shutil.which("uvx") is None:
        print("uvx not found; skipping captions", file=sys.stderr)
        return None
    subprocess.run(
        ["uvx", "yt-dlp", "--skip-download", "--write-subs", "--write-auto-subs",
         "--sub-langs", "en.*,en", "--sub-format", "vtt", "-o", str(folder / "captions"), url],
        check=False,
    )  # fmt: skip
    found = sorted(folder.glob("captions*.vtt"))
    if not found:
        return None
    target = folder / "captions.vtt"
    found[0].rename(target)
    for extra in found[1:]:
        extra.unlink()
    return target
