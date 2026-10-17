"""Golden set: hand-made ground truth for measuring perception and notes quality.

Layout (see evals/golden/README.md):

    evals/golden/videos.yaml
    evals/golden/references/<video_id>/
        captions.vtt     real YouTube captions (downloaded once, dev only)
        board_text.md    hand-typed board/slide/code text with timestamps
        boundaries.yaml  times where the board/slide changed
        keypoints.yaml   facts the notes must contain
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from evals.vtt import vtt_text
from lecnotes.domain import BoardType
from lecnotes.timefmt import parse_timestamp

GOLDEN_DIR = Path(__file__).parent / "golden"
REFERENCE_FILES = ("captions.vtt", "board_text.md", "boundaries.yaml", "keypoints.yaml")

LectureType = Literal["slides", "whiteboard", "code", "mixed"]


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GoldenVideo(_Strict):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{11}$")
    title: str
    type: LectureType
    tags: list[str] = Field(default_factory=list, description="e.g. fast-talker, poor-audio")
    terms: list[str] = Field(
        default_factory=list, description="Technical terms spoken in the video (term WER)"
    )
    notes: str = ""

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.id}"


class _VideosFile(_Strict):
    videos: list[GoldenVideo]


class Keypoint(_Strict):
    id: str
    text: str
    at: float | None = None

    @field_validator("at", mode="before")
    @classmethod
    def _parse_at(cls, value: object) -> object:
        return parse_timestamp(value) if isinstance(value, str) else value


class _KeypointsFile(_Strict):
    keypoints: list[Keypoint] = Field(min_length=1)


class _BoundariesFile(_Strict):
    boundaries: list[float]

    @field_validator("boundaries", mode="before")
    @classmethod
    def _parse(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        return sorted(parse_timestamp(v) for v in value)


@dataclass(frozen=True)
class RefBoard:
    t_s: float
    type: BoardType
    lines: list[str]


@dataclass(frozen=True)
class References:
    video_id: str
    captions: str
    boards: list[RefBoard]
    boundaries: list[float]
    keypoints: list[Keypoint]


_BOARD_HEADER = re.compile(r"^##\s+(\S+)\s+(\w+)\s*$")
_COMMENT = re.compile(r"<!--.*?-->", re.S)


def parse_board_text(source: str) -> list[RefBoard]:
    """`## <MM:SS> <type>` headers, each followed by the verbatim lines visible at that time."""
    boards: list[RefBoard] = []
    current: tuple[float, BoardType] | None = None
    lines: list[str] = []

    def flush() -> None:
        if current is not None:
            boards.append(RefBoard(t_s=current[0], type=current[1], lines=lines.copy()))

    for raw in _COMMENT.sub("", source).splitlines():
        if header := _BOARD_HEADER.match(raw):
            flush()
            current = (parse_timestamp(header.group(1)), BoardType(header.group(2)))
            lines = []
        elif raw.startswith("#"):
            continue  # document title / free-form headings
        elif current is not None and raw.strip():
            lines.append(raw.rstrip())
    flush()
    return boards


def load_videos(root: Path = GOLDEN_DIR) -> list[GoldenVideo]:
    data = yaml.safe_load((root / "videos.yaml").read_text(encoding="utf-8")) or {}
    return _VideosFile.model_validate(data).videos


def missing_reference_files(video_id: str, root: Path = GOLDEN_DIR) -> list[str]:
    folder = root / "references" / video_id
    return [name for name in REFERENCE_FILES if not (folder / name).is_file()]


def load_references(video_id: str, root: Path = GOLDEN_DIR) -> References:
    folder = root / "references" / video_id
    if missing := missing_reference_files(video_id, root):
        raise FileNotFoundError(f"{folder}: missing {missing}")

    def read_yaml(name: str) -> object:
        return yaml.safe_load((folder / name).read_text(encoding="utf-8"))

    return References(
        video_id=video_id,
        captions=vtt_text(folder / "captions.vtt"),
        boards=parse_board_text((folder / "board_text.md").read_text(encoding="utf-8")),
        boundaries=_BoundariesFile.model_validate(read_yaml("boundaries.yaml")).boundaries,
        keypoints=_KeypointsFile.model_validate(read_yaml("keypoints.yaml")).keypoints,
    )
