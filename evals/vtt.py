"""Minimal WebVTT reader for YouTube captions (manual or auto-generated).

YouTube auto-captions are "rolling": each cue repeats the previous line and adds a new one,
and words carry inline timing tags like `<00:00:01.120><c> word</c>`. We strip tags and drop
a line when it repeats the line emitted just before it, which yields clean running text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from lecnotes.timefmt import parse_timestamp

_TIMING = re.compile(r"^\s*([\d:.]+)\s+-->\s+([\d:.]+)")
_TAG = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class Cue:
    start_s: float
    end_s: float
    text: str


def parse_vtt(source: str) -> list[Cue]:
    cues: list[Cue] = []
    last_line = ""
    lines = source.splitlines()
    i = 0
    while i < len(lines):
        match = _TIMING.match(lines[i])
        i += 1
        if not match:
            continue
        start, end = parse_timestamp(match.group(1)), parse_timestamp(match.group(2))
        new_lines: list[str] = []
        # Only a truly empty line ends a cue (WebVTT spec). YouTube auto-captions put a
        # single-space line right after the timing line; it must not end the cue.
        while i < len(lines) and lines[i] != "":
            line = " ".join(_TAG.sub("", lines[i]).split())
            if line and line != last_line:
                new_lines.append(line)
                last_line = line
            i += 1
        if new_lines:
            cues.append(Cue(start_s=start, end_s=end, text=" ".join(new_lines)))
    return cues


def vtt_text(path: Path) -> str:
    return " ".join(cue.text for cue in parse_vtt(path.read_text(encoding="utf-8")))
