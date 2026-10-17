"""Knowledge units → Markdown notes (no AI). Deterministic: same units → byte-identical output.

Sections are ordered by when they are first taught, units by time. Every concept links to
the moment it was taught (`https://youtu.be/<id>?t=<s>`), one link per span, so merged
playlist units (Phase 7) list every occurrence.
"""

from __future__ import annotations

import re
from collections import defaultdict

from lecnotes.domain import ExtractedUnits, KnowledgeUnit, NotesDocument, VideoMeta, VideoSpan
from lecnotes.pipeline.context import RunContext
from lecnotes.pipeline.stage import Stage
from lecnotes.timefmt import format_timestamp

DEFAULT_SECTION = "Notes"
_BACKTICKS = re.compile(r"`+")


def fence(content: str, language: str = "") -> str:
    """A code fence longer than any backtick run inside the content, so it can't break out."""
    longest = max((len(m) for m in _BACKTICKS.findall(content)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}{language}\n{content.rstrip()}\n{ticks}"


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _link(span: VideoSpan) -> str:
    return f"[{format_timestamp(span.start_s)}](https://youtu.be/{span.video_id}?t={int(span.start_s)})"


def _unit(unit: KnowledgeUnit) -> str:
    links = " · ".join(_link(s) for s in unit.spans)
    parts = [f"### {_one_line(unit.concept)} · {links}", unit.explanation.strip()]
    if unit.board_text:
        parts += ["**On the board:**", fence(unit.board_text, "text")]
    parts += [fence(c.source, c.language or "") for c in unit.code]
    if unit.examples:
        parts += ["**Examples:**", "\n".join(f"- {_one_line(e)}" for e in unit.examples)]
    return "\n\n".join(parts)


def render_markdown(video: VideoMeta, units: list[KnowledgeUnit]) -> str:
    header = [f"# {_one_line(video.title)}"]
    source = [f"[{_one_line(video.title)}]({video.url})", format_timestamp(video.duration_s)]
    if video.channel:
        source.append(_one_line(video.channel))
    header.append("*Source:* " + " · ".join(source))

    if not units:
        return "\n\n".join([*header, "_No notes could be extracted from this video._"]) + "\n"

    sections: dict[str, list[KnowledgeUnit]] = defaultdict(list)
    for unit in sorted(units, key=lambda u: (u.start_s, u.id)):
        sections[_one_line(unit.section or DEFAULT_SECTION)].append(unit)
    # dicts keep insertion order: sections appear in the order they were first taught

    body = []
    for title, members in sections.items():
        body.append(f"## {title}")
        body += [_unit(u) for u in members]
    return "\n\n".join([*header, *body]) + "\n"


class RenderStage(Stage[ExtractedUnits, NotesDocument]):
    name = "render"
    version = 1
    input_model = ExtractedUnits
    output_model = NotesDocument

    async def run(self, inp: ExtractedUnits, ctx: RunContext) -> NotesDocument:
        video = ctx.require_video()
        return NotesDocument(
            title=video.title,
            videos=[video],
            units=inp.units,
            markdown=render_markdown(video, inp.units),
        )
