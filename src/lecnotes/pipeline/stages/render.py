"""Knowledge units → Markdown notes (no AI). Deterministic: same units → byte-identical output.

Sections are numbered and ordered by when they are first taught, units by time. Under every
concept, a line links to the moment it was taught (`https://youtu.be/<id>?t=<s>`): the
primary moment first, then "also …" for every other moment it came up — nothing is dropped.

Board text is never rendered: it is evidence for extraction and checking, and what is worth
learning from it is already written into the explanation (extract prompt v2+). Explanations
are Markdown; headings inside them are demoted to bold so they can't break the outline.
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
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
_LIST_ITEM = re.compile(r"^\s{0,3}(?:[-*+]|\d{1,3}[.)])\s+")
# A list glued into one line by the model: "steps:1.  **Define…**2.  **Build…**" or
# "options: *   **Automated** — …. *   **Human** — …". Split before a list marker that
# follows sentence punctuation and introduces a bold term.
_GLUED_ITEM = re.compile(r"(?<=[:.!?)])\s*(?=(?:[-*+]|\d{1,2}\.)\s{1,4}\*\*\S)")


def fence(content: str, language: str = "") -> str:
    """A code fence longer than any backtick run inside the content, so it can't break out."""
    longest = max((len(m) for m in _BACKTICKS.findall(content)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}{language}\n{content.rstrip()}\n{ticks}"


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _link(span: VideoSpan) -> str:
    return f"[{format_timestamp(span.start_s)}](https://youtu.be/{span.video_id}?t={int(span.start_s)})"


def _explanation(text: str) -> str:
    """The unit's Markdown, made safe to embed: heading lines become bold lines, and lists are
    separated from surrounding paragraphs by a blank line — otherwise a sentence written right
    after a list is swallowed into its last item (lazy continuation)."""
    lines: list[str] = []
    in_code = in_list = False
    for raw in _unglue_lists(text.strip()).splitlines():
        line = raw.rstrip()
        if line.lstrip().startswith(("```", "~~~")):
            in_code = not in_code
            lines.append(line)
            continue
        if in_code:
            lines.append(line)
            continue
        if not line:
            in_list = False
            lines.append(line)
            continue
        heading = _HEADING.match(line)
        if heading:
            line = f"**{heading.group(1)}**"
        is_item = bool(_LIST_ITEM.match(line))
        continues_item = in_list and line[:1].isspace()
        if lines and lines[-1] and is_item != in_list and not continues_item:
            lines.append("")  # entering or leaving a list
        in_list = is_item or continues_item
        lines.append(line)
    return "\n".join(lines)


def _unglue_lists(text: str) -> str:
    """Put list items the model glued into one line back on their own lines (outside code)."""
    out: list[str] = []
    in_code = False
    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_code = not in_code
        if in_code or len(_GLUED_ITEM.findall(line)) < 2:  # one match is likely prose
            out.append(line)
        else:
            out += [part.strip() for part in _GLUED_ITEM.split(line) if part.strip()]
    return "\n".join(out)


def _unit(unit: KnowledgeUnit) -> str:
    primary, *others = unit.spans  # primary = where it is taught (see primary_first)
    links = f"▶ {_link(primary)}"
    if others:
        links += " · also " + ", ".join(_link(s) for s in others)
    parts = [
        f"### {_one_line(unit.concept)}",
        f"*{links}*",
        _explanation(unit.explanation),
    ]
    parts += [fence(c.source, c.language or "") for c in unit.code]
    if len(unit.examples) == 1:
        parts.append(f"**Example:** {_one_line(unit.examples[0])}")
    elif unit.examples:
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
    for number, (title, members) in enumerate(sections.items(), start=1):
        body.append(f"## {number}. {title}")
        body += [_unit(u) for u in members]
    return "\n\n".join([*header, *body]) + "\n"


class RenderStage(Stage[ExtractedUnits, NotesDocument]):
    name = "render"
    version = 7  # v2: + evidence/verify; v3: single `video`; v4: primary link + (also …)
    # v5: no board text, numbered sections, timestamps on their own line, Markdown-safe
    # v6: lists separated from paragraphs, glued lists split
    # v7: timestamp line is plain Markdown (italic), not <sub> HTML
    input_model = ExtractedUnits
    output_model = NotesDocument

    async def run(self, inp: ExtractedUnits, ctx: RunContext) -> NotesDocument:
        video = ctx.require_video()
        return NotesDocument(
            title=video.title,
            video=video,
            units=inp.units,
            markdown=render_markdown(video, inp.units),
            evidence=inp.evidence,
            verify=inp.verify,
        )
