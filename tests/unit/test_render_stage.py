import os
from pathlib import Path
from uuid import uuid4

import pytest

from lecnotes.domain import (
    CodeBlock,
    Evidence,
    ExtractedUnits,
    KnowledgeUnit,
    VideoMeta,
    VideoSpan,
)
from lecnotes.pipeline.context import RunContext
from lecnotes.pipeline.stages.render import RenderStage, fence, render_markdown

SNAPSHOTS = Path(__file__).parents[1] / "fixtures" / "snapshots"
VID = "dQw4w9WgXcQ"
VIDEO = VideoMeta(video_id=VID, title="Python Iteration Tools", duration_s=754, channel="Prof X")


def unit(
    id: str, start: float, concept: str, section: str | None, **extra: object
) -> KnowledgeUnit:
    fields: dict[str, object] = {
        "id": id,
        "concept": concept,
        "explanation": f"About {concept}.",
        "section": section,
        "spans": [VideoSpan(video_id=VID, start_s=start, end_s=start + 30)],
        "evidence": [Evidence(video_id=VID, segment_ids=["s1"])],
    }
    return KnowledgeUnit.model_validate(fields | extra)


UNITS = [
    unit(
        "u3",
        400,
        "zip()",
        "Iteration tools",
        code=[CodeBlock(source="for a, b in zip(xs, ys):\n    print(a, b)", language="python")],
        examples=["zip([1, 2], 'ab') gives (1, 'a'), (2, 'b')"],
    ),
    unit(
        "u1",
        42,
        "enumerate()",
        "Iteration tools",
        explanation="Yields (index, item) pairs.\nThe index starts at zero.",
        board_text="enumerate(iterable, start=0)",
        code=[CodeBlock(source="for i, name in enumerate(names):\n    print(i, name)")],
    ),
    unit("u2", 200, "Why loops?", None),
    unit("u4", 3725, "Late topic", "Wrap-up"),
]


def check_snapshot(name: str, actual: str) -> None:
    path = SNAPSHOTS / name
    if os.environ.get("UPDATE_SNAPSHOTS") or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual, encoding="utf-8", newline="\n")
        if not os.environ.get("UPDATE_SNAPSHOTS"):
            pytest.fail(f"snapshot {name} created; review it and re-run")
    assert actual == path.read_text(encoding="utf-8")


def test_render_snapshot() -> None:
    check_snapshot("render_basic.md", render_markdown(VIDEO, UNITS))


def test_sections_ordered_by_first_appearance_units_by_time() -> None:
    md = render_markdown(VIDEO, UNITS)
    order = [md.index(s) for s in ("## Iteration tools", "## Notes", "## Wrap-up")]
    assert order == sorted(order)
    assert md.index("### enumerate()") < md.index("### zip()")


def test_timestamp_links() -> None:
    md = render_markdown(VIDEO, UNITS)
    assert f"[00:42](https://youtu.be/{VID}?t=42)" in md
    assert f"[1:02:05](https://youtu.be/{VID}?t=3725)" in md


def test_every_span_gets_a_link() -> None:
    merged = unit("u9", 10, "Recursion", "Basics").model_copy(
        update={
            "spans": [
                VideoSpan(video_id=VID, start_s=10, end_s=20),
                VideoSpan(video_id="aaaaaaaaaaa", start_s=90, end_s=99),
            ]
        }
    )
    md = render_markdown(VIDEO, [merged])
    assert "https://youtu.be/aaaaaaaaaaa?t=90" in md


def test_fence_survives_backticks_in_content() -> None:
    content = 'print("```")'
    fenced = fence(content, "python")
    assert fenced.startswith("````python\n")
    assert fenced.endswith("\n````")


def test_no_units_message() -> None:
    assert "_No notes could be extracted" in render_markdown(VIDEO, [])


async def test_render_stage_builds_notes_document() -> None:
    ctx = RunContext(job_id=uuid4(), video=VIDEO)
    doc = await RenderStage().run(ExtractedUnits(units=UNITS), ctx)
    assert doc.title == VIDEO.title
    assert doc.video == VIDEO
    assert doc.units == UNITS
    assert doc.markdown == render_markdown(VIDEO, UNITS)
