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
    order = [md.index(s) for s in ("## 1. Iteration tools", "## 2. Notes", "## 3. Wrap-up")]
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


def test_primary_link_first_then_other_moments() -> None:
    taught = unit("u1", 209, "enumerate()", "Tools").model_copy(
        update={
            "spans": [
                VideoSpan(video_id=VID, start_s=209, end_s=254),
                VideoSpan(video_id=VID, start_s=26, end_s=31),
            ]
        }
    )
    md = render_markdown(VIDEO, [taught])
    assert (
        f"### enumerate()\n\n*▶ [03:29](https://youtu.be/{VID}?t=209) "
        f"· also [00:26](https://youtu.be/{VID}?t=26)*"
    ) in md


def test_board_text_is_evidence_not_notes() -> None:
    md = render_markdown(VIDEO, UNITS)
    assert "enumerate(iterable, start=0)" not in md  # u1's board_text
    assert "On the board" not in md


def test_lists_are_separated_from_paragraphs() -> None:
    """Real v2 output: the sentence after the list was swallowed into its last bullet."""
    text = (
        "There are two main types:\n*   **Model Evals**\n*   **Application Evals**\n"
        "Model Evals test the model.\n1. Define\n   still step one\n2. Build\nDone."
    )
    md = render_markdown(VIDEO, [unit("u1", 0, "LLM evals", "Evals", explanation=text)])
    assert (
        "There are two main types:\n\n*   **Model Evals**\n*   **Application Evals**\n\n"
        "Model Evals test the model.\n\n1. Define\n   still step one\n2. Build\n\nDone."
    ) in md


@pytest.mark.parametrize(
    ("glued", "expected"),
    [
        (  # real merge-v3 output
            "The workflow:1.  **Define Task**: what to test.2.  **Build a Dataset**: rows.",
            "The workflow:\n\n1.  **Define Task**: what to test.\n2.  **Build a Dataset**: rows.",
        ),
        (
            "Three options: *   **Automated** — a script. *   **Human** — a person.",
            "Three options:\n\n*   **Automated** — a script.\n*   **Human** — a person.",
        ),
    ],
)
def test_lists_glued_into_one_line_are_split(glued: str, expected: str) -> None:
    md = render_markdown(VIDEO, [unit("u1", 0, "Workflow", "W", explanation=glued)])
    assert expected in md


def test_prose_with_one_bold_term_after_a_period_is_not_split() -> None:
    text = "Use it carefully. 2 **Notes** follow below."
    md = render_markdown(VIDEO, [unit("u1", 0, "C", "W", explanation=text)])
    assert text in md


def test_headings_inside_explanations_cannot_break_the_outline() -> None:
    note = unit(
        "u1",
        10,
        "Types of LLM evals",
        "Evals",
        explanation="Two kinds.\n\n## Model evals\n- **Model evals** — test the model\n"
        "```python\n# a comment, not a heading\n```",
    )
    md = render_markdown(VIDEO, [note])
    assert "**Model evals**\n\n- **Model evals** — test the model" in md
    assert "# a comment, not a heading" in md  # code untouched
    assert "\n## Model evals" not in md


def test_code_output_is_shown_under_its_code() -> None:
    block = CodeBlock(source="print(1 + 1)", language="python", output="2")
    md = render_markdown(VIDEO, [unit("u1", 0, "Adding", "Basics", code=[block])])
    assert "```python\nprint(1 + 1)\n```\n\n**Output:**\n\n```text\n2\n```" in md
