from uuid import uuid4

import pytest
from pydantic import ValidationError

from lecnotes.adapters.cache.memory import InMemoryStageCache
from lecnotes.domain import (
    CodeBlock,
    Evidence,
    KnowledgeUnit,
    NotesDocument,
    UnitEdit,
    VideoMeta,
    VideoSpan,
)
from lecnotes.observability.metrics import InMemoryStageRunSink
from lecnotes.pipeline.notes_editor import NotesEditor, UnitNotFound
from lecnotes.pipeline.orchestrator import Orchestrator
from lecnotes.pipeline.stages.render import RenderStage, render_markdown

VID = "dQw4w9WgXcQ"
VIDEO = VideoMeta(video_id=VID, title="Python Iteration Tools", duration_s=754)


def unit(id: str, concept: str, start: float) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=id,
        concept=concept,
        explanation=f"About {concept}.",
        board_text="zip(a, b)",
        section="Iteration",
        spans=[VideoSpan(video_id=VID, start_s=start, end_s=start + 30)],
        evidence=[Evidence(video_id=VID, segment_ids=["s1"])],
    )


UNITS = [unit("u1", "zip()", 10), unit("u2", "enumrate()", 60)]
DOC = NotesDocument(
    title=VIDEO.title, video=VIDEO, units=UNITS, markdown=render_markdown(VIDEO, UNITS)
)


def test_edit_changes_only_the_given_fields() -> None:
    edited = UnitEdit(concept="enumerate()").apply(UNITS[1])
    assert edited.concept == "enumerate()"
    assert edited.model_dump(exclude={"concept"}) == UNITS[1].model_dump(exclude={"concept"})


def test_blank_board_text_and_section_clear_them() -> None:
    edited = UnitEdit(board_text="  ", section="").apply(UNITS[0])
    assert (edited.board_text, edited.section) == (None, None)


def test_code_can_be_replaced() -> None:
    code = [CodeBlock(source="list(zip(a, b))", language="python")]
    assert UnitEdit(code=code).apply(UNITS[0]).code == code


@pytest.mark.parametrize(
    "body",
    [
        {},  # nothing to change
        {"concept": ""},  # a unit needs a name
        {"explanation": "   "},
        {"concept": None},  # null would erase a required field
        {"examples": None},
        {"spans": []},  # provenance is not editable
        {"evidence": []},
    ],
)
def test_invalid_edits_are_rejected(body: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        UnitEdit.model_validate(body)


async def test_editor_rerenders_through_the_orchestrator_without_ai() -> None:
    runs = InMemoryStageRunSink()
    editor = NotesEditor(Orchestrator(InMemoryStageCache(), runs), RenderStage())
    job_id = uuid4()

    doc = await editor.edit(
        job_id=job_id, doc=DOC, unit_id="u2", edit=UnitEdit(concept="enumerate()")
    )

    assert "### enumerate()\n" in doc.markdown
    assert "### enumrate()" not in doc.markdown
    assert [u.concept for u in doc.units] == ["zip()", "enumerate()"]
    assert doc.markdown == render_markdown(VIDEO, doc.units)
    (run,) = runs.runs
    assert (run.stage, run.job_id, run.model, run.tokens_in) == ("render", job_id, None, 0)


async def test_unknown_unit() -> None:
    editor = NotesEditor(Orchestrator(InMemoryStageCache(), InMemoryStageRunSink()), RenderStage())
    with pytest.raises(UnitNotFound):
        await editor.edit(job_id=uuid4(), doc=DOC, unit_id="nope", edit=UnitEdit(concept="x"))
