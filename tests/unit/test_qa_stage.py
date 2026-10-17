from uuid import uuid4

import pytest
from pydantic import ValidationError

from lecnotes.adapters.llm.fake import FakeLLM
from lecnotes.domain import (
    Evidence,
    KnowledgeUnit,
    QaLevel,
    QaRequest,
    QuestionLevel,
    VideoMeta,
    VideoSpan,
)
from lecnotes.llm.gateway import LLMGateway
from lecnotes.pipeline.context import RunContext
from lecnotes.pipeline.stages.qa import QaItem, QaOutput, QaStage, to_question_set

VID = "Ek49cGiAOwo"
VIDEO = VideoMeta(video_id=VID, title="Zip and Enumerate", duration_s=347)


def unit(id: str, concept: str, *starts: float) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=id,
        concept=concept,
        explanation=f"About {concept}.",
        spans=[VideoSpan(video_id=VID, start_s=s, end_s=s + 20) for s in starts],
        evidence=[Evidence(video_id=VID, segment_ids=[f"s{int(starts[0])}"])],
    )


UNITS = [unit("u1", "zip()", 26, 120), unit("u2", "enumerate()", 209), unit("u3", "Lists", 5)]


def item(question: str, *ids: str, level: str = "recall") -> QaItem:
    return QaItem(level=level, question=question, answer="An answer.", unit_ids=list(ids))  # type: ignore[arg-type]


def request(level: QaLevel = QaLevel.MIXED, count: int = 5) -> QaRequest:
    return QaRequest(units=UNITS, level=level, count=count)


def test_questions_citing_no_existing_note_are_dropped() -> None:
    raw = QaOutput(questions=[item("What is zip?", "u1"), item("Made up?", "u9"), item("X?")])
    out = to_question_set(raw, request())
    assert [q.question for q in out.questions] == ["What is zip?"]
    assert len(out.dropped) == 2
    assert all("cites no existing note" in d for d in out.dropped)


def test_unknown_ids_are_filtered_but_known_ones_kept() -> None:
    out = to_question_set(QaOutput(questions=[item("Q?", "u9", "u2", "u2")]), request())
    assert out.questions[0].unit_ids == ["u2"]


def test_spans_are_the_cited_units_primary_moments_in_time_order() -> None:
    out = to_question_set(QaOutput(questions=[item("Both?", "u2", "u1", "u3")]), request())
    assert [s.start_s for s in out.questions[0].spans] == [5, 26, 209]  # u1's 120s not used


def test_duplicates_and_empty_questions_are_dropped() -> None:
    raw = QaOutput(
        questions=[
            item("What does zip do?", "u1"),
            item("  what does ZIP do? ", "u1"),
            item("   ", "u1"),
        ]
    )
    out = to_question_set(raw, request())
    assert len(out.questions) == 1
    assert ["duplicate" in d for d in out.dropped] == [True, False]


def test_requested_level_is_enforced_unless_mixed() -> None:
    raw = QaOutput(questions=[item("A?", "u1"), item("B?", "u2", level="application")])
    mixed = to_question_set(raw, request())
    assert [q.level for q in mixed.questions] == [QuestionLevel.RECALL, QuestionLevel.APPLICATION]
    forced = to_question_set(raw, request(QaLevel.APPLICATION))
    assert {q.level for q in forced.questions} == {QuestionLevel.APPLICATION}


def test_truncated_to_count_with_sequential_ids() -> None:
    raw = QaOutput(questions=[item(f"Q{i}?", "u1") for i in range(5)])
    out = to_question_set(raw, request(count=3))
    assert [q.id for q in out.questions] == ["q01", "q02", "q03"]


@pytest.mark.parametrize("count", [0, 11])
def test_count_is_bounded(count: int) -> None:
    with pytest.raises(ValidationError):
        request(count=count)


async def test_stage_prompts_with_the_notes_and_records_usage() -> None:
    fake = FakeLLM({"qa": QaOutput(questions=[item("What is zip?", "u1")])})
    stage = QaStage(
        LLMGateway(fake.factory, default_model="gemini-2.5-flash"), model="gemini-2.5-flash"
    )
    ctx = RunContext(job_id=uuid4(), video=VIDEO)
    out = await stage.run(request(QaLevel.RECALL, 2), ctx)

    assert [q.question for q in out.questions] == ["What is zip?"]
    (call,) = fake.calls
    prompt = "\n".join(str(m.content) for m in call.messages)
    assert "[u2] enumerate()" in prompt
    assert "Write 2 questions." in prompt
    assert "All questions at level: recall." in prompt
    assert [u.tokens_in for u in ctx.take_usage()] == [100]
