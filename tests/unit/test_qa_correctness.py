from evals.metrics.qa_correctness import QaJudgement, QuestionVerdict, qa_correctness
from lecnotes.adapters.llm.fake import FakeLLM
from lecnotes.domain import Evidence, KnowledgeUnit, Question, QuestionLevel, VideoSpan
from lecnotes.llm.gateway import LLMGateway

VID = "dQw4w9WgXcQ"
SPAN = VideoSpan(video_id=VID, start_s=0, end_s=5)
ZIP = KnowledgeUnit(
    id="u1",
    concept="zip()",
    explanation="zip pairs items of two lists.",
    spans=[SPAN],
    evidence=[Evidence(video_id=VID, segment_ids=["s1"])],
)
OTHER = ZIP.model_copy(update={"id": "u2", "concept": "Secret topic"})


def question(id: str) -> Question:
    return Question(
        id=id,
        level=QuestionLevel.RECALL,
        question=f"Question {id}?",
        answer="It pairs items.",
        unit_ids=["u1"],
        spans=[SPAN],
    )


def verdict(id: str, *, grounded: bool = True, correct: bool = True) -> QuestionVerdict:
    return QuestionVerdict(question_id=id, grounded=grounded, correct=correct, reason="r " + id)


async def test_passes_only_grounded_and_correct_and_reports_the_rest() -> None:
    judgement = QaJudgement(
        verdicts=[verdict("q01"), verdict("q02", grounded=False), verdict("q03", correct=False)]
    )
    fake = FakeLLM({"judge/qa_correctness": judgement})
    gateway = LLMGateway(fake.factory, default_model="gemini-2.5-flash")
    qs = [question(f"q0{i}") for i in range(1, 5)]

    score = await qa_correctness(gateway, qs, [ZIP, OTHER])

    assert (score.passed, score.total, score.score) == (1, 3, 1 / 3)
    assert score.failures == {"q02": "r q02", "q03": "r q03"}
    assert score.ungraded == ["q04"]
    prompt = "\n".join(str(m.content) for m in fake.calls[0].messages)
    assert "zip pairs items of two lists." in prompt
    assert "Secret topic" not in prompt  # the judge only sees the cited notes


async def test_no_questions_no_call() -> None:
    fake = FakeLLM({})
    score = await qa_correctness(LLMGateway(fake.factory, default_model="gemini-2.5-flash"), [], [])
    assert score.score is None
    assert fake.calls == []
