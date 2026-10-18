from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import BaseModel, ValidationError

from lecnotes.domain import (
    AlignedChunk,
    AlignmentLink,
    AlignmentMethod,
    BoardState,
    BoardType,
    CodeBlock,
    Evidence,
    Job,
    JobStatus,
    KnowledgeUnit,
    NotesDocument,
    PerceptionResult,
    StageRun,
    StageRunStatus,
    TranscriptSegment,
    VideoMeta,
    VideoRef,
    VideoSpan,
)
from lecnotes.domain.video import is_playlist_link, parse_video_id

VID = "dQw4w9WgXcQ"
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def seg(id: str = "s1", start: float = 0, end: float = 5, text: str = "hello") -> TranscriptSegment:
    return TranscriptSegment(id=id, start_s=start, end_s=end, text=text)


def board(id: str = "b1", first: float = 0, last: float = 10) -> BoardState:
    return BoardState(
        id=id, t_first_seen=first, t_last_seen=last, type=BoardType.CODE, text="print(1)"
    )


def unit(**overrides: object) -> KnowledgeUnit:
    fields: dict[str, object] = {
        "id": "u1",
        "concept": "enumerate",
        "explanation": "Yields (index, item) pairs.",
        "examples": ["for i, x in enumerate(xs): ..."],
        "code": [CodeBlock(source="enumerate(xs)", language="python")],
        "spans": [VideoSpan(video_id=VID, start_s=10, end_s=40)],
        "evidence": [Evidence(video_id=VID, segment_ids=["s1"], board_state_ids=["b1"])],
    }
    return KnowledgeUnit.model_validate(fields | overrides)


# --- valid objects -----------------------------------------------------------------------


def test_valid_objects_build() -> None:
    u = unit()
    assert (u.start_s, u.end_s) == (10, 40)
    assert u.video_ids == [VID]
    assert VideoRef(video_id=VID).url.endswith(VID)
    meta = VideoMeta(video_id=VID, title="Lecture 1", duration_s=600)
    assert meta.url == f"https://www.youtube.com/watch?v={VID}"


def test_board_state_single_frame_allowed() -> None:
    assert board(first=12, last=12).t_last_seen == 12


def test_board_state_is_empty() -> None:
    blank = BoardState(id="b", t_first_seen=0, t_last_seen=1, type=BoardType.SLIDE, text="  ")
    assert blank.is_empty
    assert not board().is_empty


def test_text_is_stripped() -> None:
    assert seg(text="  hi  ").text == "hi"


# --- invalid objects ---------------------------------------------------------------------


@pytest.mark.parametrize(("start", "end"), [(5, 5), (10, 5), (-1, 5)])
def test_segment_bad_times_rejected(start: float, end: float) -> None:
    with pytest.raises(ValidationError):
        seg(start=start, end=end)


def test_board_state_backwards_times_rejected() -> None:
    with pytest.raises(ValidationError, match="board state b1"):
        board(first=10, last=5)


@pytest.mark.parametrize("text", ["", "   "])
def test_segment_empty_text_rejected(text: str) -> None:
    with pytest.raises(ValidationError):
        seg(text=text)


def test_board_type_must_be_enum() -> None:
    with pytest.raises(ValidationError):
        BoardState(id="b", t_first_seen=0, t_last_seen=1, type="blackboard")  # type: ignore[arg-type]


def test_invalid_video_id_rejected() -> None:
    with pytest.raises(ValidationError):
        VideoRef(video_id="not-a-video-id")


def test_unknown_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        TranscriptSegment.model_validate(
            {"id": "s", "start_s": 0, "end_s": 1, "text": "x", "speaker": "prof"}
        )


def test_models_are_immutable() -> None:
    s = seg()
    with pytest.raises(ValidationError):
        s.text = "changed"  # type: ignore[misc]


def test_perception_duplicate_ids_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate evidence ids"):
        PerceptionResult(video_id=VID, segments=[seg("x")], board_states=[board("x")])


def test_unit_without_evidence_rejected() -> None:
    with pytest.raises(ValidationError):
        unit(evidence=[])
    with pytest.raises(ValidationError, match="at least one"):
        Evidence(video_id=VID)


def test_unit_without_span_rejected() -> None:
    with pytest.raises(ValidationError):
        unit(spans=[])


def test_alignment_score_bounded() -> None:
    with pytest.raises(ValidationError):
        AlignmentLink(segment_id="s1", board_state_id="b1", method=AlignmentMethod.LEXICAL, score=2)


def test_chunk_links_must_reference_contained_ids() -> None:
    link = AlignmentLink(
        segment_id="s1", board_state_id="missing", method=AlignmentMethod.TEMPORAL, score=1
    )
    with pytest.raises(ValidationError, match="missing ids"):
        AlignedChunk(
            id="c1",
            video_id=VID,
            start_s=0,
            end_s=120,
            segments=[seg()],
            board_states=[board()],
            links=[link],
        )


def test_failed_job_requires_error() -> None:
    with pytest.raises(ValidationError, match="must carry an error"):
        Job(id=uuid4(), url="u", status=JobStatus.FAILED, created_at=NOW, updated_at=NOW)
    with pytest.raises(ValidationError, match="must not carry"):
        Job(id=uuid4(), url="u", error="x", created_at=NOW, updated_at=NOW)


def test_stage_run_error_type_iff_failed() -> None:
    base = {"job_id": uuid4(), "stage": "extract", "stage_version": 1, "latency_ms": 5}
    with pytest.raises(ValidationError):
        StageRun.model_validate(base | {"status": StageRunStatus.FAILED, "started_at": NOW})
    with pytest.raises(ValidationError):
        StageRun.model_validate(
            base | {"status": StageRunStatus.OK, "error_type": "X", "started_at": NOW}
        )


# --- JSON round-trip ---------------------------------------------------------------------


def _round_trip_cases() -> list[BaseModel]:
    s, b = seg(), board()
    link = AlignmentLink(segment_id="s1", board_state_id="b1", method="lexical", score=0.8)  # type: ignore[arg-type]
    meta = VideoMeta(video_id=VID, title="L1", duration_s=600)
    return [
        PerceptionResult(video_id=VID, segments=[s], board_states=[b]),
        AlignedChunk(
            id="c1",
            video_id=VID,
            start_s=0,
            end_s=120,
            segments=[s],
            board_states=[b],
            links=[link],
        ),
        unit(),
        NotesDocument(title="Notes", video=meta, units=[unit()], markdown="# Notes"),
        Job(id=uuid4(), url="https://youtu.be/x", created_at=NOW, updated_at=NOW),
        StageRun(
            job_id=uuid4(),
            stage="perceive",
            stage_version=1,
            status=StageRunStatus.OK,
            latency_ms=1200,
            started_at=NOW,
        ),
    ]


@pytest.mark.parametrize("obj", _round_trip_cases(), ids=lambda o: type(o).__name__)
def test_json_round_trip(obj: BaseModel) -> None:
    assert type(obj).model_validate_json(obj.model_dump_json()) == obj


# --- video links -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "link",
    [
        VID,
        f"https://www.youtube.com/watch?v={VID}",
        f"https://youtube.com/watch?v={VID}&list=PL123&t=42s",
        f"https://m.youtube.com/watch?feature=share&v={VID}",
        f"https://youtu.be/{VID}?si=abc",
        f"youtu.be/{VID}",
        f"https://www.youtube.com/shorts/{VID}",
        f"https://www.youtube.com/embed/{VID}",
        f"https://www.youtube.com/live/{VID}?feature=share",
        f"https://www.youtube-nocookie.com/embed/{VID}",
    ],
)
def test_video_ref_from_url(link: str) -> None:
    assert VideoRef.from_url(link).video_id == VID


@pytest.mark.parametrize(
    "link",
    [
        "",
        "https://vimeo.com/123456",
        "https://www.youtube.com/watch?v=short",
        "https://www.youtube.com/playlist?list=PL123",
        f"https://evil.com/watch?v={VID}",
    ],
)
def test_video_ref_rejects_non_video_links(link: str) -> None:
    assert parse_video_id(link) is None
    with pytest.raises(ValueError, match="not a YouTube video link"):
        VideoRef.from_url(link)


@pytest.mark.parametrize(
    ("link", "expected"),
    [
        ("https://www.youtube.com/playlist?list=PL123", True),
        ("youtube.com/playlist?list=PL123", True),
        (f"https://www.youtube.com/watch?v={VID}&list=PL123", False),  # that's a video link
        ("https://vimeo.com/playlist", False),
    ],
)
def test_is_playlist_link(link: str, expected: bool) -> None:
    assert is_playlist_link(link) is expected
