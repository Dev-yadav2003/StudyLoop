from uuid import uuid4

import pytest

from lecnotes.domain import (
    AlignmentLink,
    AlignmentMethod,
    BoardState,
    BoardType,
    PerceptionResult,
    TranscriptSegment,
    ValidatedEvidence,
    VideoMeta,
)
from lecnotes.pipeline.context import RunContext
from lecnotes.pipeline.stages.align.align import AlignStage
from lecnotes.pipeline.stages.align.lexical import (
    board_tokens,
    is_pointing,
    lexical_links,
    lexical_score,
    link_strength,
)

CODE = "for i, name in enumerate(names):\n    print(i, name)"


def seg(id: str, start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(id=id, start_s=start, end_s=end, text=text)


def board(id: str, first: float, last: float, text: str) -> BoardState:
    return BoardState(id=id, t_first_seen=first, t_last_seen=last, type=BoardType.CODE, text=text)


def test_board_tokens_keep_identifiers_drop_keywords_and_short_words() -> None:
    assert board_tokens(CODE) == ["name", "enumerate", "names"]  # first-appearance order


def test_plan_case_asr_split_still_links() -> None:
    """Board shows enumerate(); the transcript heard 'a numerate'."""
    tokens = board_tokens(CODE)
    assert lexical_score(tokens, "So here we call a numerate on the list") == 0.5
    assert lexical_score(tokens, "Napoleon lost at Waterloo in 1815") == 0


def test_substring_inside_another_word_does_not_match() -> None:
    assert lexical_score(["data"], "the record was updated yesterday") == 0
    assert lexical_score(["data"], "the data frame") == 0.5


def test_two_identifiers_give_full_score() -> None:
    assert lexical_score(board_tokens(CODE), "enumerate gives each name an index") == 1.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("As you can see the loop ends", True),
        ("Look at line two", True),
        ("this is important", True),
        ("Python was created in 1991", False),
    ],
)
def test_pointing_phrases(text: str, expected: bool) -> None:
    assert is_pointing(text) is expected


def temporal(seg_id: str, board_id: str, score: float) -> AlignmentLink:
    return AlignmentLink(
        segment_id=seg_id, board_state_id=board_id, method=AlignmentMethod.TEMPORAL, score=score
    )


def test_lexical_links_reach_back_to_an_earlier_board_within_window() -> None:
    boards = [board("b1", 0, 60, CODE), board("b2", 60, 400, "zip(a, b)")]
    segments = [
        seg("s1", 150, 160, "remember enumerate from before"),  # 90 s after b1 vanished
        seg("s2", 300, 310, "enumerate again"),  # 240 s after: outside 180 s window
        seg("s3", 150, 160, "the weather is nice"),
    ]
    links = lexical_links(segments, boards, temporal=[])
    assert [(lk.segment_id, lk.board_state_id) for lk in links] == [("s1", "b1")]


def test_pointing_boost_only_with_a_temporal_link() -> None:
    boards = [board("b1", 0, 60, CODE)]
    s = seg("s1", 10, 20, "as you can see enumerate returns pairs")
    with_temporal = lexical_links([s], boards, temporal=[temporal("s1", "b1", 1.0)])
    without = lexical_links([s], boards, temporal=[])
    assert with_temporal[0].score == pytest.approx(0.6)
    assert without[0].score == 0.5


def test_link_strength_lifts_extended_links_and_caps_at_one() -> None:
    lexical = AlignmentLink(
        segment_id="s1", board_state_id="b2", method=AlignmentMethod.LEXICAL, score=1.0
    )
    strength = link_strength(
        [temporal("s1", "b1", 1.0), temporal("s1", "b2", 0.2), lexical, temporal("s2", "b1", 1.0)]
    )
    assert strength == {("s1", "b1"): 1.0, ("s1", "b2"): pytest.approx(0.7), ("s2", "b1"): 1.0}


async def test_align_stage_v2_combines_both_methods() -> None:
    evidence = PerceptionResult(
        video_id="dQw4w9WgXcQ",
        segments=[seg("s1", 10, 20, "here enumerate gives the index")],
        board_states=[board("b1", 0, 60, CODE)],
    )
    ctx = RunContext(
        job_id=uuid4(), video=VideoMeta(video_id="dQw4w9WgXcQ", title="L", duration_s=100)
    )
    out = await AlignStage().run(ValidatedEvidence(evidence=evidence), ctx)
    assert sorted((lk.method.value, lk.score) for lk in out.links) == [
        ("lexical", pytest.approx(0.6)),
        ("temporal", 1.0),
    ]
    assert AlignStage.version == 2
