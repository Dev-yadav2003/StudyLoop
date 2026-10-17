import pytest

from evals.golden_set import RefBoard
from evals.metrics.board_recall import board_recall
from evals.metrics.boundary_f1 import boundary_f1, predicted_boundaries
from evals.metrics.wer_terms import normalize, term_error_rate, wer
from lecnotes.domain import BoardState, BoardType


def state(id: str, t: float, text: str, type: BoardType = BoardType.CODE) -> BoardState:
    return BoardState(id=id, t_first_seen=t, t_last_seen=t + 10, type=type, text=text)


# --- WER ---------------------------------------------------------------------------------


def test_normalize() -> None:
    assert normalize("Hello,   World! It's O(n).") == "hello world it's o n"


def test_wer_exact_and_known_errors() -> None:
    assert wer("the cat sat", "The cat, sat.") == 0
    # 1 substitution + 1 deletion over 4 reference words
    assert wer("we call enumerate here", "we call numerate") == pytest.approx(0.5)


def test_wer_empty_reference_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        wer("  ", "anything")


def test_term_error_rate_catches_mangled_terms() -> None:
    ref = "use enumerate here and enumerate there then zip"
    hyp = "use a numerate here and enumerate there then zip"
    score = term_error_rate(ref, hyp, ["enumerate", "zip", "lambda"])
    assert score is not None
    assert score.per_term == {"enumerate": (2, 1), "zip": (1, 1)}  # lambda: not in reference
    assert score.error_rate == pytest.approx(1 / 3)
    assert score.missed == ["enumerate"]


def test_term_matching_is_whole_word_and_multiword() -> None:
    score = term_error_rate("big o notation", "biggest o notation", ["big o"])
    assert score is not None
    assert score.error_rate == 1.0


def test_term_error_rate_none_when_no_terms_in_reference() -> None:
    assert term_error_rate("hello world", "hello world", ["enumerate"]) is None


# --- board recall ------------------------------------------------------------------------


REFERENCE = [
    RefBoard(t_s=0, type=BoardType.SLIDE, lines=["Python Iteration Tools"]),
    RefBoard(
        t_s=42,
        type=BoardType.CODE,
        lines=[
            "for i, name in enumerate(names):",
            "    print(i, name)",
            "x = [unclear]",
            "zip(a, b)",
        ],
    ),
]


def test_board_recall_counts_fuzzy_matches_and_skips_unclear() -> None:
    predicted = [
        state("b1", 0, "Python Iteration Tools", BoardType.SLIDE),
        state("b2", 42, "for i, name in enumerate(names) :\nprint(i,  name)"),
    ]
    score = board_recall(REFERENCE, predicted)
    assert (score.matched, score.total) == (3, 4)  # [unclear] line skipped; zip missed
    assert score.recall == pytest.approx(0.75)
    assert score.misses == ["zip(a, b)"]
    assert score.by_type == {BoardType.SLIDE: (1, 1), BoardType.CODE: (2, 3)}


def test_board_recall_is_case_sensitive_for_code() -> None:
    ref = [RefBoard(t_s=0, type=BoardType.CODE, lines=["MyList = List()"])]
    pred = [state("b1", 0, "mylist = list()")]
    assert board_recall(ref, pred).matched == 0


def test_short_reference_line_does_not_match_inside_long_line() -> None:
    ref = [RefBoard(t_s=0, type=BoardType.CODE, lines=["zip(a, b)"])]
    pred = [state("b1", 0, "result = list(zip(a, b)) + sorted(other_things, key=len)")]
    assert board_recall(ref, pred).matched == 0


def test_board_recall_with_no_predictions() -> None:
    score = board_recall(REFERENCE, [])
    assert score.recall == 0
    assert board_recall([], []).recall is None


# --- boundary F1 -------------------------------------------------------------------------


def test_boundary_f1_with_tolerance_and_drift() -> None:
    score = boundary_f1([42, 131, 300], [44, 128, 200, 305.5], tolerance_s=5)
    assert score.matched == 2  # 42~44, 131~128 ; 300 vs 305.5 is outside tolerance
    assert score.precision == pytest.approx(2 / 4)
    assert score.recall == pytest.approx(2 / 3)
    assert score.f1 == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))
    assert score.mean_drift_s == pytest.approx(2.5)


def test_boundary_matching_is_one_to_one() -> None:
    # Two predictions near one reference boundary: only one may count.
    score = boundary_f1([60], [58, 61])
    assert (score.matched, score.precision, score.recall) == (1, 0.5, 1.0)


def test_greedy_picks_the_assignment_that_maximises_matches() -> None:
    # ref 10 could take pred 13 or 6; taking 6 leaves 13 for ref 17.
    assert boundary_f1([10, 17], [6, 13], tolerance_s=4).matched == 2


def test_boundary_edge_cases() -> None:
    assert boundary_f1([], []).f1 == 1.0
    assert boundary_f1([10], []).f1 == 0.0
    assert boundary_f1([], [10]).precision == 0.0


def test_predicted_boundaries_skip_video_start_and_dedupe() -> None:
    states = [state("a", 0, "x"), state("b", 42, "y"), state("c", 42, "z"), state("d", 90, "w")]
    assert predicted_boundaries(states) == [42, 90]
