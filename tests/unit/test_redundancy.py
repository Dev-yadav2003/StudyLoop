from evals.metrics.redundancy import redundancy
from lecnotes.domain import Evidence, KnowledgeUnit, VideoSpan

VID = "Ek49cGiAOwo"


def unit(id: str, concept: str, explanation: str) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=id,
        concept=concept,
        explanation=explanation,
        spans=[VideoSpan(video_id=VID, start_s=0, end_s=1)],
        evidence=[Evidence(video_id=VID, segment_ids=["s1"])],
    )


def test_counts_units_with_a_duplicate() -> None:
    units = [
        unit("u1", "zip() Function", "zip combines two lists so you can loop over them together."),
        unit("u2", "zip()", "Loop over two lists in parallel with zip."),  # same core name
        unit("u3", "Manual indexing", "Keep a counter i and use ages[i] inside the loop."),
        unit("u4", "Index counter", "Keep a counter i and use ages[i] inside the loop body."),
        unit("u5", "List iteration", "A for loop visits each element of a list."),
    ]
    score = redundancy(units)
    assert score.pairs == [("u1", "u2"), ("u3", "u4")]
    assert (score.duplicated, score.units) == (4, 5)
    assert score.rate == 0.8


def test_no_units() -> None:
    assert redundancy([]).rate == 0.0
