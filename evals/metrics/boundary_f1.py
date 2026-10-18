"""Boundary F1: did perception start a new board state when the visible content changed?

Reference = hand-marked change times. Predicted = first-seen time of each board state
(after 0 s). Each reference boundary may match at most one predicted boundary within
`tolerance_s`. Greedy matching over the two sorted lists is optimal for 1-D tolerance
matching, so no assignment solver is needed. Also reports mean drift of matched pairs —
the "timestamp drift" step 3.1 asks about.
"""

from __future__ import annotations

from dataclasses import dataclass

from lecnotes.domain import BoardState

DEFAULT_TOLERANCE_S = 5.0


@dataclass(frozen=True)
class BoundaryScore:
    matched: int
    n_reference: int
    n_predicted: int
    mean_drift_s: float | None
    """Mean |predicted - reference| over matched pairs."""

    @property
    def precision(self) -> float:
        return self.matched / self.n_predicted if self.n_predicted else 1.0

    @property
    def recall(self) -> float:
        return self.matched / self.n_reference if self.n_reference else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def predicted_boundaries(states: list[BoardState]) -> list[float]:
    return sorted({b.t_first_seen for b in states if b.t_first_seen > 0})


def boundary_f1(
    reference: list[float],
    predicted: list[float],
    *,
    tolerance_s: float = DEFAULT_TOLERANCE_S,
) -> BoundaryScore:
    ref, pred = sorted(reference), sorted(predicted)
    i = j = 0
    drifts: list[float] = []
    while i < len(ref) and j < len(pred):
        delta = pred[j] - ref[i]
        if abs(delta) <= tolerance_s:
            drifts.append(abs(delta))
            i += 1
            j += 1
        elif delta < 0:
            j += 1  # predicted boundary too early for this and every later reference
        else:
            i += 1  # reference boundary has no prediction close enough
    return BoundaryScore(
        matched=len(drifts),
        n_reference=len(ref),
        n_predicted=len(pred),
        mean_drift_s=sum(drifts) / len(drifts) if drifts else None,
    )
