"""Anchored tests: every assertion here compares against an independently derived answer.

Motivation. Fixing four silent defects required changing four existing tests, and two of
them had been *pinning* the bugs:

- a "golden" test asserted ``bayes_threshold(cost_fp=1, cost_fn=5)`` equals
  ``optimize_thresholds(utility={"tp": 2, "tn": 1, ...}, mode="bayes")``. Those agree only
  if the API route also drops ``tp`` and ``tn`` -- which was the bug. Two wrong paths
  agreeing is indistinguishable from two right ones agreeing.
- another asserted ``result.method == "minimize"`` for ``accuracy``, which was precisely
  the suboptimal routing being fixed.

Neither could ever have failed. A survey of this suite found 29 more tests whose every
assertion is incapable of failing for the quantity asserted -- ``assert score >= 0`` for a
metric that is non-negative by construction, ``assert 0 <= threshold <= 1`` for a value
that is clipped to that range, and so on. They cluster on exactly the degenerate inputs
where a wrong answer looks plausible.

Sweeping those cases against independently derived answers found no further defects. These
tests record the anchors that established that, so the next regression is caught by the
suite rather than by someone editing the code.

The rule for anything added here: the expected value must come from a closed form, a
brute-force search, or a hand-built confusion matrix -- never from running the code and
recording what it said.
"""

import itertools

import numpy as np
import pytest

from optimal_cutoffs import optimize_thresholds
from optimal_cutoffs.metrics_core import (
    compute_metric_at_threshold,
    confusion_matrix_at_threshold,
)

METRICS = ["f1", "accuracy", "precision", "recall", "iou", "specificity"]


def brute_force_best(y_true, y_score, metric, comparison=">", sample_weight=None):
    """Best achievable score, by evaluating every distinct cut point directly.

    Independent of the optimizer under test: it walks candidate thresholds and calls the
    metric, rather than reusing any of the optimizer's own machinery.
    """
    ordered = np.sort(np.unique(y_score))
    midpoints = (ordered[:-1] + ordered[1:]) / 2 if len(ordered) > 1 else []
    candidates = np.unique(np.concatenate([[0.0, 1.0], y_score, midpoints]))
    return max(
        compute_metric_at_threshold(
            y_true, y_score, t, metric, comparison=comparison, sample_weight=sample_weight
        )
        for t in candidates
    )


DEGENERATE = {
    "all labels positive": (np.ones(6, int), np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])),
    "all labels negative": (np.zeros(6, int), np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])),
    "one positive": (np.array([0, 0, 0, 0, 0, 1]), np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.9])),
    "one negative": (np.array([1, 1, 1, 1, 1, 0]), np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.1])),
    "every score tied": (np.array([0, 1, 0, 1, 1, 0]), np.full(6, 0.5)),
    "scores at 0 and 1": (np.array([0, 0, 1, 1]), np.array([0.0, 0.0, 1.0, 1.0])),
    "single sample, positive": (np.array([1]), np.array([0.7])),
    "single sample, negative": (np.array([0]), np.array([0.3])),
}


@pytest.mark.parametrize(
    "case,metric", list(itertools.product(DEGENERATE, METRICS)), ids=lambda v: str(v)
)
def test_degenerate_inputs_reach_the_achievable_optimum(case, metric):
    """The returned threshold must achieve the best score any threshold could.

    The tests this replaces asserted ``score >= 0.5`` on four all-positive labels, where
    the achievable score is 1.0 -- predicting everything positive is perfect. That
    tolerated getting half the points wrong.
    """
    y_true, y_score = DEGENERATE[case]
    result = optimize_thresholds(y_true, y_score, metric=metric)
    achieved = compute_metric_at_threshold(y_true, y_score, result.threshold, metric)
    best = brute_force_best(y_true, y_score, metric)
    assert achieved == pytest.approx(best, abs=1e-9), (
        f"{case} / {metric}: threshold {result.threshold} scores {achieved}, "
        f"but {best} is achievable"
    )


@pytest.fixture
def signal_data():
    rng = np.random.RandomState(3)
    n = 60
    y = rng.binomial(1, 0.4, n)
    score = np.clip(0.5 + 0.3 * (y - 0.4) + 0.2 * rng.randn(n), 0.01, 0.99)
    return y, score


@pytest.mark.parametrize("metric", ["f1", "accuracy", "precision", "recall", "iou"])
def test_unit_weights_reproduce_the_unweighted_fit(metric, signal_data):
    """weights = 1 must give bit-identical results, not merely similar ones."""
    y, score = signal_data
    plain = optimize_thresholds(y, score, metric=metric).threshold
    weighted = optimize_thresholds(
        y, score, metric=metric, sample_weight=np.ones(len(y))
    ).threshold
    assert plain == weighted


@pytest.mark.parametrize("metric", ["f1", "accuracy", "precision", "recall", "iou"])
def test_duplicating_a_row_equals_giving_it_weight_two(metric, signal_data):
    """Catches weights entering the search but not the metric, or the reverse."""
    y, score = signal_data
    rng = np.random.RandomState(11)
    idx = rng.choice(len(y), 15, replace=False)

    weights = np.ones(len(y))
    weights[idx] = 2.0
    weighted_thr = optimize_thresholds(
        y, score, metric=metric, sample_weight=weights
    ).threshold
    weighted_score = compute_metric_at_threshold(
        y, score, weighted_thr, metric, sample_weight=weights
    )

    y_dup = np.concatenate([y, y[idx]])
    score_dup = np.concatenate([score, score[idx]])
    dup_thr = optimize_thresholds(y_dup, score_dup, metric=metric).threshold
    dup_score = compute_metric_at_threshold(y_dup, score_dup, dup_thr, metric)

    assert weighted_score == pytest.approx(dup_score, abs=1e-9)


@pytest.mark.parametrize("comparison", [">", ">="])
def test_confusion_matrix_against_a_hand_built_one(comparison, signal_data):
    """Anchored on counting, not on another function in the package."""
    y, score = signal_data
    thresholds = [0.0, 0.25, 0.5, 0.75, 1.0, float(np.median(score)), float(score.min())]

    for t in thresholds:
        predicted = (score > t) if comparison == ">" else (score >= t)
        expected = (
            int(((predicted == 1) & (y == 1)).sum()),  # tp
            int(((predicted == 0) & (y == 0)).sum()),  # tn
            int(((predicted == 1) & (y == 0)).sum()),  # fp
            int(((predicted == 0) & (y == 1)).sum()),  # fn
        )
        got = tuple(
            int(v) for v in confusion_matrix_at_threshold(y, score, t, comparison=comparison)[:4]
        )
        assert got == expected, f"comparison={comparison} threshold={t}"
        # The four cells must partition the sample.
        assert sum(got) == len(y)


def test_expected_mode_matches_an_independent_expected_f1_grid():
    """Dinkelbach's fixed point must land on the argmax of the objective it claims.

    E[F1](t) is computed here directly from the probabilities rather than through any of
    the package's own expected-value machinery.
    """
    rng = np.random.RandomState(5)
    n = 200
    y = rng.binomial(1, 0.35, n)
    score = np.clip(0.5 + 0.3 * (y - 0.35) + 0.25 * rng.randn(n), 0.001, 0.999)

    def expected_f1(t):
        selected = score > t
        e_tp = score[selected].sum()
        e_fp = (1 - score[selected]).sum()
        e_fn = score[~selected].sum()
        denominator = 2 * e_tp + e_fp + e_fn
        return 0.0 if denominator == 0 else 2 * e_tp / denominator

    result = optimize_thresholds(None, score, metric="f1", mode="expected")
    grid = np.unique(np.concatenate([[0.0, 1.0], score]))
    best = max(expected_f1(t) for t in grid)

    assert expected_f1(result.threshold) == pytest.approx(best, abs=1e-9)
