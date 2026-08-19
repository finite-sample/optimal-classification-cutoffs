"""Tests that the implementation honours its own documented contracts.

Each test here pins a behaviour that the package's README or docstrings state
explicitly, and that was previously computed differently without any error or
warning (silent wrongness).
"""

import numpy as np
import pytest

from optimal_cutoffs import optimize_decisions, optimize_thresholds
from optimal_cutoffs.core import Task, select_method_with_explanation
from optimal_cutoffs.metrics_core import (
    has_vectorized_implementation,
    is_piecewise_metric,
)


class TestCostMatrixOrientation:
    """optimize_decisions documents cost_matrix[i, j]: predicting j when truth is i."""

    def test_readme_cost_matrix_orientation(self):
        """README example: [[0, 1], [10, 0]] means FN costs 10x more than FP.

        With rows = true class:
            C[0, 1] = 1  -> false positive costs 1
            C[1, 0] = 10 -> false negative costs 10
        Bayes rule predicts positive iff p >= c_fp / (c_fp + c_fn) = 1/11.
        """
        cost_matrix = np.array([[0.0, 1.0], [10.0, 0.0]])
        p1 = np.array([0.02, 0.05, 0.09, 0.15, 0.30, 0.50, 0.80, 0.95])
        y_score = np.column_stack([1 - p1, p1])

        got = optimize_decisions(y_score, cost_matrix).predict(y_score)
        expected = (p1 >= 1.0 / 11.0).astype(int)

        np.testing.assert_array_equal(got, expected)

    def test_cost_matrix_minimises_documented_cost(self):
        """Decisions minimise mean cost_matrix[true, pred], as documented."""
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, 20000)
        scores = np.column_stack([1 - p, p])
        y_true = (rng.uniform(size=p.size) < p).astype(int)

        cost_matrix = np.array([[0.0, 1.0], [10.0, 0.0]])
        dec = optimize_decisions(scores, cost_matrix).predict(scores)
        cost_lib = cost_matrix[y_true, dec].mean()

        best = min(
            cost_matrix[y_true, (p >= tau).astype(int)].mean()
            for tau in np.linspace(0, 1, 501)
        )
        # `best` is the in-sample minimum over a grid, so it slightly overfits the
        # finite sample; the Bayes rule cannot beat it. Allow that sampling noise.
        # Before the orientation fix this was 4.1845 vs 0.4451 -- a 9.4x excess.
        assert cost_lib <= best + 0.01, (
            f"library mean cost {cost_lib:.4f} exceeds best threshold rule {best:.4f}"
        )

    def test_asymmetric_three_class_cost_matrix(self):
        """Rows are true classes: row i costs predicting each j when truth is i."""
        # Predicting class 0 is catastrophic when the truth is class 2.
        cost_matrix = np.array(
            [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [100.0, 1.0, 0.0],
            ]
        )
        # Sample that is mostly class 0 but with a non-trivial chance of class 2.
        probs = np.array([[0.60, 0.05, 0.35]])
        got = optimize_decisions(probs, cost_matrix).predict(probs)

        expected_cost = probs @ cost_matrix  # (1, n_actions), documented orientation
        expected = np.argmin(expected_cost, axis=1)

        np.testing.assert_array_equal(got, expected)
        # Expected costs by action: 0 -> 0.35*100 + 0.05 = 35.05, 1 -> 0.60 + 0.35 =
        # 0.95,
        # 2 -> 0.60 + 0.05 = 0.65. The catastrophic true-car/predict-dog entry must push
        # the decision away from action 0.
        assert got[0] == 2


class TestAutoSelectsExactMethodForPiecewiseMetrics:
    """README: 'Exact solutions guaranteed for piecewise-constant metrics'."""

    @pytest.mark.parametrize(
        "metric", ["f1", "accuracy", "precision", "recall", "iou", "specificity"]
    )
    def test_auto_uses_sort_scan_for_piecewise_vectorized_metrics(self, metric):
        assert is_piecewise_metric(metric)
        assert has_vectorized_implementation(metric)
        method, _ = select_method_with_explanation(Task.BINARY, metric, 100)
        assert method == "sort_scan", (
            f"{metric} is piecewise with a vectorized implementation but auto chose "
            f"{method}"
        )

    @pytest.mark.parametrize("metric", ["accuracy", "iou", "specificity"])
    def test_auto_matches_exact_optimum(self, metric):
        """Default (auto) must not be beaten by the exact sort_scan optimizer."""
        n_worse = 0
        for seed in range(25):
            rng = np.random.default_rng(seed)
            y = rng.integers(0, 2, 200)
            s = np.clip(rng.beta(2, 5, 200) + 0.3 * y, 0, 1)

            auto = optimize_thresholds(y, s, metric=metric)
            exact = optimize_thresholds(y, s, metric=metric, method="sort_scan")
            if exact.scores[0] > auto.scores[0] + 1e-12:
                n_worse += 1
        assert n_worse == 0, (
            f"auto was suboptimal for {metric} on {n_worse}/25 datasets"
        )


class TestCrossValidateReportsHeldOutScores:
    """cv.cross_validate must evaluate the training-fold threshold on the held-out fold.

    Re-optimising on the test fold reports that fold's own maximum, which is an
    optimistically biased estimate rather than a cross-validated one.
    """

    def test_scores_correspond_to_returned_thresholds(self):
        from sklearn.metrics import f1_score
        from sklearn.model_selection import StratifiedKFold

        from optimal_cutoffs import cv

        rng = np.random.default_rng(0)
        n = 400
        y = rng.integers(0, 2, n)
        s = np.clip(rng.beta(2, 5, n) + 0.35 * y, 0, 1)

        thr, sc = cv.cross_validate(y, s, metric="f1", cv=5, random_state=0)

        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        folds = zip(splitter.split(y, y), thr, sc, strict=True)
        for (_, test_idx), t, reported in folds:
            honest = f1_score(
                y[test_idx], (s[test_idx] > t).astype(int), zero_division=0
            )
            assert reported == pytest.approx(honest, abs=1e-9), (
                f"reported fold score {reported:.6f} != metric of returned "
                f"threshold {t:.6f} on the held-out fold ({honest:.6f})"
            )

    def test_scores_are_not_test_fold_maxima(self):
        """The reported score must not simply be the test fold's own optimum."""
        from sklearn.metrics import f1_score
        from sklearn.model_selection import StratifiedKFold

        from optimal_cutoffs import cv

        rng = np.random.default_rng(0)
        n = 400
        y = rng.integers(0, 2, n)
        s = np.clip(rng.beta(2, 5, n) + 0.35 * y, 0, 1)

        _thr, sc = cv.cross_validate(y, s, metric="f1", cv=5, random_state=0)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

        test_optima = [
            max(
                f1_score(y[test_idx], (s[test_idx] > c).astype(int), zero_division=0)
                for c in np.unique(s[test_idx])
            )
            for _, test_idx in splitter.split(y, y)
        ]
        # A held-out score can coincidentally equal the fold optimum, but not on every
        # fold of a dataset with this much threshold variation.
        assert not np.allclose(sc, test_optima), (
            "every fold's reported score equals that fold's own optimum, "
            "indicating the threshold was re-fit on the test data"
        )


class TestBayesModeUsesAllDocumentedUtilityKeys:
    """optimize_thresholds documents keys 'tp', 'tn', 'fp', 'fn' for mode='bayes'."""

    def test_bayes_mode_honours_tp_and_tn(self):
        utility = {"tp": 10.0, "tn": 1.0, "fp": -1.0, "fn": -5.0}
        u_tp, u_tn, u_fp, u_fn = 10.0, 1.0, -1.0, -5.0
        expected = (u_tn - u_fp) / ((u_tp - u_fn) + (u_tn - u_fp))

        p = np.linspace(0.01, 0.99, 200)
        got = optimize_thresholds(None, p, mode="bayes", utility=utility).threshold

        assert got == pytest.approx(expected, abs=1e-9), (
            f"mode='bayes' returned {got:.6f}, closed form is {expected:.6f}"
        )

    def test_bayes_mode_agrees_with_empirical_utility_path(self):
        """The same utility dict must give the same closed-form threshold either way."""
        utility = {"tp": 10.0, "tn": 1.0, "fp": -1.0, "fn": -5.0}
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, 5000)
        y = (rng.uniform(size=p.size) < p).astype(int)

        t_bayes = optimize_thresholds(None, p, mode="bayes", utility=utility).threshold
        t_emp = optimize_thresholds(y, p, utility=utility).threshold

        assert t_bayes == pytest.approx(t_emp, abs=1e-9)

    def test_bayes_mode_tp_tn_actually_change_the_threshold(self):
        """Changing the documented tp/tn keys must move the threshold."""
        base = {"tp": 0.0, "tn": 0.0, "fp": -1.0, "fn": -5.0}
        richer = {"tp": 20.0, "tn": 0.0, "fp": -1.0, "fn": -5.0}

        p = np.linspace(0.01, 0.99, 100)
        t_base = optimize_thresholds(None, p, mode="bayes", utility=base).threshold
        t_rich = optimize_thresholds(None, p, mode="bayes", utility=richer).threshold

        assert t_rich < t_base, (
            "raising the true-positive benefit must lower the threshold, "
            f"got {t_rich:.6f} vs {t_base:.6f}"
        )
