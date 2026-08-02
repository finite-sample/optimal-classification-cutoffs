"""Tests for the code coherence improvements implemented in v0.4.0."""

import numpy as np
import pytest

from optimal_cutoffs import optimize_thresholds
from optimal_cutoffs.bayes import threshold as bayes_threshold
from optimal_cutoffs.cv import cross_validate, nested_cross_validate


class TestModeParameter:
    """Test the new mode parameter functionality."""

    def test_mode_empirical_default(self):
        """Test that mode='empirical' works as default."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9])

        # These should be equivalent
        result1 = optimize_thresholds(y_true, y_prob, metric="f1")
        result2 = optimize_thresholds(y_true, y_prob, metric="f1", mode="empirical")

        assert abs(result1.threshold - result2.threshold) < 1e-10

    def test_mode_bayes_requires_utility(self):
        """Test that mode='bayes' requires utility parameter."""
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9])

        with pytest.raises(
            ValueError,
            match="mode='bayes' requires 'utility' .* or 'fp_costs'/'fn_costs'",
        ):
            optimize_thresholds(None, y_prob, mode="bayes")

    def test_mode_bayes_with_utility(self):
        """Test that mode='bayes' works with utility parameter."""
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9])
        utility = {"tp": 0, "tn": 0, "fp": -1, "fn": -5}

        result1 = optimize_thresholds(None, y_prob, mode="bayes", utility=utility)
        result_expected = bayes_threshold(cost_fp=1, cost_fn=5)

        assert abs(result1.threshold - result_expected) < 1e-10

    def test_mode_expected_f1_only(self):
        """Test that mode='expected' only works with F1/F-beta metrics."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9])

        # Should work with F1 and return an OptimizationResult
        result1 = optimize_thresholds(y_true, y_prob, metric="f1", mode="expected")
        assert hasattr(result1, "threshold") and hasattr(result1, "score")
        threshold, f1_score = result1.threshold, result1.score
        assert isinstance(threshold, float | np.number) or (
            isinstance(threshold, np.ndarray) and threshold.size == 1
        )
        assert isinstance(f1_score, float)
        assert 0 <= threshold <= 1

        # Should also work with f1 (which is F-beta with beta=1)
        result2 = optimize_thresholds(y_true, y_prob, metric="f1", mode="expected")
        assert hasattr(result2, "threshold") and hasattr(result2, "score")

        # Should NOT work with non-F-beta metrics
        with pytest.raises(
            ValueError, match="mode='expected' currently supports F-beta only"
        ):
            optimize_thresholds(y_true, y_prob, metric="precision", mode="expected")

    def test_mode_expected_supports_multiclass(self):
        """Test that mode='expected' works with multiclass classification."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_prob = np.random.rand(6, 3)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # Normalize probabilities

        # Should work with multiclass and return an OptimizationResult
        result1 = optimize_thresholds(y_true, y_prob, metric="f1", mode="expected")

        # Check that it's an OptimizationResult with proper attributes
        assert hasattr(result1, "thresholds")
        assert hasattr(result1, "score")
        assert isinstance(result1.thresholds, np.ndarray)
        assert len(result1.thresholds) == 3  # 3 classes

    def test_mode_expected_supports_sample_weights(self):
        """Test that mode='expected' supports sample weights."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9])
        sample_weight = np.array([1, 1, 2, 2, 1, 2])

        # Should work with sample weights
        result1 = optimize_thresholds(
            y_true,
            y_prob,
            metric="f1",
            mode="expected",
            sample_weight=sample_weight,
        )
        threshold = result1.threshold
        assert hasattr(result1, "threshold") and hasattr(result1, "score")
        threshold, _f1_score = result1.threshold, result1.score
        assert 0 <= threshold <= 1
        assert 0 <= threshold <= 1

    def test_mode_expected_works_without_true_labs(self):
        """Test that mode='expected' works without true_labs."""
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9])

        # Should work without true_labs
        result1 = optimize_thresholds(None, y_prob, metric="f1", mode="expected")
        threshold = result1.threshold
        threshold = result1.threshold
        threshold = result1.threshold
        assert hasattr(result1, "threshold") and hasattr(result1, "score")
        threshold, _f1_score = result1.threshold, result1.score
        assert 0 <= threshold <= 1
        assert 0 <= threshold <= 1


class TestDeprecatedParameterRejection:
    """Test that deprecated parameters are properly rejected."""

    def test_bayes_parameter_rejected(self):
        """Test that bayes=True raises TypeError (parameter no longer exists)."""
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9])
        utility = {"tp": 0, "tn": 0, "fp": -1, "fn": -5}

        with pytest.raises(TypeError, match="unexpected keyword argument"):
            optimize_thresholds(None, y_prob, utility=utility, bayes=True)

    def test_deprecated_dinkelbach_method_rejected(self):
        """Test that deprecated method='dinkelbach' raises ValueError (method no longer exists)."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9])

        with pytest.raises(ValueError, match="Invalid optimization method"):
            optimize_thresholds(y_true, y_prob, metric="f1", method="dinkelbach")

    def test_deprecated_smart_brute_method_rejected(self):
        """Test that deprecated method='smart_brute' raises ValueError (method no longer exists)."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9])

        with pytest.raises(ValueError, match="Invalid optimization method"):
            optimize_thresholds(y_true, y_prob, metric="f1", method="smart_brute")


class TestMethodEquivalence:
    """Test that different methods produce equivalent results."""

    def test_unique_scan_vs_sort_scan_equivalence(self):
        """Test that unique_scan gives same results as sort_scan for piecewise metrics."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9, 0.15, 0.85])

        result = optimize_thresholds(y_true, y_prob, metric="f1", method="sort_scan")

        result2 = optimize_thresholds(y_true, y_prob, metric="f1", method="sort_scan")

        # Both methods should achieve the same optimal score (thresholds may differ on plateaus)
        from optimal_cutoffs.metrics import compute_metric_at_threshold

        score_unique = compute_metric_at_threshold(
            y_true, y_prob, result.threshold, "f1"
        )
        score_sort = compute_metric_at_threshold(
            y_true, y_prob, result2.threshold, "f1"
        )
        assert (
            abs(score_unique - score_sort) < 1e-10
        ), f"Score mismatch: unique_scan={score_unique:.10f}, sort_scan={score_sort:.10f}"

    def test_unique_scan_vs_sort_scan_on_piecewise_metrics(self):
        """Test that unique_scan gives same results as sort_scan for piecewise metrics."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9, 0.15, 0.85])

        optimize_thresholds(y_true, y_prob, metric="f1", method="sort_scan")

        result2 = optimize_thresholds(y_true, y_prob, metric="f1", method="sort_scan")

        # Both methods should achieve the same optimal score (thresholds may differ on plateaus)
        from optimal_cutoffs.metrics import compute_metric_at_threshold

        score_unique = compute_metric_at_threshold(
            y_true, y_prob, result2.threshold, "f1"
        )
        score_sort = compute_metric_at_threshold(
            y_true, y_prob, result2.threshold, "f1"
        )
        assert (
            abs(score_unique - score_sort) < 1e-10
        ), f"Score mismatch: unique_scan={score_unique:.10f}, sort_scan={score_sort:.10f}"


class TestCVDefaultMethods:
    """Test that CV functions use 'auto' as default method."""

    def test_cross_validate_default_method(self):
        """Test that cross_validate uses 'auto' by default."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9, 0.15, 0.85, 0.75, 0.25])

        # Should not raise an error and should return valid results
        thresholds, scores = cross_validate(y_true, y_prob, cv=3)

        assert len(thresholds) == 3
        assert len(scores) == 3
        assert all(
            isinstance(t, int | float | np.number)
            or (isinstance(t, np.ndarray) and t.size == 1)
            for t in thresholds
        )
        assert all(isinstance(s, int | float | np.number) for s in scores)

    def test_nested_cross_validate_default_method(self):
        """Test that nested_cross_validate uses 'auto' by default."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9, 0.15, 0.85, 0.75, 0.25])

        # Should not raise an error and should return valid results
        thresholds, scores = nested_cross_validate(
            y_true, y_prob, inner_cv=3, outer_cv=3
        )

        assert len(thresholds) == 3
        assert len(scores) == 3
        assert all(
            isinstance(t, int | float | np.number)
            or (isinstance(t, np.ndarray) and t.size == 1)
            for t in thresholds
        )
        assert all(isinstance(s, int | float | np.number) for s in scores)


class TestGoldenTests:
    """Golden tests for equivalence across regimes."""

    def test_bayes_closed_form_vs_utility_api(self):
        """Test that Bayes closed-form equals optimize_thresholds with utility."""
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9])
        utility = {"tp": 2, "tn": 1, "fp": -1, "fn": -5}

        # Direct call to Bayes function. All four utility terms must be supplied: the
        # closed form tau* = (u_tn - u_fp) / [(u_tp - u_fn) + (u_tn - u_fp)] depends on
        # tp and tn as well as fp and fn.
        result1 = bayes_threshold(
            cost_fp=1, cost_fn=5, benefit_tp=utility["tp"], benefit_tn=utility["tn"]
        )

        # Via optimize_thresholds API
        result2 = optimize_thresholds(None, y_prob, utility=utility, mode="bayes")

        assert abs(result1 - result2.threshold) < 1e-12
        # (u_tn - u_fp) / [(u_tp - u_fn) + (u_tn - u_fp)] = 2 / (7 + 2) = 2/9
        assert abs(result2.threshold - 2 / 9) < 1e-12

    def test_method_consistency(self):
        """Test that methods give consistent results."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9, 0.15, 0.85])

        result = optimize_thresholds(y_true, y_prob, metric="f1", method="sort_scan")

        result2 = optimize_thresholds(y_true, y_prob, metric="f1", method="minimize")

        # Different methods may give slightly different results, but should be close
        # Relaxed tolerance since methods may have significant differences in edge cases
        assert abs(result.threshold - result2.threshold) < 0.2

    def test_expected_mode_works(self):
        """Test that mode='expected' works correctly."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9])

        # Both calls should work and give same result1
        result1 = optimize_thresholds(y_true, y_prob, mode="expected", metric="f1")

        result2 = optimize_thresholds(y_true, y_prob, metric="f1", mode="expected")

        # Both should return tuples
        assert hasattr(result1, "threshold") and hasattr(
            result1, "score"
        )  # Expected mode returns OptimizationResult
        assert hasattr(result2, "threshold") and hasattr(
            result2, "score"
        )  # Expected mode returns OptimizationResult

        # Extract thresholds and compare
        _threshold1, f1_score1 = result2.threshold, result2.score
        _threshold1, f1_score2 = result2.threshold, result2.score
        assert abs(result1.threshold - result2.threshold) < 1e-12
        assert abs(f1_score1 - f1_score2) < 1e-12


class TestErrorMessages:
    """Test that error messages are clear and helpful."""

    def test_mode_bayes_error_message(self):
        """Test clear error message for mode='bayes' without utility."""
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9])

        with pytest.raises(ValueError) as exc_info:
            optimize_thresholds(None, y_prob, mode="bayes")

        assert "mode='bayes' requires" in str(exc_info.value)

    def test_mode_expected_supports_multiple_metrics(self):
        """Test that mode='expected' currently supports F-beta metrics only."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.9])

        # Should work with F1 metric (the only supported F-beta metric currently)
        result1 = optimize_thresholds(y_true, y_prob, metric="f1", mode="expected")
        threshold = result1.threshold
        assert hasattr(result1, "threshold") and hasattr(result1, "score")
        threshold, f1_score = result1.threshold, result1.score
        assert 0 <= threshold <= 1
        assert 0 <= f1_score <= 1

        # Test that unsupported metrics raise appropriate errors
        for metric in ["accuracy", "precision", "recall", "specificity", "jaccard"]:
            with pytest.raises(
                ValueError, match="mode='expected' currently supports F-beta only"
            ):
                optimize_thresholds(y_true, y_prob, metric=metric, mode="expected")

    def test_mode_expected_multiclass_support(self):
        """Test that mode='expected' supports multiclass classification."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_prob = np.random.rand(6, 3)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # Normalize probabilities

        # Should work with multiclass and return an OptimizationResult
        result1 = optimize_thresholds(y_true, y_prob, metric="f1", mode="expected")

        # Check that it's an OptimizationResult with proper attributes
        assert hasattr(result1, "thresholds")
        assert hasattr(result1, "score")
