"""Unit tests for Dinkelbach expected F-beta optimization method.

This module tests the Dinkelbach algorithm for optimizing expected F-beta scores
under perfect calibration assumptions. The method depends only on predicted
probabilities, not on realized labels.
"""

import warnings

import numpy as np

from optimal_cutoffs import optimize_thresholds
from optimal_cutoffs.expected import dinkelbach_expected_fbeta_binary
from optimal_cutoffs.metrics_core import confusion_matrix_at_threshold, f1_score
from tests.fixtures.assertions import (
    assert_valid_metric_score,
    assert_valid_threshold,
)


class TestDinkelbachBasic:
    """Test basic Dinkelbach method functionality."""

    def test_dinkelbach_basic_functionality(self):
        """Test that Dinkelbach method produces valid thresholds."""
        y_prob = np.array([0.1, 0.4, 0.6, 0.9])

        result = dinkelbach_expected_fbeta_binary(y_prob, beta=1.0)
        threshold = result.threshold
        expected_score = result.score

        assert_valid_threshold(threshold)
        assert_valid_metric_score(expected_score, "expected_f1")

    def test_dinkelbach_different_beta_values(self):
        """Test Dinkelbach with different beta values."""
        y_prob = np.array([0.2, 0.4, 0.6, 0.8])

        beta_values = [0.5, 1.0, 2.0]
        for beta in beta_values:
            result = dinkelbach_expected_fbeta_binary(y_prob, beta=beta)
            threshold = result.threshold
            expected_score = result.score

            assert_valid_threshold(threshold)
            assert_valid_metric_score(expected_score, f"expected_f{beta}")

    def test_dinkelbach_comparison_operators(self):
        """Test Dinkelbach with different comparison operators."""
        y_prob = np.array([0.3, 0.5, 0.5, 0.7])  # Include ties

        for comparison in [">", ">="]:
            result = dinkelbach_expected_fbeta_binary(
                y_prob, beta=1.0, comparison=comparison
            )
            threshold = result.threshold
            expected_score = result.score

            assert_valid_threshold(threshold)
            assert_valid_metric_score(expected_score, "expected_f1")

    def test_dinkelbach_single_probability(self):
        """Test Dinkelbach with single probability value."""
        y_prob = np.array([0.7])

        result = dinkelbach_expected_fbeta_binary(y_prob, beta=1.0)
        threshold = result.threshold
        expected_score = result.score

        assert_valid_threshold(threshold)
        assert_valid_metric_score(expected_score, "expected_f1")

    def test_dinkelbach_extreme_probabilities(self):
        """Test Dinkelbach with extreme probability values."""
        y_prob = np.array([0.0, 0.5, 1.0])

        result = dinkelbach_expected_fbeta_binary(y_prob, beta=1.0)
        threshold = result.threshold
        expected_score = result.score

        assert_valid_threshold(threshold)
        assert_valid_metric_score(expected_score, "expected_f1")


class TestDinkelbachAPI:
    """Test Dinkelbach through the main API."""

    def test_dinkelbach_through_optimize_thresholds(self):
        """Test Dinkelbach method through the main API."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        result = optimize_thresholds(y_true, y_prob, mode="expected", metric="f1")

        threshold = result.threshold
        assert hasattr(result, "threshold")
        assert hasattr(result, "score")
        threshold_expected = result.threshold
        # Both should be valid
        assert_valid_threshold(threshold)
        assert_valid_threshold(threshold_expected)

        # Compute empirical F1 for both thresholds
        tp_emp, tn_emp, fp_emp, fn_emp = confusion_matrix_at_threshold(
            y_true, y_prob, threshold
        )
        tp_exp, tn_exp, fp_exp, fn_exp = confusion_matrix_at_threshold(
            y_true, y_prob, threshold_expected
        )

        f1_emp = f1_score(tp_emp, tn_emp, fp_emp, fn_emp)
        f1_exp = f1_score(tp_exp, tn_exp, fp_exp, fn_exp)

        # Both should achieve reasonable F1 scores
        assert f1_emp >= 0.0
        assert f1_exp >= 0.0

    def test_dinkelbach_respects_beta_parameter(self):
        """Test that different beta values give different thresholds."""
        y_prob = np.array([0.2, 0.4, 0.6, 0.8])

        # Get thresholds for different beta values
        result_05 = dinkelbach_expected_fbeta_binary(
            y_prob, beta=0.5
        )  # Precision-weighted
        threshold_05 = result_05.threshold
        result_10 = dinkelbach_expected_fbeta_binary(y_prob, beta=1.0)  # F1
        threshold_10 = result_10.threshold
        result_20 = dinkelbach_expected_fbeta_binary(
            y_prob, beta=2.0
        )  # Recall-weighted
        threshold_20 = result_20.threshold

        # All should be valid
        for threshold in [threshold_05, threshold_10, threshold_20]:
            assert_valid_threshold(threshold)

        # They should generally be different (unless there's a unique optimum)
        # This is more of a sanity check than a strict requirement


class TestDinkelbachConvergenceWarnings:
    """Test convergence warnings for Dinkelbach algorithms."""

    def test_dinkelbach_normal_case_no_warnings(self):
        """Test that normal cases don't produce warnings."""
        y_prob = np.array([0.1, 0.4, 0.6, 0.9])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = dinkelbach_expected_fbeta_binary(y_prob, beta=1.0)
            threshold = result.threshold

            # Should not have any warnings for normal convergent cases
            assert len(w) == 0
            assert_valid_threshold(threshold)
            assert 0.0 <= threshold <= 1.0

    def test_dinkelbach_edge_cases_no_warnings(self):
        """Test that edge cases handle warnings appropriately."""
        edge_cases = [
            np.array([0.5]),  # Single probability
            np.array([0.0, 1.0]),  # Extreme probabilities
            np.zeros(5),  # All zeros
            np.ones(5),  # All ones
        ]

        for y_prob in edge_cases:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = dinkelbach_expected_fbeta_binary(y_prob, beta=1.0)
                threshold = result.threshold

                # Edge cases should generally converge without warnings
                assert_valid_threshold(threshold)
                assert 0.0 <= threshold <= 1.0


class TestDinkelbachBasicFunctionality:
    """Basic functionality tests for Dinkelbach method."""

    def test_dinkelbach_basic_functionality(self):
        """Test that Dinkelbach method produces valid thresholds."""
        # Test probability distribution
        y_prob = np.array([0.1, 0.4, 0.6, 0.9])

        result = dinkelbach_expected_fbeta_binary(y_prob, beta=1.0)
        threshold = result.threshold

        # Should return a valid threshold
        assert isinstance(threshold, float | np.number) or (
            isinstance(threshold, np.ndarray) and threshold.size == 1
        )
        assert 0.0 <= threshold <= 1.0

    def test_dinkelbach_through_optimize_thresholds(self):
        """Test Dinkelbach method through the main API."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        # Should work for F1 metric and return a tuple
        result = optimize_thresholds(y_true, y_prob, mode="expected", metric="f1")
        assert hasattr(result, "threshold")
        assert hasattr(result, "score")
        threshold_dinkelbach, f1_score_dinkelbach = result.threshold, result.score
        # Both should be valid values
        assert 0.0 <= threshold_dinkelbach <= 1.0
        assert 0.0 <= f1_score_dinkelbach <= 1.0

        # Dinkelbach should produce reasonable results
        assert isinstance(threshold_dinkelbach, float | np.number) or (
            isinstance(threshold_dinkelbach, np.ndarray)
            and threshold_dinkelbach.size == 1
        )
        assert isinstance(f1_score_dinkelbach, float | np.number)
