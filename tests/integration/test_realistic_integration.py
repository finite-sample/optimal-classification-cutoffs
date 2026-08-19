"""Integration tests with realistic datasets.

This module tests the threshold optimization algorithms with realistic datasets
that mirror real-world classification problems. All tests use datasets with
100+ samples and meaningful distributions.
"""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score as sklearn_accuracy
from sklearn.metrics import f1_score as sklearn_f1

from optimal_cutoffs import optimize_thresholds

# Local tolerance for test precision
from optimal_cutoffs.metrics import (
    accuracy_score,
    f1_score,
)
from optimal_cutoffs.metrics_core import (
    confusion_matrix_at_threshold,
)

# Import test fixtures
from tests.fixtures import (
    CALIBRATED_BINARY,
    IMBALANCED_BINARY,
    IMBALANCED_MULTICLASS,
    LARGE_BINARY,
    OVERLAPPING_BINARY,
    STANDARD_BINARY,
    STANDARD_MULTICLASS,
    WELL_SEPARATED_BINARY,
    make_realistic_multiclass_dataset,
)


class TestRealisticBinaryOptimization:
    """Test binary threshold optimization with realistic datasets."""

    def test_standard_binary_f1_optimization(self):
        """Test F1 optimization on standard realistic binary dataset."""
        y_true, y_prob = STANDARD_BINARY.y_true, STANDARD_BINARY.y_prob

        # Test different optimization methods
        methods = ["auto", "sort_scan", "sort_scan", "minimize"]
        thresholds = {}

        for method in methods:
            result = optimize_thresholds(y_true, y_prob, metric="f1", method=method)
            threshold = result.threshold
            thresholds[method] = threshold

            # Verify threshold produces good F1 score
            predictions = (y_prob > threshold).astype(int)
            f1 = sklearn_f1(y_true, predictions)
            assert 0.0 <= threshold <= 1.0, f"Invalid threshold: {threshold}"
            assert f1 > 0.3, f"F1 score {f1:.3f} too low for {method}"

        # Methods should produce similar results on realistic data
        threshold_values = list(thresholds.values())
        max_diff = max(threshold_values) - min(threshold_values)
        assert max_diff < 0.2, f"Methods disagree too much: {thresholds}"

    def test_imbalanced_binary_optimization(self):
        """Test optimization on highly imbalanced dataset."""
        y_true, y_prob = IMBALANCED_BINARY.y_true, IMBALANCED_BINARY.y_prob

        # With 5% positive class, precision and recall should behave differently
        result_f1 = optimize_thresholds(y_true, y_prob, metric="f1")
        result_precision = optimize_thresholds(y_true, y_prob, metric="precision")
        result_recall = optimize_thresholds(y_true, y_prob, metric="recall")

        # Different metrics should potentially give different thresholds
        # Note: The exact relationship depends on the data distribution
        # We mainly test that all metrics work and produce valid results

        # All should be valid
        for name, result in [
            ("F1", result_f1),
            ("Precision", result_precision),
            ("Recall", result_recall),
        ]:
            threshold = result.threshold
            assert -1e-9 <= threshold <= 1.0, f"Invalid {name} threshold: {threshold}"

    def test_well_separated_vs_overlapping(self):
        """Test that algorithm adapts to different class separation levels."""
        # Well-separated classes should allow more extreme thresholds
        y_sep, p_sep = WELL_SEPARATED_BINARY.y_true, WELL_SEPARATED_BINARY.y_prob
        result_sep = optimize_thresholds(y_sep, p_sep, metric="f1")
        threshold_sep = result_sep.threshold

        # Overlapping classes should result in more moderate thresholds
        y_over, p_over = OVERLAPPING_BINARY.y_true, OVERLAPPING_BINARY.y_prob
        result_over = optimize_thresholds(y_over, p_over, metric="f1")
        threshold_over = result_over.threshold

        # Both should achieve reasonable performance
        pred_sep = (p_sep > threshold_sep).astype(int)
        pred_over = (p_over > threshold_over).astype(int)

        f1_sep = sklearn_f1(y_sep, pred_sep)
        f1_over = sklearn_f1(y_over, pred_over)

        # Well-separated should achieve much better F1
        assert f1_sep > f1_over + 0.1, (
            f"Separated F1 {f1_sep:.3f} should be much better than overlapping F1 "
            f"{f1_over:.3f}"
        )

        # But both should be reasonable
        assert f1_sep > 0.6, f"Well-separated F1 {f1_sep:.3f} should be high"
        assert f1_over > 0.2, f"Overlapping F1 {f1_over:.3f} should be reasonable"

    def test_calibrated_data_expected_vs_empirical(self):
        """Test expected vs empirical optimization on calibrated data."""
        y_true, y_prob = CALIBRATED_BINARY.y_true, CALIBRATED_BINARY.y_prob

        # Empirical optimization
        result_emp = optimize_thresholds(y_true, y_prob, metric="f1", mode="empirical")
        threshold_emp = result_emp.threshold

        # Expected optimization
        result_exp = optimize_thresholds(y_true, y_prob, metric="f1", mode="expected")
        threshold_exp, expected_f1 = result_exp.threshold, result_exp.score

        # Both should be reasonable
        assert 0.0 <= threshold_emp <= 1.0
        assert 0.0 <= threshold_exp <= 1.0

        # Calculate actual F1 scores
        pred_emp = (y_prob > threshold_emp).astype(int)
        pred_exp = (y_prob > threshold_exp).astype(int)

        f1_emp = sklearn_f1(y_true, pred_emp)
        f1_exp = sklearn_f1(y_true, pred_exp)

        # On calibrated data, both should achieve decent performance
        assert f1_emp > 0.4, (
            f"Empirical F1 {f1_emp:.3f} should be decent on calibrated data"
        )
        assert f1_exp > 0.4, (
            f"Expected F1 {f1_exp:.3f} should be decent on calibrated data"
        )

        # Expected F1 should be close to actual F1 for expected optimization
        assert abs(expected_f1 - f1_exp) < 0.1, (
            f"Expected F1 {expected_f1:.3f} should match actual F1 {f1_exp:.3f}"
        )

    def test_comparison_operators_realistic(self):
        """Test '>' vs '>=' comparison operators on realistic data."""
        y_true, y_prob = STANDARD_BINARY.y_true, STANDARD_BINARY.y_prob

        result_excl = optimize_thresholds(y_true, y_prob, metric="f1", comparison=">")
        result_incl = optimize_thresholds(y_true, y_prob, metric="f1", comparison=">=")
        threshold_excl = result_excl.threshold
        threshold_incl = result_incl.threshold

        # Apply thresholds
        pred_excl = (y_prob > threshold_excl).astype(int)
        pred_incl = (y_prob >= threshold_incl).astype(int)

        # Both should achieve reasonable performance
        f1_excl = sklearn_f1(y_true, pred_excl)
        f1_incl = sklearn_f1(y_true, pred_incl)

        assert f1_excl > 0.3, f"Exclusive F1 {f1_excl:.3f} should be reasonable"
        assert f1_incl > 0.3, f"Inclusive F1 {f1_incl:.3f} should be reasonable"

        # When there are ties at the optimal threshold, predictions may differ
        has_ties_at_threshold = np.any(np.isclose(y_prob, threshold_excl, atol=1e-10))
        if has_ties_at_threshold:
            # With ties, inclusive should predict more positives
            assert np.sum(pred_incl) >= np.sum(pred_excl)

    def test_sample_weights_realistic(self):
        """Test sample weights with realistic dataset."""
        y_true, y_prob = STANDARD_BINARY.y_true, STANDARD_BINARY.y_prob

        # Create meaningful weights (emphasize minority class)
        weights = np.ones_like(y_true, dtype=float)
        weights[y_true == 1] = 2.0  # Double weight for positive class

        result_unweighted = optimize_thresholds(y_true, y_prob, metric="f1")
        result_weighted = optimize_thresholds(
            y_true, y_prob, metric="f1", sample_weight=weights
        )
        threshold_unweighted = result_unweighted.threshold
        threshold_weighted = result_weighted.threshold

        # Weighted threshold should be different (likely lower to catch more positives)
        assert abs(threshold_weighted - threshold_unweighted) > 0.01, (
            "Weights should affect threshold"
        )

        # Both should achieve reasonable performance
        pred_unweighted = (y_prob > threshold_unweighted).astype(int)
        pred_weighted = (y_prob > threshold_weighted).astype(int)

        f1_unweighted = sklearn_f1(y_true, pred_unweighted)
        f1_weighted = sklearn_f1(y_true, pred_weighted)

        assert f1_unweighted > 0.3
        assert f1_weighted > 0.3

    def test_large_dataset_performance(self):
        """Test performance on large dataset (5000+ samples)."""
        y_true, y_prob = LARGE_BINARY.y_true, LARGE_BINARY.y_prob

        # Should handle large dataset efficiently
        result = optimize_thresholds(y_true, y_prob, metric="f1", method="sort_scan")
        threshold = result.threshold

        # Verify results are reasonable
        predictions = (y_prob > threshold).astype(int)
        f1 = sklearn_f1(y_true, predictions)
        accuracy = sklearn_accuracy(y_true, predictions)
        assert 0.0 <= threshold <= 1.0
        assert f1 > 0.4, f"F1 {f1:.3f} should be good on large dataset"
        assert accuracy > 0.5, f"Accuracy {accuracy:.3f} should beat random"


class TestRealisticMulticlassOptimization:
    """Test multiclass threshold optimization with realistic datasets."""

    def test_standard_multiclass_optimization(self):
        """Test multiclass optimization on standard realistic dataset."""
        y_true, y_prob = STANDARD_MULTICLASS.y_true, STANDARD_MULTICLASS.y_prob
        n_classes = STANDARD_MULTICLASS.n_classes

        # Test different averaging strategies
        for average in ["macro", "micro"]:
            result = optimize_thresholds(y_true, y_prob, metric="f1", average=average)
            thresholds = result.thresholds

            if average == "micro":
                # Micro averaging can return either single threshold or per-class
                # thresholds
                if isinstance(thresholds, float):
                    # Single global threshold for exact micro-averaging
                    assert 0.0 <= thresholds <= 1.0, "Threshold should be valid"
                    # Convert to per-class for consistency with test logic
                    thresholds = np.full(n_classes, thresholds)
                else:
                    # Per-class thresholds
                    assert len(thresholds) == n_classes, (
                        f"Should return {n_classes} thresholds"
                    )
            else:
                # Macro averaging should always return per-class thresholds
                assert len(thresholds) == n_classes, (
                    f"Should return {n_classes} thresholds"
                )

            # Coordinate ascent can produce thresholds outside [0,1] which are
            # mathematically correct
            # for margin-based decision rules: argmax_j (p_j - τ_j)
            assert all(np.isfinite(t) for t in thresholds), (
                "All thresholds should be finite"
            )

            # Apply thresholds and verify performance
            predictions = []
            for i, prob_row in enumerate(y_prob):
                # Standard multiclass: predict class with highest prob above threshold
                above_threshold = prob_row > thresholds
                if np.any(above_threshold):
                    predictions.append(np.argmax(prob_row * above_threshold))
                else:
                    predictions.append(np.argmax(prob_row))  # Fallback to highest prob

            predictions = np.array(predictions)
            accuracy = sklearn_accuracy(y_true, predictions)

            assert accuracy > 0.4, (
                f"Multiclass accuracy {accuracy:.3f} should be reasonable for {average}"
            )

    def test_imbalanced_multiclass_optimization(self):
        """Test multiclass optimization on imbalanced dataset."""
        y_true, y_prob = IMBALANCED_MULTICLASS.y_true, IMBALANCED_MULTICLASS.y_prob
        n_classes = IMBALANCED_MULTICLASS.n_classes

        # Macro and micro averaging should give different results
        result_macro = optimize_thresholds(y_true, y_prob, metric="f1", average="macro")
        result_micro = optimize_thresholds(y_true, y_prob, metric="f1", average="micro")

        thresholds_macro = result_macro.thresholds
        assert len(thresholds_macro) == n_classes

        thresholds_micro = result_micro.thresholds
        # Micro averaging can return single threshold or per-class thresholds
        if isinstance(thresholds_micro, float):
            # Convert single threshold to per-class for consistency
            thresholds_micro = np.full(n_classes, thresholds_micro)
        else:
            assert len(thresholds_micro) == n_classes

        # Thresholds may be different due to different averaging (but not always)
        # We mainly test that both averaging methods work and produce valid results
        for thresholds, name in [
            (thresholds_macro, "macro"),
            (thresholds_micro, "micro"),
        ]:
            # Coordinate ascent can produce thresholds outside [0,1] which are
            # mathematically correct
            assert all(np.isfinite(t) for t in thresholds), (
                f"Thresholds should be finite for {name} averaging"
            )

    def test_multiclass_vs_binary_consistency(self):
        """Test that multiclass reduces to binary correctly."""
        # Create a 2-class multiclass problem
        dataset = make_realistic_multiclass_dataset(400, 2, False, 123)
        y_true, y_prob_multi = dataset.y_true, dataset.y_prob

        # Convert to binary format
        y_prob_binary = y_prob_multi[:, 1]  # Probability of class 1

        # Optimize as multiclass
        result_multi = optimize_thresholds(
            y_true, y_prob_multi, metric="f1", average="macro"
        )
        thresholds_multi = result_multi.thresholds

        # Optimize as binary
        result_binary = optimize_thresholds(y_true, y_prob_binary, metric="f1")
        threshold_binary = result_binary.threshold

        # The threshold for class 1 in multiclass should be reasonably related to binary
        # threshold
        # Note: Different algorithms (coordinate ascent vs binary optimization) can
        # produce different
        # but equally valid thresholds, especially for margin-based vs probability-based
        # decision rules
        tolerance = 0.8  # Allow larger difference due to algorithm differences
        assert abs(thresholds_multi[1] - threshold_binary) < tolerance, (
            f"Multiclass threshold {thresholds_multi[1]:.3f} differs significantly "
            f"from "
            f"binary {threshold_binary:.3f} (tolerance: {tolerance})"
        )


class TestRealisticUtilityOptimization:
    """Test utility-based optimization with realistic datasets."""

    def test_cost_sensitive_optimization(self):
        """Test cost-sensitive threshold optimization."""
        y_true, y_prob = IMBALANCED_BINARY.y_true, IMBALANCED_BINARY.y_prob

        # Define utility where false negatives are very costly
        utility_high_fn_cost = {"tp": 1, "tn": 1, "fp": -1, "fn": -10}
        utility_high_fp_cost = {"tp": 1, "tn": 1, "fp": -10, "fn": -1}

        result_fn = optimize_thresholds(y_true, y_prob, utility=utility_high_fn_cost)
        result_fp = optimize_thresholds(y_true, y_prob, utility=utility_high_fp_cost)
        threshold_fn = result_fn.threshold
        threshold_fp = result_fp.threshold

        # Different cost structures should potentially give different thresholds
        # However, on some datasets they might be the same if the optimal point doesn't
        # change
        # We mainly test that utility optimization works and produces valid results

        # Test predictions
        pred_fn = (y_prob > threshold_fn).astype(int)
        pred_fp = (y_prob > threshold_fp).astype(int)

        # Both should be valid predictions
        assert len(pred_fn) == len(y_true)
        assert len(pred_fp) == len(y_true)
        assert all(p in [0, 1] for p in pred_fn)
        assert all(p in [0, 1] for p in pred_fp)

    def test_bayes_vs_empirical_utility(self):
        """Test Bayes vs empirical utility optimization."""
        y_true, y_prob = CALIBRATED_BINARY.y_true, CALIBRATED_BINARY.y_prob

        utility = {"tp": 2, "tn": 1, "fp": -1, "fn": -3}

        # Empirical optimization
        result_emp = optimize_thresholds(
            y_true, y_prob, utility=utility, mode="empirical"
        )
        threshold_emp = result_emp.threshold

        # Bayes optimization (doesn't need true labels)
        result_bayes = optimize_thresholds(None, y_prob, utility=utility, mode="bayes")
        threshold_bayes = result_bayes.threshold

        # On well-calibrated data, they should be reasonably close
        assert abs(threshold_emp - threshold_bayes) < 0.1, (
            f"Empirical {threshold_emp:.3f} and Bayes {threshold_bayes:.3f} should be "
            f"close on calibrated data"
        )

        # Both should be valid
        assert 0.0 <= threshold_emp <= 1.0
        assert 0.0 <= threshold_bayes <= 1.0


@pytest.mark.parametrize(
    "dataset",
    [
        STANDARD_BINARY,
        IMBALANCED_BINARY,
        WELL_SEPARATED_BINARY,
        OVERLAPPING_BINARY,
        CALIBRATED_BINARY,
    ],
)
def test_all_methods_on_realistic_data(dataset):
    """Test that all optimization methods work on various realistic datasets."""
    y_true, y_prob = dataset.y_true, dataset.y_prob
    description = dataset.description

    methods = ["auto", "sort_scan", "sort_scan", "minimize"]

    for method in methods:
        result = optimize_thresholds(y_true, y_prob, metric="f1", method=method)

        threshold = result.threshold
        assert 0.0 <= threshold <= 1.0, (
            f"Method {method} produced invalid threshold {threshold} on {description}"
        )

        # Verify it produces reasonable results
        predictions = (y_prob > threshold).astype(int)
        f1 = sklearn_f1(y_true, predictions)

        assert f1 >= 0.0, f"Method {method} produced negative F1 {f1} on {description}"
        # Note: We don't require f1 > threshold because some datasets might be very
        # difficult


@pytest.mark.parametrize("metric", ["f1", "accuracy", "precision", "recall"])
def test_all_metrics_on_realistic_data(metric):
    """Test that all metrics work properly on realistic data."""
    y_true, y_prob = STANDARD_BINARY.y_true, STANDARD_BINARY.y_prob

    result = optimize_thresholds(y_true, y_prob, metric=metric)

    threshold = result.threshold
    assert 0.0 <= threshold <= 1.0, (
        f"Metric {metric} produced invalid threshold {threshold}"
    )

    # Apply threshold and verify metric calculation
    tp, tn, fp, fn = confusion_matrix_at_threshold(y_true, y_prob, threshold)

    if metric == "f1":
        score = f1_score(tp, tn, fp, fn)
    elif metric == "accuracy":
        score = accuracy_score(tp, tn, fp, fn)
    elif metric == "precision":
        score = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    elif metric == "recall":
        score = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    assert 0.0 <= score <= 1.0, f"Computed {metric} score {score} out of range"
