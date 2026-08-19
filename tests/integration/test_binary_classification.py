"""Integration tests for binary classification workflows.

This module tests end-to-end binary classification scenarios, including
sample weights, comparison operators, and method interactions.
"""

import numpy as np

from optimal_cutoffs import optimize_thresholds
from optimal_cutoffs.cv import cross_validate
from optimal_cutoffs.metrics_core import confusion_matrix_at_threshold
from tests.fixtures.assertions import (
    assert_method_consistency,
    assert_valid_confusion_matrix,
    assert_valid_metric_score,
    assert_valid_threshold,
)
from tests.fixtures.data_generators import (
    generate_binary_data,
    generate_imbalanced_data,
    generate_sample_weights,
    generate_tied_probabilities,
)


class TestBinaryClassificationWorkflows:
    """Test complete binary classification workflows."""

    def test_end_to_end_binary_optimization(self):
        """Test complete binary classification optimization workflow."""
        y_true, y_prob = generate_binary_data(100, random_state=42)

        # Test full workflow with different metrics
        for metric in ["f1", "accuracy", "precision", "recall"]:
            result = optimize_thresholds(y_true, y_prob, metric=metric)

            threshold = result.threshold
            assert_valid_threshold(threshold)

            # Compute confusion matrix and verify
            tp, tn, fp, fn = confusion_matrix_at_threshold(y_true, y_prob, threshold)
            assert_valid_confusion_matrix(tp, tn, fp, fn, total_samples=len(y_true))

            # Compute achieved metric score
            if metric == "f1":
                precision = tp / (tp + fp) if tp + fp > 0 else 0.0
                recall = tp / (tp + fn) if tp + fn > 0 else 0.0
                score = (
                    2 * precision * recall / (precision + recall)
                    if precision + recall > 0
                    else 0.0
                )
            elif metric == "accuracy":
                score = (tp + tn) / (tp + tn + fp + fn)
            elif metric == "precision":
                score = tp / (tp + fp) if tp + fp > 0 else 0.0
            elif metric == "recall":
                score = tp / (tp + fn) if tp + fn > 0 else 0.0

            assert_valid_metric_score(score, metric)

    def test_binary_with_multiple_methods(self):
        """Test binary classification with different optimization methods."""
        y_true, y_prob = generate_binary_data(50, random_state=42)

        methods = ["sort_scan", "minimize", "gradient", "auto"]
        results = {}

        for method in methods:
            result = optimize_thresholds(y_true, y_prob, metric="f1", method=method)
            threshold = result.threshold
            assert_valid_threshold(threshold)

            # Compute achieved F1 score
            tp, _tn, fp, fn = confusion_matrix_at_threshold(y_true, y_prob, threshold)
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0.0
            )

            results[method] = (threshold, f1)

        # All methods should achieve reasonable performance
        for method, (threshold, f1) in results.items():
            assert f1 >= 0.0, f"Method {method} achieved negative F1: {f1}"

    def test_binary_mode_comparison(self):
        """Test empirical vs expected mode optimization."""
        y_true, y_prob = generate_binary_data(50, random_state=42)

        # Empirical mode
        result = optimize_thresholds(y_true, y_prob, mode="empirical")
        threshold = result.threshold
        assert_valid_threshold(threshold)

        # Expected mode (now returns OptimizationResult)
        result = optimize_thresholds(y_true, y_prob, mode="expected")
        threshold = result.threshold
        score = result.score
        assert_valid_threshold(threshold)
        assert_valid_metric_score(score, "expected_f1")


class TestSampleWeights:
    """Test binary classification with sample weights."""

    def test_sample_weights_basic(self):
        """Test basic sample weight functionality."""
        y_true, y_prob = generate_binary_data(30, random_state=42)
        weights = generate_sample_weights(len(y_true), "random", random_state=42)

        result = optimize_thresholds(y_true, y_prob, metric="f1", sample_weight=weights)
        threshold = result.threshold
        assert_valid_threshold(threshold)

        # Compute weighted confusion matrix
        tp, tn, fp, fn = confusion_matrix_at_threshold(
            y_true, y_prob, threshold, weights
        )
        assert_valid_confusion_matrix(tp, tn, fp, fn, total_weight=np.sum(weights))

    def test_sample_weights_vs_expansion(self):
        """Test that weighted approach matches sample expansion."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.7])
        weights = np.array([2, 3, 1, 2])  # Integer weights for exact expansion

        # Weighted approach
        result_weighted = optimize_thresholds(
            y_true, y_prob, metric="accuracy", sample_weight=weights
        )
        threshold_weighted = result_weighted.threshold

        # Expansion approach
        y_expanded = np.repeat(y_true, weights)
        p_expanded = np.repeat(y_prob, weights)
        result_expanded = optimize_thresholds(y_expanded, p_expanded, metric="accuracy")
        threshold_expanded = result_expanded.threshold

        # Should be nearly identical
        assert_method_consistency(
            threshold_weighted,
            threshold_expanded,
            "weighted",
            "expanded",
            tolerance=1e-10,
        )

    def test_sample_weights_different_types(self):
        """Test sample weights with different weight distributions."""
        y_true, y_prob = generate_binary_data(40, random_state=42)

        weight_types = ["uniform", "random", "integer", "extreme"]

        for weight_type in weight_types:
            weights = generate_sample_weights(len(y_true), weight_type, random_state=42)

            result = optimize_thresholds(
                y_true, y_prob, metric="f1", sample_weight=weights
            )
            threshold = result.threshold
            assert_valid_threshold(threshold)

    def test_sample_weights_with_different_methods(self):
        """Test sample weights work with different optimization methods."""
        y_true, y_prob = generate_binary_data(30, random_state=42)
        weights = generate_sample_weights(len(y_true), "random", random_state=42)

        # Test methods that support weights
        methods = ["sort_scan", "minimize"]

        for method in methods:
            result = optimize_thresholds(
                y_true, y_prob, metric="f1", method=method, sample_weight=weights
            )
            threshold = result.threshold
            assert_valid_threshold(threshold)


class TestComparisonOperators:
    """Test binary classification with different comparison operators."""

    def test_comparison_operators_basic(self):
        """Test that different comparison operators work."""
        y_true, y_prob = generate_tied_probabilities(30, random_state=42)

        for comparison in [">", ">="]:
            result = optimize_thresholds(
                y_true, y_prob, metric="f1", comparison=comparison
            )
            threshold = result.threshold
            assert_valid_threshold(threshold)

    def test_comparison_operators_with_ties(self):
        """Test comparison operators specifically on tied data."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.3, 0.5, 0.7, 0.5])  # Two values at 0.5
        threshold = 0.5

        # With ">" (exclusive), prob=0.5 should be negative
        tp_gt, tn_gt, fp_gt, fn_gt = confusion_matrix_at_threshold(
            y_true, y_prob, threshold, comparison=">"
        )
        # Predictions: [0, 0, 1, 0] -> TP=0, TN=1, FP=1, FN=2
        assert (tp_gt, tn_gt, fp_gt, fn_gt) == (0, 1, 1, 2)

        # With ">=" (inclusive), prob=0.5 should be positive
        tp_gte, tn_gte, fp_gte, fn_gte = confusion_matrix_at_threshold(
            y_true, y_prob, threshold, comparison=">="
        )
        # Predictions: [0, 1, 1, 1] -> TP=2, TN=1, FP=1, FN=0
        assert (tp_gte, tn_gte, fp_gte, fn_gte) == (2, 1, 1, 0)

        # Results should be different
        assert (tp_gt, tn_gt, fp_gt, fn_gt) != (tp_gte, tn_gte, fp_gte, fn_gte)

    def test_comparison_operators_with_weights(self):
        """Test comparison operators with sample weights."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.3, 0.5, 0.7, 0.5])
        weights = np.array([1.0, 2.0, 1.0, 3.0])
        threshold = 0.5

        for comparison in [">", ">="]:
            tp, tn, fp, fn = confusion_matrix_at_threshold(
                y_true, y_prob, threshold, weights, comparison=comparison
            )

            # Should be float values due to weights
            assert isinstance(tp, float)
            assert_valid_confusion_matrix(tp, tn, fp, fn, total_weight=np.sum(weights))

    def test_comparison_operators_optimization(self):
        """Test that optimization works with both comparison operators."""
        y_true, y_prob = generate_tied_probabilities(40, random_state=42)

        thresholds = {}
        for comparison in [">", ">="]:
            result = optimize_thresholds(
                y_true, y_prob, metric="f1", comparison=comparison
            )
            thresholds[comparison] = result.threshold
            assert_valid_threshold(thresholds[comparison])

        # Thresholds might be different due to tie handling
        # Both should be valid optimization results


class TestImbalancedData:
    """Test binary classification on imbalanced datasets."""

    def test_highly_imbalanced_data(self):
        """Test optimization on highly imbalanced datasets."""
        y_true, y_prob = generate_imbalanced_data(
            1000, imbalance_ratio=0.01, random_state=42
        )

        # Should handle extreme imbalance
        result = optimize_thresholds(y_true, y_prob, metric="f1")
        threshold = result.threshold
        assert_valid_threshold(threshold)

        # Verify confusion matrix validity
        tp, tn, fp, fn = confusion_matrix_at_threshold(y_true, y_prob, threshold)
        assert_valid_confusion_matrix(tp, tn, fp, fn, total_samples=len(y_true))

    def test_imbalanced_different_metrics(self):
        """Test different metrics on imbalanced data."""
        y_true, y_prob = generate_imbalanced_data(
            500, imbalance_ratio=0.05, random_state=42
        )

        # Test metrics that handle imbalance differently
        metrics = ["f1", "accuracy", "precision", "recall"]

        for metric in metrics:
            result = optimize_thresholds(y_true, y_prob, metric=metric)
            threshold = result.threshold
            assert_valid_threshold(threshold)

            tp, tn, fp, fn = confusion_matrix_at_threshold(y_true, y_prob, threshold)
            assert_valid_confusion_matrix(tp, tn, fp, fn, total_samples=len(y_true))

    def test_imbalanced_with_weights(self):
        """Test imbalanced data with sample weights."""
        y_true, y_prob = generate_imbalanced_data(
            200, imbalance_ratio=0.1, random_state=42
        )

        # Generate weights that favor minority class
        weights = np.ones(len(y_true))
        weights[y_true == 1] *= 5.0  # Upweight positive class

        result = optimize_thresholds(y_true, y_prob, metric="f1", sample_weight=weights)
        threshold = result.threshold
        assert_valid_threshold(threshold)


class TestCrossValidation:
    """Test cross-validation functionality for binary classification."""

    def test_cross_validate_basic(self):
        """Test basic cross-validation threshold optimization."""
        y_true, y_prob = generate_binary_data(100, random_state=42)

        thresholds, scores = cross_validate(
            y_true, y_prob, method="sort_scan", cv=5, random_state=42
        )

        assert thresholds.shape == (5,)  # Binary thresholds are scalars per fold
        assert scores.shape == (5,)
        assert np.all((thresholds >= 0) & (thresholds <= 1))
        assert np.all((scores >= 0) & (scores <= 1))

    def test_cv_different_methods(self):
        """Test cross-validation with different optimization methods."""
        y_true, y_prob = generate_binary_data(80, random_state=42)

        methods = ["sort_scan", "minimize"]

        for method in methods:
            thresholds, scores = cross_validate(
                y_true, y_prob, method=method, cv=3, random_state=42
            )

            assert thresholds.shape[0] == 3
            assert len(scores) == 3
            for threshold in thresholds.ravel():
                assert_valid_threshold(threshold)
            for score in scores:
                assert_valid_metric_score(score, "f1")

    def test_cv_with_different_metrics(self):
        """Test cross-validation with different metrics."""
        y_true, y_prob = generate_binary_data(60, random_state=42)

        for metric in ["f1", "accuracy", "precision", "recall"]:
            thresholds, scores = cross_validate(
                y_true, y_prob, metric=metric, cv=3, random_state=42
            )

            assert len(thresholds) == 3
            assert len(scores) == 3


class TestMethodInteractions:
    """Test interactions between different optimization parameters."""

    def test_method_metric_combinations(self):
        """Test all combinations of methods and metrics."""
        y_true, y_prob = generate_binary_data(40, random_state=42)

        methods = ["sort_scan", "minimize", "gradient"]
        metrics = ["f1", "accuracy", "precision", "recall"]

        for method in methods:
            for metric in metrics:
                result = optimize_thresholds(
                    y_true, y_prob, method=method, metric=metric
                )
                threshold = result.threshold
                assert_valid_threshold(threshold)

    def test_method_comparison_combinations(self):
        """Test methods with different comparison operators."""
        y_true, y_prob = generate_tied_probabilities(30, random_state=42)

        methods = ["sort_scan", "minimize"]
        comparisons = [">", ">="]

        for method in methods:
            for comparison in comparisons:
                result = optimize_thresholds(
                    y_true, y_prob, method=method, comparison=comparison
                )
                threshold = result.threshold
                assert_valid_threshold(threshold)

    def test_weights_comparison_combinations(self):
        """Test sample weights with comparison operators."""
        y_true, y_prob = generate_tied_probabilities(25, random_state=42)
        weights = generate_sample_weights(len(y_true), "random", random_state=42)

        for comparison in [">", ">="]:
            result = optimize_thresholds(
                y_true, y_prob, comparison=comparison, sample_weight=weights
            )
            threshold = result.threshold
            assert_valid_threshold(threshold)


class TestBinaryPerformance:
    """Test performance characteristics of binary classification."""

    def test_optimization_performance_scaling(self):
        """Test that binary optimization scales reasonably."""
        import time

        sizes = [100, 500, 1000]
        times = []

        for size in sizes:
            y_true, y_prob = generate_binary_data(size, random_state=42)

            start_time = time.time()
            optimize_thresholds(y_true, y_prob, metric="f1")
            end_time = time.time()

            elapsed = end_time - start_time
            times.append(elapsed)

            # Should complete in reasonable time
            assert elapsed < 5.0, f"Optimization took {elapsed:.2f}s for {size} samples"

    def test_method_performance_comparison(self):
        """Test relative performance of optimization methods."""
        import time

        y_true, y_prob = generate_binary_data(500, random_state=42)

        methods = ["sort_scan", "minimize", "gradient"]
        times = {}

        for method in methods:
            start_time = time.time()
            optimize_thresholds(y_true, y_prob, metric="f1", method=method)
            end_time = time.time()

            times[method] = end_time - start_time

            # All methods should be reasonably fast
            assert times[method] < 10.0, f"Method {method} took {times[method]:.2f}s"


class TestBinaryEdgeCaseIntegration:
    """Test edge cases in binary classification integration."""

    def test_perfect_separation_integration(self):
        """Test complete workflow on perfectly separable data."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        # Should achieve perfect or near-perfect performance
        for metric in ["f1", "accuracy", "precision", "recall"]:
            result = optimize_thresholds(y_true, y_prob, metric=metric)
            threshold = result.threshold

            tp, tn, fp, fn = confusion_matrix_at_threshold(y_true, y_prob, threshold)

            if metric == "accuracy":
                score = (tp + tn) / (tp + tn + fp + fn)
                assert score >= 0.99  # Near perfect
            elif metric == "f1":
                precision = tp / (tp + fp) if tp + fp > 0 else 0.0
                recall = tp / (tp + fn) if tp + fn > 0 else 0.0
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    assert f1 >= 0.99

    def test_minimal_data_integration(self):
        """Test integration workflow on minimal datasets."""
        # Two samples, one of each class
        y_true = np.array([0, 1])
        y_prob = np.array([0.3, 0.7])

        result = optimize_thresholds(y_true, y_prob, metric="accuracy")
        threshold = result.threshold
        assert_valid_threshold(threshold)

        # Should achieve perfect classification
        tp, tn, _fp, _fn = confusion_matrix_at_threshold(y_true, y_prob, threshold)
        assert tp + tn == 2  # Perfect accuracy possible

    def test_all_same_predictions_integration(self):
        """Test integration when all samples have same predicted probability."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])

        for comparison in [">", ">="]:
            result = optimize_thresholds(
                y_true, y_prob, metric="f1", comparison=comparison
            )
            threshold = result.threshold
            assert_valid_threshold(threshold)

            tp, tn, fp, fn = confusion_matrix_at_threshold(
                y_true, y_prob, threshold, comparison=comparison
            )
            assert_valid_confusion_matrix(tp, tn, fp, fn, total_samples=len(y_true))
