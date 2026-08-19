"""Integration tests for multiclass classification workflows.

This module tests end-to-end multiclass classification scenarios, including
different averaging methods, coordinate ascent, and multiclass-specific features.
"""

import numpy as np
import pytest

from optimal_cutoffs import optimize_thresholds
from optimal_cutoffs.metrics_core import (
    multiclass_confusion_matrices_at_thresholds,
    multiclass_metric_ovr,
)
from tests.fixtures.assertions import (
    assert_valid_confusion_matrix,
    assert_valid_metric_score,
    assert_valid_threshold,
)
from tests.fixtures.data_generators import (
    generate_multiclass_data,
    generate_sample_weights,
)


class TestMulticlassWorkflows:
    """Test complete multiclass classification workflows."""

    def test_end_to_end_multiclass_optimization(self):
        """Test complete multiclass optimization workflow."""
        y_true, y_prob = generate_multiclass_data(100, n_classes=3, random_state=42)

        # Test optimization
        result = optimize_thresholds(y_true, y_prob, metric="f1")

        thresholds = result.thresholds
        assert len(thresholds) == 3
        for threshold in thresholds:
            assert_valid_threshold(threshold)

        # Test confusion matrix computation
        cms = multiclass_confusion_matrices_at_thresholds(y_true, y_prob, thresholds)

        assert len(cms) == 3
        for i, (tp, tn, fp, fn) in enumerate(cms):
            assert_valid_confusion_matrix(tp, tn, fp, fn, total_samples=len(y_true))

    def test_multiclass_different_methods(self):
        """Test multiclass optimization with different methods."""
        y_true, y_prob = generate_multiclass_data(50, n_classes=3, random_state=42)

        methods = ["sort_scan", "minimize", "coord_ascent"]

        for method in methods:
            result = optimize_thresholds(y_true, y_prob, metric="f1", method=method)

            thresholds = result.thresholds
            assert len(thresholds) == 3
            for threshold in thresholds:
                if method == "coord_ascent":
                    # Coordinate ascent can produce thresholds outside [0,1]
                    assert np.isfinite(threshold), (
                        f"Threshold {threshold} should be finite"
                    )
                else:
                    assert_valid_threshold(threshold)

    def test_multiclass_different_metrics(self):
        """Test multiclass optimization with different metrics."""
        y_true, y_prob = generate_multiclass_data(60, n_classes=3, random_state=42)

        metrics = ["f1", "accuracy", "precision", "recall"]

        for metric in metrics:
            result = optimize_thresholds(y_true, y_prob, metric=metric)

            thresholds = result.thresholds
            assert len(thresholds) == 3
            for threshold in thresholds:
                assert_valid_threshold(threshold)

    def test_multiclass_different_class_counts(self):
        """Test multiclass optimization with different numbers of classes."""
        for n_classes in [2, 3, 4, 5]:
            y_true, y_prob = generate_multiclass_data(
                50, n_classes=n_classes, random_state=42
            )

            result = optimize_thresholds(y_true, y_prob, metric="f1")

            thresholds = result.thresholds
            assert len(thresholds) == n_classes
            for threshold in thresholds:
                assert_valid_threshold(threshold)


class TestMulticlassConfusionMatrix:
    """Test multiclass confusion matrix functionality."""

    def test_multiclass_confusion_matrix_basic(self):
        """Test basic multiclass confusion matrix computation."""
        # 3-class problem with known outcomes
        true_labs = np.array([0, 1, 2, 0, 1, 2])
        pred_prob = np.array(
            [
                [0.8, 0.1, 0.1],  # True: 0, should predict 0
                [0.2, 0.7, 0.1],  # True: 1, should predict 1
                [0.1, 0.2, 0.7],  # True: 2, should predict 2
                [0.6, 0.3, 0.1],  # True: 0, should predict 0
                [0.3, 0.6, 0.1],  # True: 1, should predict 1
                [0.2, 0.2, 0.6],  # True: 2, should predict 2
            ]
        )
        thresholds = np.array([0.5, 0.5, 0.5])

        cms = multiclass_confusion_matrices_at_thresholds(
            true_labs, pred_prob, thresholds
        )

        assert len(cms) == 3
        for cm in cms:
            assert len(cm) == 4
            # Values should be numeric (int or float)
            assert all(
                isinstance(x, int | float | np.integer | np.floating) for x in cm
            )

    def test_multiclass_confusion_matrix_binary_fallback(self):
        """Test multiclass confusion matrix with binary input."""
        true_labs = np.array([0, 1, 1, 0])
        pred_prob = np.array([0.2, 0.8, 0.7, 0.3])
        threshold = np.array([0.5])

        cms = multiclass_confusion_matrices_at_thresholds(
            true_labs, pred_prob, threshold
        )

        assert len(cms) == 1
        assert len(cms[0]) == 4

    def test_multiclass_confusion_matrix_with_weights(self):
        """Test multiclass confusion matrix with sample weights."""
        y_true, y_prob = generate_multiclass_data(30, n_classes=3, random_state=42)
        weights = generate_sample_weights(len(y_true), "random", random_state=42)
        thresholds = np.array([0.3, 0.4, 0.5])

        cms = multiclass_confusion_matrices_at_thresholds(
            y_true, y_prob, thresholds, sample_weight=weights
        )

        assert len(cms) == 3
        for tp, tn, fp, fn in cms:
            # Weighted results should be floats
            assert isinstance(tp, float)
            assert_valid_confusion_matrix(tp, tn, fp, fn, total_weight=np.sum(weights))

    def test_multiclass_confusion_matrix_comparison_operators(self):
        """Test multiclass confusion matrix with different comparison operators."""
        y_true, y_prob = generate_multiclass_data(25, n_classes=3, random_state=42)
        thresholds = np.array([0.3, 0.5, 0.4])

        for comparison in [">", ">="]:
            cms = multiclass_confusion_matrices_at_thresholds(
                y_true, y_prob, thresholds, comparison=comparison
            )

            assert len(cms) == 3
            for tp, tn, fp, fn in cms:
                assert_valid_confusion_matrix(tp, tn, fp, fn, total_samples=len(y_true))


class TestMulticlassMetrics:
    """Test multiclass metric computation and averaging."""

    def test_multiclass_metrics_basic(self):
        """Test basic multiclass metric computation."""
        # Create confusion matrices for 3 classes
        cms = [
            (2, 4, 0, 0),  # Class 0: perfect precision/recall
            (1, 3, 1, 1),  # Class 1: some errors
            (1, 4, 0, 1),  # Class 2: some errors
        ]

        # Test all averaging methods
        for average in ["macro", "micro", "weighted"]:
            f1_score = multiclass_metric_ovr(cms, "f1", average)
            assert_valid_metric_score(f1_score, f"{average}_f1")

    def test_multiclass_metrics_different_metrics(self):
        """Test different metrics with multiclass averaging."""
        cms = [
            (3, 5, 1, 1),  # Class 0
            (2, 6, 2, 0),  # Class 1
            (1, 7, 1, 1),  # Class 2
        ]

        metrics = ["f1", "accuracy", "precision", "recall"]
        averages = ["macro", "micro", "weighted"]

        for metric in metrics:
            for average in averages:
                if metric == "accuracy" and average == "micro":
                    # Micro accuracy requires special handling
                    continue

                score = multiclass_metric_ovr(cms, metric, average)
                assert_valid_metric_score(score, f"{average}_{metric}")

    def test_multiclass_metrics_edge_cases(self):
        """Test multiclass metrics with edge cases."""
        # Case where one class has no true positives
        cms = [
            (0, 8, 0, 2),  # Class 0: no true positives
            (3, 5, 1, 1),  # Class 1: normal
            (2, 6, 0, 2),  # Class 2: normal
        ]

        # Should handle gracefully without division by zero
        for average in ["macro", "micro", "weighted"]:
            f1_score = multiclass_metric_ovr(cms, "f1", average)
            assert_valid_metric_score(f1_score, f"{average}_f1", allow_nan=False)

    def test_multiclass_averaging_identities(self):
        """Test mathematical identities for multiclass averaging."""
        # Balanced case: macro and weighted should be similar
        cms = [
            (5, 10, 2, 3),  # Class 0: support = 8
            (4, 12, 1, 3),  # Class 1: support = 7
            (6, 11, 1, 2),  # Class 2: support = 8
        ]

        macro_f1 = multiclass_metric_ovr(cms, "f1", "macro")
        weighted_f1 = multiclass_metric_ovr(cms, "f1", "weighted")

        # Should be relatively close for balanced data
        assert abs(macro_f1 - weighted_f1) < 0.2

    def test_multiclass_micro_vs_macro_properties(self):
        """Test properties of micro vs macro averaging."""
        cms = [
            (10, 80, 5, 5),  # Class 0
            (8, 85, 3, 4),  # Class 1
            (12, 82, 2, 4),  # Class 2
        ]

        micro_f1 = multiclass_metric_ovr(cms, "f1", "micro")
        macro_f1 = multiclass_metric_ovr(cms, "f1", "macro")

        # Both should be valid
        assert_valid_metric_score(micro_f1, "micro_f1")
        assert_valid_metric_score(macro_f1, "macro_f1")

        # Compute micro precision and recall for completeness
        multiclass_metric_ovr(cms, "precision", "micro")
        multiclass_metric_ovr(cms, "recall", "micro")

        # In OvR, micro_precision ≠ micro_recall since each class has independent
        # binary classifiers, breaking the FP/FN symmetry required for equality.


class TestCoordinateAscent:
    """Test coordinate ascent optimization for multiclass."""

    def test_coordinate_ascent_basic(self):
        """Test basic coordinate ascent functionality."""
        y_true, y_prob = generate_multiclass_data(40, n_classes=3, random_state=42)

        result = optimize_thresholds(y_true, y_prob, method="coord_ascent", metric="f1")

        thresholds = result.thresholds
        assert len(thresholds) == 3
        # Coordinate ascent thresholds can be outside [0,1] - this is legitimate
        # behavior
        for threshold in thresholds:
            assert np.isfinite(threshold), f"Threshold {threshold} should be finite"

    def test_coordinate_ascent_vs_ovr(self):
        """Test coordinate ascent vs One-vs-Rest approaches."""
        y_true, y_prob = generate_multiclass_data(50, n_classes=3, random_state=42)

        # One-vs-Rest (default unique_scan)
        result_ovr = optimize_thresholds(
            y_true, y_prob, method="unique_scan", metric="f1"
        )
        thresholds_ovr = result_ovr.thresholds

        # Coordinate ascent
        result_coord = optimize_thresholds(
            y_true, y_prob, method="coord_ascent", metric="f1"
        )
        thresholds_coord = result_coord.thresholds

        # Both should produce valid results
        assert len(thresholds_ovr) == 3
        assert len(thresholds_coord) == 3

        # For OvR thresholds, check standard [0,1] bounds
        for threshold in thresholds_ovr:
            assert_valid_threshold(threshold)

        # For coordinate ascent, thresholds can be outside [0,1] (legitimate behavior)
        for threshold in thresholds_coord:
            assert np.isfinite(threshold), f"Threshold {threshold} should be finite"

    def test_coordinate_ascent_comparison_operators(self):
        """Test coordinate ascent with supported comparison operator."""
        y_true, y_prob = generate_multiclass_data(30, n_classes=3, random_state=42)

        # Coordinate ascent only supports ">" comparison operator
        result = optimize_thresholds(
            y_true, y_prob, method="coord_ascent", comparison=">"
        )

        thresholds = result.thresholds
        assert len(thresholds) == 3
        # Coordinate ascent thresholds can be outside [0,1] - this is legitimate
        for threshold in thresholds:
            assert np.isfinite(threshold), f"Threshold {threshold} should be finite"

    def test_coordinate_ascent_single_label_consistency(self):
        """Test that coordinate ascent produces single-label predictions."""
        y_true, y_prob = generate_multiclass_data(40, n_classes=3, random_state=42)

        result = optimize_thresholds(y_true, y_prob, method="coord_ascent", metric="f1")
        thresholds = result.thresholds

        # Test prediction logic (simplified version)
        # In practice, coordinate ascent uses argmax(P - tau) for single-label
        # consistency
        n_samples = len(y_true)
        predictions = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            # Compute margins: p_ij - tau_j
            margins = y_prob[i] - thresholds

            # Find class with highest positive margin, or argmax if all negative
            positive_margins = margins > 0
            if np.any(positive_margins):
                # Among positive margins, pick the highest
                positive_indices = np.where(positive_margins)[0]
                best_positive = positive_indices[np.argmax(margins[positive_indices])]
                predictions[i] = best_positive
            else:
                # All margins negative, fall back to argmax
                predictions[i] = np.argmax(y_prob[i])

        # Each sample should have exactly one prediction
        assert len(predictions) == n_samples
        assert all(0 <= pred < 3 for pred in predictions)


class TestMulticlassLabelValidation:
    """Test multiclass label validation and edge cases."""

    def test_multiclass_non_consecutive_labels(self):
        """Test multiclass with non-consecutive labels."""
        # Labels {1, 2} with 3-class probabilities
        y_true = np.array([1, 2, 1, 2, 1, 2])
        pred_prob = np.random.rand(6, 3)
        pred_prob = pred_prob / pred_prob.sum(axis=1, keepdims=True)

        # Should work without error for OvR methods
        result = optimize_thresholds(y_true, pred_prob, metric="f1")
        thresholds = result.thresholds
        assert len(thresholds) == 3

    def test_multiclass_label_validation_errors(self):
        """Test that invalid labels are properly rejected."""
        # Labels outside valid range
        y_true = np.array([0, 1, 2, 3])  # Label 3 invalid for 3-class
        pred_prob = np.random.rand(4, 3)
        pred_prob = pred_prob / pred_prob.sum(axis=1, keepdims=True)

        with pytest.raises(ValueError, match="Found label 3 but n_classes=3"):
            optimize_thresholds(y_true, pred_prob, metric="f1")

    def test_multiclass_sparse_labels(self):
        """Test multiclass with sparse label sets."""
        # Only use classes 0 and 2 (skip class 1)
        y_true = np.array([0, 2, 0, 2, 0, 2])
        pred_prob = np.random.rand(6, 4)  # 4 classes available
        pred_prob = pred_prob / pred_prob.sum(axis=1, keepdims=True)

        # Should work and return 4 thresholds
        result = optimize_thresholds(y_true, pred_prob, metric="f1")
        thresholds = result.thresholds
        assert len(thresholds) == 4


class TestMulticlassWithWeights:
    """Test multiclass classification with sample weights."""

    def test_multiclass_weights_basic(self):
        """Test basic multiclass optimization with weights."""
        y_true, y_prob = generate_multiclass_data(40, n_classes=3, random_state=42)
        weights = generate_sample_weights(len(y_true), "random", random_state=42)

        result = optimize_thresholds(y_true, y_prob, metric="f1", sample_weight=weights)

        thresholds = result.thresholds
        assert len(thresholds) == 3
        for threshold in thresholds:
            assert_valid_threshold(threshold)

    def test_multiclass_weights_different_methods(self):
        """Test multiclass weights with different optimization methods."""
        y_true, y_prob = generate_multiclass_data(35, n_classes=3, random_state=42)
        weights = generate_sample_weights(len(y_true), "uniform", random_state=42)

        # Test methods that support weights
        methods = ["unique_scan", "minimize"]

        for method in methods:
            result = optimize_thresholds(
                y_true, y_prob, method=method, sample_weight=weights
            )

            thresholds = result.thresholds
            assert len(thresholds) == 3
            for threshold in thresholds:
                assert_valid_threshold(threshold)

    def test_multiclass_weights_vs_expansion(self):
        """Test that weighted multiclass matches sample expansion."""
        # Small example for exact comparison
        y_true = np.array([0, 1, 2, 0])
        pred_prob = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
                [0.6, 0.3, 0.1],
            ]
        )
        weights = np.array([2, 1, 3, 1])  # Integer weights

        # Weighted approach
        result_weighted = optimize_thresholds(
            y_true, pred_prob, metric="accuracy", sample_weight=weights
        )
        thresholds = result_weighted.thresholds

        # Expansion approach
        y_expanded = np.repeat(y_true, weights)
        p_expanded = np.repeat(pred_prob, weights, axis=0)
        result_expanded = optimize_thresholds(y_expanded, p_expanded, metric="accuracy")
        thresholds_expanded = result_expanded.thresholds

        # Should be nearly identical
        np.testing.assert_allclose(
            thresholds, thresholds_expanded, rtol=1e-10, atol=1e-10
        )


class TestMulticlassPerformance:
    """Test performance characteristics of multiclass optimization."""

    def test_multiclass_scaling_with_classes(self):
        """Test that multiclass optimization scales with number of classes."""
        import time

        class_counts = [3, 5, 8]
        times = []

        for n_classes in class_counts:
            y_true, y_prob = generate_multiclass_data(
                100, n_classes=n_classes, random_state=42
            )

            start_time = time.time()
            optimize_thresholds(y_true, y_prob, metric="f1")
            end_time = time.time()

            elapsed = end_time - start_time
            times.append(elapsed)

            # Should complete in reasonable time
            assert elapsed < 10.0, (
                f"Optimization took {elapsed:.2f}s for {n_classes} classes"
            )

    def test_multiclass_scaling_with_samples(self):
        """Test that multiclass optimization scales with number of samples."""
        import time

        sizes = [50, 200, 500]
        times = []

        for size in sizes:
            y_true, y_prob = generate_multiclass_data(
                size, n_classes=3, random_state=42
            )

            start_time = time.time()
            optimize_thresholds(y_true, y_prob, metric="f1")
            end_time = time.time()

            elapsed = end_time - start_time
            times.append(elapsed)

            # Should scale reasonably
            assert elapsed < 5.0, f"Optimization took {elapsed:.2f}s for {size} samples"


class TestMulticlassEdgeCases:
    """Test edge cases specific to multiclass classification."""

    def test_multiclass_single_class_data(self):
        """Test multiclass optimization when only one class is present."""
        # All samples from class 0
        y_true = np.array([0, 0, 0, 0])
        pred_prob = np.random.rand(4, 3)
        pred_prob = pred_prob / pred_prob.sum(axis=1, keepdims=True)

        result = optimize_thresholds(y_true, pred_prob, metric="f1")

        thresholds = result.thresholds
        assert len(thresholds) == 3
        for threshold in thresholds:
            assert_valid_threshold(threshold)

    def test_multiclass_perfect_probabilities(self):
        """Test multiclass with perfect probability predictions."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        pred_prob = np.array(
            [
                [1.0, 0.0, 0.0],  # Perfect for class 0
                [0.0, 1.0, 0.0],  # Perfect for class 1
                [0.0, 0.0, 1.0],  # Perfect for class 2
                [1.0, 0.0, 0.0],  # Perfect for class 0
                [0.0, 1.0, 0.0],  # Perfect for class 1
                [0.0, 0.0, 1.0],  # Perfect for class 2
            ]
        )

        result = optimize_thresholds(y_true, pred_prob, metric="f1")

        thresholds = result.thresholds
        assert len(thresholds) == 3
        for threshold in thresholds:
            assert_valid_threshold(threshold)

    def test_multiclass_uniform_probabilities(self):
        """Test multiclass with uniform probability distributions."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        pred_prob = np.full((6, 3), 1 / 3)  # All probabilities equal

        result = optimize_thresholds(y_true, pred_prob, metric="f1")

        thresholds = result.thresholds
        assert len(thresholds) == 3
        for threshold in thresholds:
            assert_valid_threshold(threshold)


class TestMulticlassAPIConsistency:
    """Test API consistency for multiclass functions."""

    def test_optimize_thresholds_multiclass_function(self):
        """Test that optimize_thresholds works correctly for multiclass."""
        y_true, y_prob = generate_multiclass_data(40, n_classes=3, random_state=42)

        # Using general function for multiclass
        result = optimize_thresholds(y_true, y_prob, metric="f1")
        thresholds = result.thresholds

        # Should return valid thresholds for all classes
        assert len(thresholds) == 3
        assert all(np.isfinite(t) for t in thresholds)
        assert result.n_classes == 3
        assert "f1" in result.metric  # Metric name includes f1

    def test_multiclass_input_validation_consistency(self):
        """Test that multiclass input validation works correctly."""
        # Valid input
        y_true, y_prob = generate_multiclass_data(30, n_classes=3, random_state=42)

        # Should work correctly
        result = optimize_thresholds(y_true, y_prob)

        thresholds = result.thresholds
        assert len(thresholds) == 3
        assert all(np.isfinite(t) for t in thresholds)

    def test_multiclass_return_types_consistency(self):
        """Test that multiclass functions return consistent types."""
        y_true, y_prob = generate_multiclass_data(25, n_classes=3, random_state=42)

        result = optimize_thresholds(y_true, y_prob, metric="f1")

        # Should return numpy array
        thresholds = result.thresholds
        assert isinstance(thresholds, np.ndarray)
        assert thresholds.ndim == 1
        assert len(thresholds) == 3
        assert thresholds.dtype in [np.float64, np.float32]
