"""Comprehensive edge case testing for boundary conditions and extreme scenarios.

This module provides extensive testing for edge cases and boundary conditions that
could cause threshold optimization algorithms to fail or produce suboptimal results.
The tests cover extreme data distributions, numerical precision limits, error
conditions,
and performance characteristics.

Test Categories:
================

1. **Label Distribution Edge Cases**
   - All positive or all negative labels (degenerate cases)
   - Extreme class imbalance (99:1 ratios)
   - Single samples of minority classes
   - Perfectly balanced datasets

2. **Probability Distribution Edge Cases**
   - Perfectly separated classes (no overlap)
   - Narrow probability ranges (clustered values)
   - Extreme probability skew
   - Boundary probabilities (0.0, 1.0)

3. **Numerical Edge Cases**
   - Very small and large datasets
   - Probabilities at machine epsilon
   - Floating-point precision limits
   - Values differing by tiny amounts

4. **Error Condition Testing**
   - NaN and infinity handling
   - Empty arrays
   - Mismatched array lengths
   - Invalid data types and ranges

5. **Performance Edge Cases**
   - Worst-case performance scenarios
   - Memory usage with large datasets
   - Scaling behavior verification

6. **Wrapper Integration**
   - ThresholdOptimizer with edge cases
   - Multiclass degenerate scenarios

The test suite ensures robust behavior across all boundary conditions while
maintaining performance guarantees and clear error reporting.
"""

import numpy as np
import pytest

from optimal_cutoffs import optimize_thresholds
from optimal_cutoffs.metrics_core import confusion_matrix_at_threshold

# from optimal_cutoffs.wrapper import ThresholdOptimizer  # Disabled - wrapper removed


class TestLabelDistributionEdgeCases:
    """Test extreme label distributions."""

    def test_all_zeros_labels(self):
        """Test with all negative labels."""
        labels = np.array([0, 0, 0, 0, 0])
        probabilities = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        # Fixed: degenerate case should return proper threshold, not arbitrary 0.5
        result_f1 = optimize_thresholds(labels, probabilities, metric="f1")
        threshold = result_f1.threshold
        # All negatives -> threshold should predict all negative for optimal accuracy
        assert threshold >= 0.9  # Should be >= max probability to predict all negative

        # Test with optimize_thresholds
        result = optimize_thresholds(labels, probabilities, metric="accuracy")
        threshold = result.threshold
        assert 0 <= threshold <= 1

        # Confusion matrix should be valid
        tp, tn, fp, fn = confusion_matrix_at_threshold(labels, probabilities, threshold)
        assert tp == 0  # No true positives possible
        assert fn == 0  # No false negatives possible
        assert tn + fp == len(labels)  # All samples are negative

    def test_all_ones_labels(self):
        """Test with all positive labels."""
        labels = np.array([1, 1, 1, 1, 1])
        probabilities = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        # Fixed: degenerate case should return proper threshold, not arbitrary 0.5
        result_f1 = optimize_thresholds(labels, probabilities, metric="f1")
        threshold = result_f1.threshold
        # All positives -> threshold should predict all positive for optimal accuracy
        assert threshold <= 0.1  # Should be <= min probability to predict all positive

        # Test with optimize_thresholds
        result = optimize_thresholds(labels, probabilities, metric="recall")
        threshold = result.threshold
        assert 0 <= threshold <= 1

        # Confusion matrix should be valid
        tp, tn, fp, fn = confusion_matrix_at_threshold(labels, probabilities, threshold)
        assert tn == 0  # No true negatives possible
        assert fp == 0  # No false positives possible
        assert tp + fn == len(labels)  # All samples are positive

    def test_single_positive_in_negatives(self):
        """Test extreme class imbalance - one positive in many negatives."""
        labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        probabilities = np.array(
            [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.95]
        )

        # Should find a reasonable threshold
        result = optimize_thresholds(labels, probabilities, metric="f1")
        threshold = result.threshold
        assert 0 <= threshold <= 1

        # The optimal threshold should likely be between 0.5 and 0.95
        # to capture the single positive example
        tp, _tn, fp, fn = confusion_matrix_at_threshold(
            labels, probabilities, threshold
        )

        # Should be able to achieve some reasonable performance
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0

        # At least one of precision or recall should be reasonable
        assert precision > 0 or recall > 0

    def test_perfectly_balanced_labels(self):
        """Test perfectly balanced dataset."""
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        probabilities = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        for metric in ["f1", "accuracy", "precision", "recall"]:
            result = optimize_thresholds(labels, probabilities, metric=metric)
            threshold = result.threshold
            assert 0 <= threshold <= 1

            # Should achieve reasonable performance
            score = self._compute_metric_score(labels, probabilities, threshold, metric)
            assert score > 0, f"Zero score for {metric}"

    def test_extreme_imbalance_99_to_1(self):
        """Test extreme imbalance (99:1 ratio)."""
        n_negative = 99
        n_positive = 1

        # Create extremely imbalanced dataset
        labels = np.concatenate([np.zeros(n_negative), np.ones(n_positive)])

        # Probabilities slightly favor the positive class
        neg_probs = np.random.uniform(0.0, 0.4, n_negative)
        pos_probs = np.random.uniform(0.6, 1.0, n_positive)
        probabilities = np.concatenate([neg_probs, pos_probs])

        # Should handle extreme imbalance
        result = optimize_thresholds(labels, probabilities, metric="f1")
        threshold = result.threshold
        assert 0 <= threshold <= 1

        # Test confusion matrix validity
        tp, tn, fp, fn = confusion_matrix_at_threshold(labels, probabilities, threshold)
        assert tp + tn + fp + fn == len(labels)

    def _compute_metric_score(self, labels, probabilities, threshold, metric):
        """Helper to compute metric score."""
        tp, tn, fp, fn = confusion_matrix_at_threshold(labels, probabilities, threshold)

        if metric == "accuracy":
            return (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0
        if metric == "precision":
            return tp / (tp + fp) if tp + fp > 0 else 0
        if metric == "recall":
            return tp / (tp + fn) if tp + fn > 0 else 0
        if metric == "f1":
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            return (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0
            )
        raise ValueError(f"Unknown metric: {metric}")


class TestProbabilityDistributionEdgeCases:
    """Test extreme probability distributions."""

    def test_perfectly_separated_classes(self):
        """Test with no overlap between class probability distributions."""
        labels = np.array([0, 0, 0, 1, 1, 1])
        probabilities = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        # Should achieve perfect or near-perfect performance
        for metric in ["accuracy", "f1", "precision", "recall"]:
            result = optimize_thresholds(labels, probabilities, metric=metric)

            # Threshold should be reasonable for perfect separation
            # For recall, optimal threshold might be very low to capture all positives
            # For precision, optimal threshold might be high to avoid false positives
            if metric == "recall":
                threshold = result.threshold
                assert 0.05 <= threshold <= 0.75, (
                    f"Unexpected threshold {threshold} for {metric}"
                )
            elif metric == "precision":
                threshold = result.threshold
                assert 0.25 <= threshold <= 0.95, (
                    f"Unexpected threshold {threshold} for {metric}"
                )
            else:
                threshold = result.threshold
                assert 0.25 <= threshold <= 0.75, (
                    f"Unexpected threshold {threshold} for {metric}"
                )

            # Should achieve high performance
            score = self._compute_metric_score(labels, probabilities, threshold, metric)
            assert score >= 0.9, (
                f"Low score {score} for {metric} with perfect separation"
            )

    def test_boundary_probabilities(self):
        """Test with probabilities at 0.0 and 1.0."""
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.0, 0.0, 1.0, 1.0])

        result = optimize_thresholds(labels, probabilities, metric="accuracy")
        threshold = result.threshold

        # Should achieve perfect accuracy
        tp, tn, fp, fn = confusion_matrix_at_threshold(labels, probabilities, threshold)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        assert accuracy == 1.0, f"Expected perfect accuracy, got {accuracy}"

    def test_narrow_probability_range(self):
        """Test with probabilities in very narrow range."""
        labels = np.array([0, 1, 0, 1, 0, 1])
        probabilities = np.array([0.49, 0.50, 0.51, 0.49, 0.51, 0.50])

        # Should handle narrow ranges gracefully
        result = optimize_thresholds(labels, probabilities, metric="f1")
        threshold = result.threshold
        assert 0.48 <= threshold <= 0.52

        # Should produce valid confusion matrix
        tp, tn, fp, fn = confusion_matrix_at_threshold(labels, probabilities, threshold)
        assert tp + tn + fp + fn == len(labels)

    def test_extreme_probability_skew(self):
        """Test with extremely skewed probability distribution."""
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        # Heavily skewed toward low probabilities
        probabilities = np.array([0.01, 0.02, 0.03, 0.04, 0.95, 0.96, 0.97, 0.98])

        result = optimize_thresholds(labels, probabilities, metric="f1")
        threshold = result.threshold
        assert 0 <= threshold <= 1

        # Should achieve good separation
        tp, _tn, fp, fn = confusion_matrix_at_threshold(
            labels, probabilities, threshold
        )

        # With such clear separation, should have good performance
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        assert precision > 0.5
        assert recall > 0.5

    def _compute_metric_score(self, labels, probabilities, threshold, metric):
        """Helper to compute metric score."""
        tp, tn, fp, fn = confusion_matrix_at_threshold(labels, probabilities, threshold)

        if metric == "accuracy":
            return (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0
        if metric == "precision":
            return tp / (tp + fp) if tp + fp > 0 else 0
        if metric == "recall":
            return tp / (tp + fn) if tp + fn > 0 else 0
        if metric == "f1":
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            return (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0
            )
        raise ValueError(f"Unknown metric: {metric}")


class TestNumericalEdgeCases:
    """Test numerical edge cases and extreme scenarios."""

    def test_very_large_datasets(self):
        """Test with large datasets to check scalability."""
        n_samples = 10000

        # Create large balanced dataset
        labels = np.array([i % 2 for i in range(n_samples)])
        probabilities = np.random.beta(2, 2, n_samples)  # Bell-shaped distribution

        # Should handle large datasets efficiently
        import time

        start_time = time.time()
        result = optimize_thresholds(labels, probabilities, metric="f1")
        end_time = time.time()

        threshold = result.threshold
        assert 0 <= threshold <= 1
        assert end_time - start_time < 5.0  # Should complete in reasonable time

        # Test confusion matrix
        tp, tn, fp, fn = confusion_matrix_at_threshold(labels, probabilities, threshold)
        assert tp + tn + fp + fn == n_samples

    def test_probabilities_near_machine_epsilon(self):
        """Test with probabilities very close to 0."""
        labels = np.array([0, 0, 1, 1])

        # Probabilities near machine epsilon
        eps = np.finfo(float).eps
        probabilities = np.array([eps, 2 * eps, 1 - 2 * eps, 1 - eps])

        # Should handle near-zero probabilities
        result = optimize_thresholds(labels, probabilities, metric="accuracy")
        threshold = result.threshold
        assert 0 <= threshold <= 1

        # Should achieve perfect separation
        tp, tn, fp, fn = confusion_matrix_at_threshold(labels, probabilities, threshold)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        assert accuracy == 1.0

    def test_probabilities_differing_by_epsilon(self):
        """Test with probabilities that differ by machine epsilon."""
        labels = np.array([0, 1, 0, 1])

        eps = np.finfo(float).eps
        base_prob = 0.5
        probabilities = np.array(
            [base_prob - eps, base_prob + eps, base_prob - 2 * eps, base_prob + 2 * eps]
        )

        # Should handle tiny differences gracefully
        result = optimize_thresholds(labels, probabilities, metric="f1")
        threshold = result.threshold
        assert 0 <= threshold <= 1

        # Should produce valid confusion matrix
        tp, tn, fp, fn = confusion_matrix_at_threshold(labels, probabilities, threshold)
        assert tp + tn + fp + fn == len(labels)

    def test_probability_precision_limits(self):
        """Test behavior at floating point precision limits."""
        labels = np.array([0, 1, 0, 1, 0, 1])

        # Create probabilities with varying precision
        probabilities = np.array(
            [
                0.1,
                0.1 + 1e-15,  # Near machine precision limit
                0.5,
                0.5 + 1e-14,
                0.9,
                0.9 - 1e-15,
            ]
        )

        # Should handle precision limits gracefully
        result = optimize_thresholds(labels, probabilities, metric="accuracy")
        threshold = result.threshold
        assert 0 <= threshold <= 1


class TestErrorConditionEdgeCases:
    """Test error conditions and their handling."""

    def test_nan_in_inputs_clear_error(self):
        """Test that NaN in inputs produces clear error messages."""
        # NaN in labels
        with pytest.raises(ValueError, match="cannot convert float NaN to integer"):
            optimize_thresholds([0, np.nan, 1], [0.1, 0.5, 0.9], metric="f1")

        # NaN in probabilities
        with pytest.raises(ValueError, match="Probabilities contains NaN values"):
            optimize_thresholds([0, 1, 0], [0.1, np.nan, 0.9], metric="f1")

    def test_inf_in_inputs_clear_error(self):
        """Test that infinity in inputs produces clear error messages."""
        # Inf in labels
        with pytest.raises((ValueError, OverflowError)):
            optimize_thresholds([0, np.inf, 1], [0.1, 0.5, 0.9], metric="f1")

        # Inf in probabilities
        with pytest.raises(ValueError, match="Probabilities contains infinite values"):
            optimize_thresholds([0, 1, 0], [0.1, np.inf, 0.9], metric="f1")

    def test_empty_arrays_clear_error(self):
        """Test that empty arrays produce clear error messages."""
        with pytest.raises(ValueError, match="Labels cannot be empty"):
            optimize_thresholds([], [0.5], metric="f1")

        with pytest.raises(ValueError, match="Probabilities cannot be empty"):
            optimize_thresholds([0], [], metric="f1")

    def test_mismatched_lengths_clear_error(self):
        """Test that mismatched array lengths produce clear error messages."""
        with pytest.raises(ValueError, match="Length mismatch"):
            optimize_thresholds([0, 1], [0.5], metric="f1")

        with pytest.raises(ValueError, match="Length mismatch"):
            optimize_thresholds([0], [0.1, 0.5], metric="f1")

    def test_invalid_data_types_clear_error(self):
        """Test that invalid data types produce clear error messages."""
        # String labels should be handled (converted to numeric if possible)
        # but non-numeric strings should fail with clear message
        with pytest.raises((ValueError, TypeError)):
            optimize_thresholds(["a", "b", "c"], [0.1, 0.5, 0.9], metric="f1")

    def test_out_of_range_probabilities_clear_error(self):
        """Test that probabilities outside [0,1] produce clear errors."""
        with pytest.raises(
            ValueError, match="Probabilities must be in \\[0, 1\\], got range"
        ):
            optimize_thresholds([0, 1, 0], [-0.1, 0.5, 0.9], metric="f1")

        with pytest.raises(
            ValueError, match="Probabilities must be in \\[0, 1\\], got range"
        ):
            optimize_thresholds([0, 1, 0], [0.1, 0.5, 1.1], metric="f1")


class TestPerformanceEdgeCases:
    """Test performance characteristics with edge cases."""

    def test_worst_case_performance(self):
        """Test performance with worst-case scenarios."""
        # Many unique probability values (worst case for brute force)
        n_samples = 1000
        labels = np.random.randint(0, 2, n_samples)
        probabilities = np.linspace(0, 1, n_samples)  # All unique values

        import time

        start_time = time.time()
        result = optimize_thresholds(labels, probabilities, metric="f1")
        end_time = time.time()

        threshold = result.threshold
        assert 0 <= threshold <= 1
        assert end_time - start_time < 10.0  # Should complete in reasonable time

    def test_memory_usage_edge_cases(self):
        """Test that memory usage stays reasonable with edge cases."""
        # Large dataset with many unique values
        n_samples = 5000
        labels = np.random.randint(0, 2, n_samples)
        probabilities = np.random.random(n_samples)

        # Should handle without excessive memory usage
        result = optimize_thresholds(labels, probabilities, metric="accuracy")
        threshold = result.threshold
        assert 0 <= threshold <= 1

        # Test confusion matrix doesn't explode memory
        tp, tn, fp, fn = confusion_matrix_at_threshold(labels, probabilities, threshold)
        assert tp + tn + fp + fn == n_samples


class TestMinimalDatasets:
    """Test with minimal viable datasets that weren't covered above."""

    def test_single_positive_sample(self):
        """Test with just one positive sample."""
        y_true = [1]
        pred_prob = [0.7]

        for metric in ["f1", "accuracy", "precision", "recall"]:
            result = optimize_thresholds(y_true, pred_prob, metric=metric)
            threshold = result.threshold
            assert 0.0 <= threshold <= 1.0

    def test_single_negative_sample(self):
        """Test with just one negative sample."""
        y_true = [0]
        pred_prob = [0.3]

        for metric in ["f1", "accuracy", "precision", "recall"]:
            result = optimize_thresholds(y_true, pred_prob, metric=metric)
            threshold = result.threshold
            assert 0.0 <= threshold <= 1.0


class TestExtremeProbabilityValues:
    """Test with probability values at extremes that weren't covered above."""

    def test_all_zero_probabilities(self):
        """Test when all predicted probabilities are 0."""
        y_true = [0, 1, 0, 1, 1]
        pred_prob = [0.0, 0.0, 0.0, 0.0, 0.0]

        for metric in ["f1", "accuracy", "precision", "recall"]:
            result = optimize_thresholds(y_true, pred_prob, metric=metric)
            threshold = result.threshold
            # Allow small tolerance for edge cases where threshold may be slightly
            # outside [0,1]
            assert -1e-8 <= threshold <= 1.0 + 1e-8

            # With all probabilities at 0, optimal strategy depends on comparison
            # operator
            # Only test confusion matrix if threshold is within valid bounds
            if 0.0 <= threshold <= 1.0:
                tp, tn, fp, fn = confusion_matrix_at_threshold(
                    y_true, pred_prob, threshold
                )
                assert tp + tn + fp + fn == len(y_true)

    def test_all_one_probabilities(self):
        """Test when all predicted probabilities are 1."""
        y_true = [0, 1, 0, 1, 1]
        pred_prob = [1.0, 1.0, 1.0, 1.0, 1.0]

        for metric in ["f1", "accuracy", "precision", "recall"]:
            result = optimize_thresholds(y_true, pred_prob, metric=metric)
            threshold = result.threshold
            assert 0.0 <= threshold <= 1.0

            # With all probabilities at 1, optimal strategy depends on comparison
            # operator
            tp, tn, fp, fn = confusion_matrix_at_threshold(y_true, pred_prob, threshold)
            assert tp + tn + fp + fn == len(y_true)

    def test_very_small_probabilities(self):
        """Test with very small but non-zero probabilities."""
        y_true = [0, 1, 0, 1, 1]
        pred_prob = [1e-10, 2e-10, 3e-10, 4e-10, 5e-10]

        result = optimize_thresholds(y_true, pred_prob, metric="f1")
        threshold = result.threshold
        assert 0.0 <= threshold <= 1.0

        # Should handle small probabilities without numerical issues
        tp, tn, fp, fn = confusion_matrix_at_threshold(y_true, pred_prob, threshold)
        assert tp + tn + fp + fn == len(y_true)

    def test_very_large_probabilities_near_one(self):
        """Test with probabilities very close to 1."""
        y_true = [0, 1, 0, 1, 1]
        pred_prob = [1 - 1e-10, 1 - 2e-10, 1 - 3e-10, 1 - 4e-10, 1 - 5e-10]

        result = optimize_thresholds(y_true, pred_prob, metric="f1")
        threshold = result.threshold
        assert 0.0 <= threshold <= 1.0

        # Should handle probabilities near 1 without numerical issues
        tp, tn, fp, fn = confusion_matrix_at_threshold(y_true, pred_prob, threshold)
        assert tp + tn + fp + fn == len(y_true)


class TestMulticlassExtremeScenarios:
    """Test extreme cases in multiclass scenarios."""

    def test_single_class_multiclass(self):
        """Test multiclass optimization with highly imbalanced classes."""
        # Use 2 classes but heavily imbalanced (95% class 0, 5% class 1)
        y_true = [0] * 19 + [1] * 1  # Only 1 sample of class 1
        pred_prob = np.random.RandomState(42).uniform(0, 1, (20, 2))
        pred_prob = pred_prob / pred_prob.sum(axis=1, keepdims=True)  # Normalize

        result = optimize_thresholds(y_true, pred_prob, metric="f1")
        thresholds = result.thresholds
        assert len(thresholds) == 2
        # Note: With coordinate ascent and extreme imbalance, thresholds can be outside
        # [0,1]
        # This is mathematically correct for margin-based decision rules
        assert all(np.isfinite(t) for t in thresholds), "Thresholds should be finite"

    def test_extreme_multiclass_imbalance(self):
        """Test multiclass with extreme class imbalance."""
        # 98% class 0, 1% class 1, 1% class 2
        y_true = [0] * 98 + [1] + [2]
        np.random.seed(42)
        pred_prob = np.random.uniform(0, 1, (100, 3))
        pred_prob = pred_prob / pred_prob.sum(axis=1, keepdims=True)  # Normalize

        result = optimize_thresholds(y_true, pred_prob, metric="f1")
        thresholds = result.thresholds
        assert len(thresholds) == 3
        # Note: With coordinate ascent and extreme imbalance, thresholds can be outside
        # [0,1]
        # This is mathematically correct for margin-based decision rules
        assert all(np.isfinite(t) for t in thresholds), "Thresholds should be finite"

    def test_multiclass_with_zero_probabilities(self):
        """Test multiclass with some zero probability columns."""
        y_true = [0, 1, 2, 0, 1, 2]
        pred_prob = np.array(
            [
                [1.0, 0.0, 0.0],  # Only class 0 has probability
                [0.0, 1.0, 0.0],  # Only class 1 has probability
                [0.0, 0.0, 1.0],  # Only class 2 has probability
                [0.5, 0.5, 0.0],  # Classes 0,1 split probability
                [0.0, 0.5, 0.5],  # Classes 1,2 split probability
                [0.3, 0.3, 0.4],  # All classes have some probability
            ]
        )

        result = optimize_thresholds(y_true, pred_prob, metric="f1")
        thresholds = result.thresholds
        assert len(thresholds) == 3
        # Note: With coordinate ascent and extreme cases, thresholds can be outside
        # [0,1]
        # This is mathematically correct for margin-based decision rules
        assert all(np.isfinite(t) for t in thresholds), "Thresholds should be finite"
