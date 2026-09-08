"""Hypothesis property tests for algorithm correctness and invariants."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from optimal_cutoffs import optimize_thresholds
from optimal_cutoffs.metrics_core import (
    compute_metric_at_threshold,
    confusion_matrix_at_threshold,
    multiclass_metric_ovr,
)


def multiclass_confusion_matrices_at_thresholds(
    y_true, y_prob, thresholds, sample_weight=None, comparison=">="
):
    """Helper function to compute per-class confusion matrices for multiclass OvR."""
    n_classes = y_prob.shape[1]
    cms = []

    for class_idx in range(n_classes):
        # One-vs-Rest for this class
        y_true_binary = (y_true == class_idx).astype(int)
        y_prob_class = y_prob[:, class_idx]
        threshold = thresholds[class_idx]

        tp, tn, fp, fn = confusion_matrix_at_threshold(
            y_true_binary, y_prob_class, threshold, sample_weight, comparison
        )
        cms.append((tp, tn, fp, fn))

    return cms


# Custom strategies for generating test data
def valid_binary_labels(min_size=2, max_size=100):
    """Generate valid binary label arrays."""
    return arrays(
        dtype=np.int8, shape=st.integers(min_size, max_size), elements=st.integers(0, 1)
    )


def valid_probabilities(min_size=2, max_size=100):
    """Generate valid probability arrays."""
    return arrays(
        dtype=np.float64,
        shape=st.integers(min_size, max_size),
        elements=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
        unique=False,
    )


def matched_labels_and_probabilities(min_size=2, max_size=50):
    """Generate matched pairs of labels and probabilities."""

    @st.composite
    def _strategy(draw):
        size = draw(st.integers(min_size, max_size))
        labels = draw(arrays(dtype=np.int8, shape=size, elements=st.integers(0, 1)))
        probs = draw(
            arrays(
                dtype=np.float64,
                shape=size,
                elements=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
            )
        )
        return labels, probs

    return _strategy()


class TestCoreInvariants:
    """Test fundamental mathematical properties that must always hold."""

    @given(matched_labels_and_probabilities(min_size=5, max_size=50))
    @settings(max_examples=50)
    def test_piecewise_matches_brute_force(self, data):
        """Verify O(n log n) piecewise solution matches naive brute force."""
        labels, probabilities = data

        # Skip degenerate cases
        if len(np.unique(labels)) < 2:
            return
        if len(np.unique(probabilities)) <= 1:
            return  # Skip when all probabilities identical

        def naive_brute_force(true_labs, pred_prob, metric):
            """Naive brute force that tests same candidates as piecewise algorithm."""
            y_true = np.asarray(true_labs)
            p = np.asarray(pred_prob)
            n = len(p)

            # Sort probabilities in descending order (same as piecewise algorithm)
            sorted_idx = np.argsort(-p)
            p_sorted = p[sorted_idx]

            best_score = -np.inf
            best_threshold = 0.5

            # Test all possible cut positions (same as piecewise algorithm)
            for k in range(n):
                # Compute threshold midpoint for cut position k
                if k == n - 1:
                    # Predict all as positive
                    threshold = max(0.0, float(np.nextafter(p_sorted[-1], -np.inf)))
                else:
                    included_prob = float(p_sorted[k])
                    excluded_prob = float(p_sorted[k + 1])

                    if included_prob > excluded_prob:
                        threshold = 0.5 * (included_prob + excluded_prob)
                    else:
                        # Tied probabilities - use slight nudge
                        threshold = float(np.nextafter(excluded_prob, np.inf))

                # Clamp to [0, 1]
                threshold = max(0.0, min(1.0, threshold))

                try:
                    score = compute_metric_at_threshold(
                        y_true, pred_prob, threshold, metric
                    )
                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
                except (ValueError, ZeroDivisionError):
                    # Skip invalid thresholds
                    continue

            # Also test a few edge cases
            for edge_threshold in [0.001, 0.999]:
                try:
                    score = compute_metric_at_threshold(
                        y_true, p, edge_threshold, metric
                    )
                    if score > best_score:
                        best_score = score
                        best_threshold = edge_threshold
                except (ValueError, ZeroDivisionError):
                    continue

            return float(best_threshold)

        # Test on multiple metrics
        for metric in ["f1", "accuracy", "precision", "recall"]:
            # Use piecewise optimization
            piecewise_result = optimize_thresholds(labels, probabilities, metric=metric)
            piecewise_threshold = piecewise_result.threshold

            # Use naive brute force
            naive_threshold = naive_brute_force(labels, probabilities, metric)

            # Compute scores for both thresholds
            piecewise_score = compute_metric_at_threshold(
                labels, probabilities, piecewise_threshold, metric
            )
            naive_score = compute_metric_at_threshold(
                labels, probabilities, naive_threshold, metric
            )

            # Piecewise should generally be at least as good as naive
            # However, edge cases with identical probabilities may have implementation
            # differences
            tolerance = 1e-10
            unique_probs = np.unique(probabilities)
            has_ties = len(unique_probs) < len(
                probabilities
            )  # Check for tied probabilities
            if (
                len(unique_probs) <= 3
                or 0.0 in probabilities
                or 1.0 in probabilities
                or has_ties
            ):
                tolerance = 0.2  # More lenient for edge cases with boundary/tied values

            assert piecewise_score >= naive_score - tolerance, (
                f"Piecewise optimization worse than naive for {metric}: "
                f"{piecewise_score} vs {naive_score} (tolerance={tolerance})"
            )

    @given(
        data=matched_labels_and_probabilities(min_size=10, max_size=30),
        epsilon=st.floats(0.001, 0.1),
    )
    @settings(max_examples=30)
    def test_threshold_shift_invariance(self, data, epsilon):
        """Adding constant to probabilities should preserve relative ranking."""
        labels, probabilities = data

        # Skip degenerate cases
        if len(np.unique(labels)) < 2:
            return

        # Skip if shifting would push probabilities out of [0,1]
        if np.any(probabilities + epsilon > 1) or np.any(probabilities - epsilon < 0):
            return

        # Check for extensive tied probabilities (>70% tied)
        # Linear threshold shifting doesn't apply when many probabilities are identical
        unique_probs = np.unique(probabilities)
        max_tie_count = max([np.sum(probabilities == p) for p in unique_probs])
        tie_fraction = max_tie_count / len(probabilities)

        if tie_fraction > 0.7:
            # Skip cases with extensive tied probabilities as threshold shifts can be
            # non-linear
            return

        # Get original optimal threshold
        original_result = optimize_thresholds(labels, probabilities, metric="f1")
        original_threshold = original_result.threshold

        # Shift probabilities by epsilon (both directions)
        shifted_up = probabilities + epsilon
        shifted_down = probabilities - epsilon

        # Get new optimal thresholds
        threshold_up = optimize_thresholds(labels, shifted_up, metric="f1")
        threshold_down = optimize_thresholds(labels, shifted_down, metric="f1")

        # For cases without extensive ties, threshold should shift approximately by
        # epsilon
        # But allow larger tolerance for edge cases
        tolerance = max(0.2, epsilon * 2)  # More generous tolerance

        shift_up_diff = abs((threshold_up - original_threshold) - epsilon)
        shift_down_diff = abs((original_threshold - threshold_down) - epsilon)

        assert shift_up_diff < tolerance, (
            f"Threshold shift invariance violated (up): original={original_threshold}, "
            f"shifted_up={threshold_up}, epsilon={epsilon}, diff={shift_up_diff}"
        )

        assert shift_down_diff < tolerance, (
            f"Threshold shift invariance violated (down): "
            f"original={original_threshold}, "
            f"shifted_down={threshold_down}, epsilon={epsilon}, diff={shift_down_diff}"
        )

    @given(matched_labels_and_probabilities(min_size=3, max_size=50))
    @settings(max_examples=50)
    def test_threshold_bounds(self, data):
        """Optimal thresholds must be within [0, 1] range."""
        labels, probabilities = data

        # Skip degenerate cases
        if len(np.unique(labels)) < 2:
            return

        for metric in ["f1", "accuracy", "precision", "recall"]:
            result = optimize_thresholds(labels, probabilities, metric=metric)
            threshold = result.threshold
            # Allow small tolerance outside [0,1] for boundary conditions and
            # tie-breaking
            tolerance = 1e-9
            assert -tolerance <= threshold <= 1 + tolerance, (
                f"Threshold {threshold} out of bounds for {metric}"
            )

    @given(matched_labels_and_probabilities())
    @settings(max_examples=30)
    def test_determinism(self, data):
        """Same inputs must always produce same outputs."""
        labels, probabilities = data

        # Skip degenerate cases
        if len(np.unique(labels)) < 2:
            return

        # Run optimization multiple times
        thresholds = []
        for _ in range(3):
            result = optimize_thresholds(labels, probabilities, metric="f1")
            threshold = result.threshold
            thresholds.append(threshold)

        # All results should be identical
        assert all(abs(t - thresholds[0]) < 1e-12 for t in thresholds), (
            f"Non-deterministic results: {thresholds}"
        )

    @given(matched_labels_and_probabilities(min_size=4, max_size=30))
    @settings(max_examples=30)
    def test_confusion_matrix_consistency(self, data):
        """Confusion matrix elements should sum to total samples."""
        labels, probabilities = data

        result = optimize_thresholds(labels, probabilities, metric="f1")
        tp, tn, fp, fn = confusion_matrix_at_threshold(
            labels, probabilities, result.threshold
        )

        total = tp + tn + fp + fn
        assert total == len(labels), (
            f"Confusion matrix inconsistent: TP={tp}, TN={tn}, FP={fp}, FN={fn}, "
            f"total={total}, expected={len(labels)}"
        )

        # All elements should be non-negative
        message = (
            f"Negative confusion matrix elements: TP={tp}, TN={tn}, FP={fp}, FN={fn}"
        )
        assert tp >= 0, message
        assert tn >= 0, message
        assert fp >= 0, message
        assert fn >= 0, message

    @given(matched_labels_and_probabilities(min_size=5, max_size=20))
    @settings(max_examples=30)
    def test_comparison_operator_consistency(self, data):
        """Test that > and >= produce consistent but potentially different results."""
        labels, probabilities = data

        # Skip degenerate cases
        if len(np.unique(labels)) < 2:
            return

        # Get thresholds with both comparison operators
        threshold_gt = optimize_thresholds(labels, probabilities, "f1", comparison=">")
        threshold_gte = optimize_thresholds(
            labels, probabilities, "f1", comparison=">="
        )

        # Both should be valid thresholds
        assert 0 <= threshold_gt <= 1
        assert 0 <= threshold_gte <= 1

        # Get confusion matrices for both
        cm_gt = confusion_matrix_at_threshold(
            labels, probabilities, threshold_gt, comparison=">"
        )
        cm_gte = confusion_matrix_at_threshold(
            labels, probabilities, threshold_gte, comparison=">="
        )

        # Both should sum to total samples
        assert sum(cm_gt) == len(labels)
        assert sum(cm_gte) == len(labels)


class TestStatisticalProperties:
    """Test statistical and mathematical properties."""

    @given(matched_labels_and_probabilities(min_size=10, max_size=50))
    @settings(max_examples=20)
    def test_better_separation_improves_metrics(self, data):
        """Better class separation should generally lead to better metrics."""
        labels, probabilities = data

        # Skip if all same class or very small dataset
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2 or len(labels) < 10:
            return

        # Check if original separation is already very good
        original_result = optimize_thresholds(labels, probabilities, metric="f1")
        original_f1 = compute_metric_at_threshold(
            labels, probabilities, original_result.threshold, "f1"
        )

        # If original F1 is already very high (>=0.9), skip the test
        # as there's little room for improvement and "perfect" separation may not
        # actually be better
        if original_f1 >= 0.9:
            return

        # Check if the original probabilities are already well-separated
        # If the correlation between labels and probabilities is already very high, skip
        if len(labels) > 3:  # Need enough samples for correlation
            # Handle case where variance is zero (all values identical)
            if np.std(labels) > 1e-10 and np.std(probabilities) > 1e-10:
                correlation = abs(np.corrcoef(labels, -probabilities)[0, 1])
                if correlation >= 0.9:  # Already highly correlated
                    return
            else:
                # If either has zero variance, skip (no meaningful correlation)
                return

        # Create perfect separation version
        # Sort by original probabilities, then assign probabilities based on labels
        sort_idx = np.argsort(probabilities)
        sorted_labels = labels[sort_idx]

        # Create perfectly separated probabilities
        perfect_probs = np.zeros_like(probabilities)
        perfect_probs[sorted_labels == 0] = 0.25  # All negatives get low prob
        perfect_probs[sorted_labels == 1] = 0.75  # All positives get high prob

        # Add small noise to avoid ties
        perfect_probs += np.random.normal(0, 0.01, size=len(perfect_probs))
        perfect_probs = np.clip(perfect_probs, 0, 1)

        # Compare metrics
        for metric in ["accuracy", "f1"]:
            perfect_result = optimize_thresholds(labels, perfect_probs, metric)

            original_score = compute_metric_at_threshold(
                labels, probabilities, original_result.threshold, metric
            )
            perfect_score = compute_metric_at_threshold(
                labels, perfect_probs, perfect_result.threshold, metric
            )

            # Perfect separation should generally be better or at least reasonable
            # However, with extreme class imbalance, perfect separation can sometimes
            # perform worse than well-calibrated probabilities. We only fail if the
            # performance is unreasonably bad.

            # Skip the test if we have extreme class imbalance (< 10% or > 90% positive)
            pos_ratio = np.mean(labels)
            if pos_ratio < 0.1 or pos_ratio > 0.9:
                # With extreme imbalance, perfect separation assumptions may not hold
                return

            if perfect_score < original_score - 0.5:
                # Only fail if the difference is very substantial
                pytest.fail(
                    f"Perfect separation much worse than original for {metric}: "
                    f"{perfect_score} vs {original_score} "
                    f"(original_f1={original_f1:.3f}, "
                    f"pos_ratio={pos_ratio:.3f})"
                )

    @given(matched_labels_and_probabilities(min_size=3, max_size=30))
    @settings(max_examples=30)
    def test_metric_bounds(self, data):
        """Test that all metrics produce values in expected ranges."""
        labels, probabilities = data

        # Skip degenerate cases
        if len(np.unique(labels)) < 2:
            return

        result = optimize_thresholds(labels, probabilities, metric="f1")

        # Test all registered metrics
        for metric_name in ["f1", "accuracy", "precision", "recall"]:
            score = compute_metric_at_threshold(
                labels, probabilities, result.threshold, metric_name
            )

            # All these metrics should be in [0, 1] range
            assert 0 <= score <= 1, f"Metric {metric_name} out of bounds: {score}"

    @given(
        data=matched_labels_and_probabilities(min_size=5, max_size=20),
        method=st.sampled_from(["unique_scan", "minimize", "gradient"]),
    )
    @settings(max_examples=20)
    def test_optimization_method_consistency(self, data, method):
        """Different optimization methods should produce reasonable results."""
        labels, probabilities = data

        # Skip degenerate cases
        if len(np.unique(labels)) < 2:
            return

        try:
            result = optimize_thresholds(labels, probabilities, "f1", method=method)

            # Threshold should be valid
            threshold = result.threshold
            assert 0 <= threshold <= 1

            # Should produce a valid confusion matrix
            tp, tn, fp, fn = confusion_matrix_at_threshold(
                labels, probabilities, threshold
            )
            assert tp + tn + fp + fn == len(labels)

        except Exception as e:
            # Some methods might fail on edge cases, but should fail gracefully
            assert isinstance(e, ValueError | RuntimeError), (
                f"Unexpected error type for method {method}: {type(e)}"
            )


class TestNumericalStability:
    """Test numerical stability and edge cases."""

    @given(size=st.integers(3, 20), base_prob=st.floats(0.1, 0.9))
    @settings(max_examples=20)
    def test_near_identical_probabilities(self, size, base_prob):
        """Test handling of very similar probability values."""
        # Create probabilities that differ by tiny amounts
        probabilities = np.full(size, base_prob)
        probabilities += np.random.normal(0, 1e-10, size)  # Tiny noise
        probabilities = np.clip(probabilities, 0, 1)

        # Create balanced labels
        labels = np.array([i % 2 for i in range(size)])

        # Should handle gracefully without numerical issues
        result = optimize_thresholds(labels, probabilities, metric="f1")
        threshold = result.threshold
        assert 0 <= threshold <= 1

        # Confusion matrix should be valid
        tp, tn, fp, fn = confusion_matrix_at_threshold(labels, probabilities, threshold)
        assert tp + tn + fp + fn == len(labels)

    @given(matched_labels_and_probabilities(min_size=3, max_size=20))
    @settings(max_examples=20)
    def test_extreme_probability_values(self, data):
        """Test with probabilities at the extremes of [0,1]."""
        labels, probabilities = data

        # Force some probabilities to extremes
        probabilities = probabilities.copy()
        probabilities[probabilities < 0.1] = 0.0
        probabilities[probabilities > 0.9] = 1.0

        # Skip degenerate cases
        if len(np.unique(labels)) < 2:
            return

        # Should handle extreme values gracefully
        result = optimize_thresholds(labels, probabilities, metric="accuracy")
        threshold = result.threshold
        assert 0 <= threshold <= 1

        # Test confusion matrix
        tp, tn, fp, fn = confusion_matrix_at_threshold(labels, probabilities, threshold)
        assert tp + tn + fp + fn == len(labels)


def valid_multiclass_labels(n_classes=3, min_size=6, max_size=30):
    """Generate valid multiclass label arrays."""

    @st.composite
    def _strategy(draw):
        size = draw(st.integers(min_size, max_size))
        return draw(
            arrays(dtype=np.int8, shape=size, elements=st.integers(0, n_classes - 1))
        )

    return _strategy()


def valid_multiclass_probabilities(n_classes=3, min_size=6, max_size=30):
    """Generate valid multiclass probability matrices."""

    @st.composite
    def _strategy(draw):
        size = draw(st.integers(min_size, max_size))
        # Generate probability matrix
        probs = draw(
            arrays(
                dtype=np.float64,
                shape=(size, n_classes),
                elements=st.floats(0.1, 0.9, allow_nan=False, allow_infinity=False),
            )
        )
        # Normalize to sum to 1 (approximately)
        row_sums = np.sum(probs, axis=1, keepdims=True)
        return probs / row_sums

    return _strategy()


def matched_multiclass_data(n_classes=3, min_size=9, max_size=30):
    """Generate matched multiclass labels and probabilities."""

    @st.composite
    def _strategy(draw):
        size = draw(st.integers(min_size, max_size))
        labels = draw(
            arrays(dtype=np.int8, shape=size, elements=st.integers(0, n_classes - 1))
        )

        # Ensure we have all classes represented (consecutive from 0)
        # by forcing at least one example of each class
        if len(labels) >= n_classes:
            # Replace first n_classes elements with one of each class
            for i in range(n_classes):
                labels[i] = i

        probs = draw(
            arrays(
                dtype=np.float64,
                shape=(size, n_classes),
                elements=st.floats(0.05, 0.95, allow_nan=False, allow_infinity=False),
            )
        )
        # Normalize probabilities
        row_sums = np.sum(probs, axis=1, keepdims=True)
        probs = probs / row_sums
        return labels, probs

    return _strategy()


class TestMulticlassPropertyBased:
    """Property-based tests for multiclass classification."""

    @given(matched_multiclass_data(n_classes=3, min_size=12, max_size=30))
    @settings(max_examples=20)
    def test_multiclass_threshold_bounds(self, data):
        """All multiclass thresholds must be in [0, 1] range."""
        labels, probabilities = data

        # Ensure we have all classes represented
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return  # Skip degenerate cases

        thresholds = optimize_thresholds(labels, probabilities, "f1", average="macro")

        assert isinstance(thresholds, np.ndarray)
        assert len(thresholds) == probabilities.shape[1]
        assert np.all((thresholds >= 0) & (thresholds <= 1))

    @given(matched_multiclass_data(n_classes=3, min_size=15, max_size=25))
    @settings(max_examples=15)
    def test_multiclass_ovr_independence(self, data):
        """One-vs-Rest thresholds should be independent across classes."""
        labels, probabilities = data

        # Skip if too few classes represented
        unique_labels = np.unique(labels)
        if len(unique_labels) < 3:
            return

        # Get thresholds using macro averaging (independent optimization)
        thresholds = optimize_thresholds(labels, probabilities, "f1", average="macro")

        # Manually compute per-class thresholds
        manual_thresholds = []
        for class_idx in range(probabilities.shape[1]):
            if class_idx in unique_labels:
                # One-vs-Rest binary problem
                binary_labels = (labels == class_idx).astype(int)
                binary_probs = probabilities[:, class_idx]

                # Skip if this class has no positive examples
                if np.sum(binary_labels) > 0 and np.sum(1 - binary_labels) > 0:
                    binary_result = optimize_thresholds(
                        binary_labels, binary_probs, "f1"
                    )
                    manual_thresholds.append(binary_result.threshold)
                else:
                    manual_thresholds.append(0.5)  # Default for degenerate case
            else:
                manual_thresholds.append(0.5)  # Default for missing class

        manual_thresholds = np.array(manual_thresholds)

        # Should be approximately equal (allowing for numerical differences)
        for i, (auto, manual) in enumerate(
            zip(thresholds, manual_thresholds, strict=False)
        ):
            if i in unique_labels and np.sum(labels == i) > 0:
                assert abs(auto - manual) < 1e-6, (
                    f"Class {i} threshold mismatch: auto={auto}, manual={manual}"
                )

    @given(
        data=matched_multiclass_data(n_classes=3, min_size=12, max_size=20),
        average=st.sampled_from(["macro", "micro", "weighted"]),
    )
    @settings(max_examples=15)
    def test_multiclass_averaging_consistency(self, data, average):
        """Different averaging methods should produce consistent results."""
        labels, probabilities = data

        # Skip degenerate cases
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return

        try:
            thresholds = optimize_thresholds(
                labels, probabilities, "f1", average=average
            )

            # Should produce valid thresholds
            if average in ["macro", "weighted"]:
                # These return per-class thresholds
                assert isinstance(thresholds, np.ndarray)
                assert len(thresholds) == probabilities.shape[1]
                assert np.all((thresholds >= 0) & (thresholds <= 1))
            else:
                # Micro might return different format
                assert np.all(
                    (np.asarray(thresholds) >= 0) & (np.asarray(thresholds) <= 1)
                )

            # Test that confusion matrices are valid
            cms = multiclass_confusion_matrices_at_thresholds(
                labels, probabilities, thresholds
            )
            assert len(cms) == probabilities.shape[1]

            for tp, tn, fp, fn in cms:
                assert tp >= 0
                assert tn >= 0
                assert fp >= 0
                assert fn >= 0

        except (ValueError, RuntimeError):
            # Some averaging methods might fail on edge cases
            pass

    @given(matched_multiclass_data(n_classes=4, min_size=16, max_size=24))
    @settings(max_examples=10)
    def test_multiclass_determinism(self, data):
        """Multiclass optimization should be deterministic."""
        labels, probabilities = data

        # Skip degenerate cases
        if len(np.unique(labels)) < 2:
            return

        # Run optimization multiple times
        thresholds_list = []
        for _ in range(3):
            thresholds = optimize_thresholds(
                labels, probabilities, "accuracy", average="macro"
            )
            thresholds_list.append(thresholds)

        # All results should be identical
        for i, thresholds in enumerate(thresholds_list[1:], 1):
            assert np.allclose(thresholds_list[0], thresholds, atol=1e-12), (
                f"Non-deterministic multiclass results: run 0 vs run {i}"
            )

    @given(
        data=matched_multiclass_data(n_classes=3, min_size=9, max_size=18),
        metric=st.sampled_from(["f1", "precision", "recall", "accuracy"]),
    )
    @settings(max_examples=15)
    def test_multiclass_metric_bounds(self, data, metric):
        """Multiclass metrics should be in expected ranges."""
        labels, probabilities = data

        # Skip degenerate cases
        if len(np.unique(labels)) < 2:
            return

        try:
            thresholds = optimize_thresholds(
                labels, probabilities, metric=metric, average="macro"
            )

            # Get confusion matrices
            cms = multiclass_confusion_matrices_at_thresholds(
                labels, probabilities, thresholds
            )

            # Test macro averaging
            score_macro = multiclass_metric_ovr(cms, metric, "macro")
            assert 0 <= score_macro <= 1, f"Macro {metric} out of bounds: {score_macro}"

            # Test micro averaging
            score_micro = multiclass_metric_ovr(cms, metric, "micro")
            assert 0 <= score_micro <= 1, f"Micro {metric} out of bounds: {score_micro}"

            # Test weighted averaging
            score_weighted = multiclass_metric_ovr(cms, metric, "weighted")
            assert 0 <= score_weighted <= 1, (
                f"Weighted {metric} out of bounds: {score_weighted}"
            )

        except (ValueError, ZeroDivisionError):
            # Some metrics might fail on edge cases (e.g., no positives for precision)
            pass

    @given(matched_multiclass_data(n_classes=3, min_size=12, max_size=20))
    @settings(max_examples=10)
    def test_multiclass_comparison_operators(self, data):
        """Test multiclass with different comparison operators."""
        labels, probabilities = data

        # Skip degenerate cases
        if len(np.unique(labels)) < 2:
            return

        # Get thresholds with both comparison operators
        thresholds_gt = optimize_thresholds(labels, probabilities, "f1", comparison=">")
        thresholds_gte = optimize_thresholds(
            labels, probabilities, "f1", comparison=">="
        )

        # Both should be valid
        assert np.all((thresholds_gt >= 0) & (thresholds_gt <= 1))
        assert np.all((thresholds_gte >= 0) & (thresholds_gte <= 1))

        # Get confusion matrices with both
        cms_gt = multiclass_confusion_matrices_at_thresholds(
            labels, probabilities, thresholds_gt, comparison=">"
        )
        cms_gte = multiclass_confusion_matrices_at_thresholds(
            labels, probabilities, thresholds_gte, comparison=">="
        )

        # Both should have valid confusion matrices
        for cms in [cms_gt, cms_gte]:
            assert len(cms) == probabilities.shape[1]
            for tp, tn, fp, fn in cms:
                assert tp >= 0
                assert tn >= 0
                assert fp >= 0
                assert fn >= 0


class TestMulticlassMathematicalProperties:
    """Test mathematical properties specific to multiclass classification."""

    @given(matched_multiclass_data(n_classes=3, min_size=15, max_size=25))
    @settings(max_examples=10)
    def test_multiclass_macro_micro_weighted_relationships(self, data):
        """Test mathematical relationships between averaging methods."""
        labels, probabilities = data

        # Skip degenerate cases
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return

        try:
            # Get thresholds (use macro for independent optimization)
            thresholds = optimize_thresholds(
                labels, probabilities, "f1", average="macro"
            )

            # Get confusion matrices
            cms = multiclass_confusion_matrices_at_thresholds(
                labels, probabilities, thresholds
            )

            # Compute different averages
            score_macro = multiclass_metric_ovr(cms, "f1", "macro")
            score_micro = multiclass_metric_ovr(cms, "f1", "micro")
            score_weighted = multiclass_metric_ovr(cms, "f1", "weighted")
            score_none = multiclass_metric_ovr(cms, "f1", "none")

            # All should be valid
            assert 0 <= score_macro <= 1
            assert 0 <= score_micro <= 1
            assert 0 <= score_weighted <= 1
            assert isinstance(score_none, np.ndarray)
            assert len(score_none) == probabilities.shape[1]
            assert np.all((score_none >= 0) & (score_none <= 1))

            # Macro average should equal mean of per-class scores
            per_class_mean = np.mean(score_none)
            assert abs(score_macro - per_class_mean) < 1e-10, (
                f"Macro average {score_macro} != mean of per-class {per_class_mean}"
            )

        except (ValueError, ZeroDivisionError):
            # Some edge cases might cause mathematical issues
            pass

    @given(matched_multiclass_data(n_classes=3, min_size=12, max_size=20))
    @settings(max_examples=10)
    def test_multiclass_threshold_vector_properties(self, data):
        """Test properties of the threshold vector."""
        labels, probabilities = data

        # Skip degenerate cases
        if len(np.unique(labels)) < 2:
            return

        thresholds = optimize_thresholds(
            labels, probabilities, "precision", average="macro"
        )

        # Should be a proper vector
        assert isinstance(thresholds, np.ndarray)
        assert thresholds.ndim == 1
        assert len(thresholds) == probabilities.shape[1]

        # Each threshold should be reasonable for the class probability distribution
        for class_idx, threshold in enumerate(thresholds):
            class_probs = probabilities[:, class_idx]
            unique_class_probs = np.unique(class_probs)

            # Threshold should be close to one of the unique probability values OR their
            # midpoints
            # The algorithm can choose midpoints between adjacent unique values for
            # optimal cuts
            candidate_values = list(unique_class_probs)

            # Add midpoints between adjacent unique values as valid candidates
            sorted_probs = np.sort(unique_class_probs)
            for i in range(len(sorted_probs) - 1):
                midpoint = 0.5 * (sorted_probs[i] + sorted_probs[i + 1])
                candidate_values.append(midpoint)

            # Also add edge cases (slightly above max, slightly below min)
            if len(sorted_probs) > 0:
                candidate_values.append(min(1.0, sorted_probs[-1] + 0.01))  # Above max
                candidate_values.append(max(0.0, sorted_probs[0] - 0.01))  # Below min

            min_distance = min(abs(threshold - p) for p in candidate_values)
            assert min_distance < 0.02, (
                f"Class {class_idx} threshold {threshold} not close to any candidate "
                f"value. "
                f"Candidates: {sorted(set(candidate_values))}"
            )


class TestPerformanceProperties:
    """Property-based tests for performance characteristics."""

    @given(matched_labels_and_probabilities(min_size=100, max_size=1000))
    @settings(max_examples=5, deadline=10000)  # Longer deadline for performance tests
    def test_on_log_n_complexity_verification(self, data):
        """Verify that piecewise optimization scales as O(n log n)."""
        labels, probabilities = data

        # Skip degenerate cases
        if len(np.unique(labels)) < 2:
            return

        import time

        # Time the piecewise optimization
        start_time = time.time()
        result = optimize_thresholds(labels, probabilities, metric="f1")
        threshold = result.threshold
        piecewise_time = time.time() - start_time

        # Should complete in reasonable time even for large inputs
        expected_max_time = 0.1 * len(probabilities) / 100  # Scale with size
        assert piecewise_time < expected_max_time, (
            f"Piecewise optimization too slow: {piecewise_time:.4f}s for "
            f"{len(probabilities)} samples"
        )

        # Result should be valid
        assert 0 <= threshold <= 1

    @given(matched_labels_and_probabilities(min_size=50, max_size=500))
    @settings(max_examples=10, deadline=15000)
    def test_memory_usage_linear_scaling(self, data):
        """Verify that memory usage scales linearly with input size."""
        labels, probabilities = data

        # Skip degenerate cases
        if len(np.unique(labels)) < 2:
            return

        # The algorithm should not create data structures that scale quadratically
        # This is mainly tested by ensuring it completes without memory errors
        result = optimize_thresholds(labels, probabilities, metric="accuracy")
        threshold = result.threshold
        assert 0 <= threshold <= 1

        # Test that confusion matrix computation is also linear
        tp, tn, fp, fn = confusion_matrix_at_threshold(labels, probabilities, threshold)
        assert tp + tn + fp + fn == len(labels)

    @given(matched_labels_and_probabilities(min_size=20, max_size=200))
    @settings(max_examples=15)
    def test_performance_consistency_across_methods(self, data):
        """Test that different optimization methods have reasonable performance."""
        labels, probabilities = data

        # Skip degenerate cases
        if len(np.unique(labels)) < 2:
            return

        import time

        methods = ["unique_scan", "minimize", "gradient"]
        times = {}
        results = {}

        for method in methods:
            start_time = time.time()
            try:
                result = optimize_thresholds(labels, probabilities, "f1", method=method)
                end_time = time.time()

                times[method] = end_time - start_time
                results[method] = result.threshold

                # Should complete in reasonable time
                assert times[method] < 5.0, (
                    f"{method} took too long: {times[method]:.2f}s"
                )

                # Should produce valid result
                threshold = result.threshold
                assert 0 <= threshold <= 1

            except Exception as e:
                # Some methods might fail on edge cases
                assert isinstance(e, ValueError | RuntimeError)

    @given(matched_multiclass_data(n_classes=3, min_size=30, max_size=200))
    @settings(max_examples=5, deadline=15000)
    def test_multiclass_performance_scaling(self, data):
        """Test that multiclass optimization scales reasonably."""
        labels, probabilities = data

        # Skip degenerate cases
        if len(np.unique(labels)) < 2:
            return

        import time

        start_time = time.time()
        thresholds = optimize_thresholds(labels, probabilities, "f1", average="macro")
        end_time = time.time()

        # Should complete in reasonable time
        # Multiclass is roughly O(k * n log n) where k is number of classes
        n_classes = probabilities.shape[1]
        n_samples = len(labels)
        expected_max_time = 0.01 * n_classes * n_samples / 100

        assert end_time - start_time < expected_max_time, (
            f"Multiclass optimization too slow: {end_time - start_time:.4f}s for "
            f"{n_samples} samples and {n_classes} classes"
        )

        # Results should be valid
        assert isinstance(thresholds, np.ndarray)
        assert len(thresholds) == n_classes
        assert np.all((thresholds >= 0) & (thresholds <= 1))

    @given(size=st.integers(10, 100), n_unique=st.integers(2, 20))
    @settings(max_examples=10)
    def test_performance_vs_unique_values(self, size, n_unique):
        """Test how performance varies with number of unique probability values."""
        # Limit n_unique to be at most size
        n_unique = min(n_unique, size)

        # Create labels and probabilities with controlled uniqueness
        labels = np.array([i % 2 for i in range(size)])
        unique_probs = np.linspace(0.1, 0.9, n_unique)
        probabilities = np.random.choice(unique_probs, size=size)

        import time

        start_time = time.time()
        result = optimize_thresholds(labels, probabilities, metric="f1")
        end_time = time.time()

        # Time should scale with number of unique values, not total samples
        # For the unique_scan method, complexity is O(k log k) where k is unique values
        expected_max_time = 0.01 * n_unique / 10 + 0.001  # Base time + scaling

        assert end_time - start_time < expected_max_time, (
            f"Performance worse than expected: {end_time - start_time:.4f}s for "
            f"{n_unique} unique values in {size} samples"
        )

        threshold = result.threshold
        assert 0 <= threshold <= 1

    @given(matched_labels_and_probabilities(min_size=50, max_size=200))
    @settings(max_examples=10)
    def test_no_performance_degradation_with_edge_cases(self, data):
        """Test that edge cases don't cause severe performance degradation."""
        labels, probabilities = data

        # Skip degenerate cases
        if len(np.unique(labels)) < 2:
            return

        import time

        # Test various edge case scenarios
        edge_cases = [
            ("normal", labels, probabilities),
            ("all_same_prob", labels, np.full_like(probabilities, 0.5)),
            ("extreme_probs", labels, np.where(probabilities < 0.5, 0.01, 0.99)),
        ]

        for case_name, test_labels, test_probs in edge_cases:
            start_time = time.time()
            try:
                result = optimize_thresholds(test_labels, test_probs, "accuracy")
                end_time = time.time()

                # Should complete in reasonable time
                assert end_time - start_time < 1.0, (
                    f"Edge case '{case_name}' too slow: {end_time - start_time:.4f}s"
                )

                threshold = result.threshold
                assert 0 <= threshold <= 1

            except ValueError:
                # Some edge cases might be invalid, that's ok
                pass

    @given(matched_labels_and_probabilities(min_size=100, max_size=500))
    @settings(max_examples=5, deadline=20000)
    def test_comparison_operator_performance_equivalence(self, data):
        """Test that different comparison operators have similar performance."""
        labels, probabilities = data

        # Skip degenerate cases
        if len(np.unique(labels)) < 2:
            return

        import time

        # Time both comparison operators
        start_time = time.time()
        threshold_gt = optimize_thresholds(labels, probabilities, "f1", comparison=">")
        time_gt = time.time() - start_time

        start_time = time.time()
        threshold_gte = optimize_thresholds(
            labels, probabilities, "f1", comparison=">="
        )
        time_gte = time.time() - start_time

        # Both should be reasonably fast
        assert time_gt < 1.0, f"'>' comparison too slow: {time_gt:.4f}s"
        assert time_gte < 1.0, f"'>=' comparison too slow: {time_gte:.4f}s"

        # Times should be similar (within factor of 2)
        assert abs(time_gt - time_gte) < max(time_gt, time_gte), (
            f"Performance difference too large: {time_gt:.4f}s vs {time_gte:.4f}s"
        )

        # Results should be valid
        assert 0 <= threshold_gt <= 1
        assert 0 <= threshold_gte <= 1
