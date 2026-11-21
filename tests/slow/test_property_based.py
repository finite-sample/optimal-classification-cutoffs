"""Hypothesis property tests for algorithm correctness and invariants."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from sklearn.metrics import f1_score

from optimal_cutoffs import optimize_thresholds
from optimal_cutoffs.metrics_core import (
    compute_metric_at_threshold,
    compute_multiclass_metrics_from_labels,
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

        # Strict cutoffs at each observed score and just below the minimum
        # cover every distinct prediction, including tied and constant inputs.
        candidates = [
            np.nextafter(probabilities.min(), -np.inf),
            *np.unique(probabilities),
        ]
        best = dict.fromkeys(["f1", "accuracy", "precision", "recall"], 0.0)
        for threshold in candidates:
            pred = probabilities > threshold
            tp = np.count_nonzero(pred & (labels == 1))
            denom = pred.sum() + labels.sum()
            values = {
                "accuracy": np.mean(pred == labels),
                "f1": 2 * tp / denom if denom else 0.0,
                "precision": tp / pred.sum() if pred.any() else 0.0,
                "recall": tp / labels.sum() if labels.any() else 0.0,
            }
            for metric, value in values.items():
                best[metric] = max(best[metric], value)

        for metric, value in best.items():
            result = optimize_thresholds(labels, probabilities, metric=metric)
            assert result.score == pytest.approx(value, abs=1e-12)

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
        threshold_up = optimize_thresholds(labels, shifted_up, metric="f1").threshold
        threshold_down = optimize_thresholds(
            labels, shifted_down, metric="f1"
        ).threshold

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
        threshold_gt = optimize_thresholds(
            labels, probabilities, metric="f1", comparison=">"
        ).threshold
        threshold_gte = optimize_thresholds(
            labels, probabilities, metric="f1", comparison=">="
        ).threshold

        # Both should be valid thresholds
        assert np.nextafter(0.0, -np.inf) <= threshold_gt <= np.nextafter(1.0, np.inf)
        assert np.nextafter(0.0, -np.inf) <= threshold_gte <= np.nextafter(1.0, np.inf)

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

        perfect_probs = np.where(labels == 1, 0.75, 0.25)
        for metric in ["accuracy", "f1"]:
            original = optimize_thresholds(labels, probabilities, metric=metric)
            perfect = optimize_thresholds(labels, perfect_probs, metric=metric)
            np.testing.assert_array_equal(perfect.predict(perfect_probs), labels)
            assert perfect.score == 1.0
            assert perfect.score >= original.score

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
        method=st.sampled_from(["sort_scan", "minimize", "gradient"]),
    )
    @settings(max_examples=20)
    def test_optimization_method_consistency(self, data, method):
        """Different optimization methods should produce reasonable results."""
        labels, probabilities = data

        # Skip degenerate cases
        if len(np.unique(labels)) < 2:
            return

        result = optimize_thresholds(labels, probabilities, metric="f1", method=method)
        threshold = result.threshold
        assert np.nextafter(0.0, -np.inf) <= threshold <= np.nextafter(1.0, np.inf)
        tp, tn, fp, fn = confusion_matrix_at_threshold(labels, probabilities, threshold)
        assert tp + tn + fp + fn == len(labels)


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
        assert np.nextafter(0.0, -np.inf) <= threshold <= np.nextafter(1.0, np.inf)

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
        assert np.nextafter(0.0, -np.inf) <= threshold <= np.nextafter(1.0, np.inf)

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
        """Independent one-vs-rest fits produce a cutoff for each class."""
        labels, probabilities = data
        result = optimize_thresholds(
            labels, probabilities, metric="f1", method="independent"
        )

        assert result.thresholds.shape == (probabilities.shape[1],)
        assert np.all((result.thresholds >= 0) & (result.thresholds <= 1))

    @given(matched_multiclass_data(n_classes=3, min_size=15, max_size=25))
    @settings(max_examples=15)
    def test_multiclass_ovr_independence(self, data):
        """OvR cutoffs and predictions agree with separate binary fits."""
        labels, probabilities = data
        result = optimize_thresholds(
            labels, probabilities, metric="f1", method="independent"
        )
        individual = [
            optimize_thresholds(labels == k, probabilities[:, k], metric="f1")
            for k in range(probabilities.shape[1])
        ]
        np.testing.assert_allclose(
            result.thresholds, [fit.threshold for fit in individual]
        )
        np.testing.assert_array_equal(
            result.predict(probabilities),
            np.column_stack(
                [fit.predict(probabilities[:, k]) for k, fit in enumerate(individual)]
            ),
        )

    @given(
        data=matched_multiclass_data(n_classes=3, min_size=12, max_size=20),
        average=st.sampled_from(["macro", "micro", "weighted"]),
    )
    @settings(max_examples=15)
    def test_multiclass_averaging_consistency(self, data, average):
        """Score single-label predictions with each requested F1 average."""
        labels, probabilities = data
        result = optimize_thresholds(
            labels, probabilities, metric="f1", average=average
        )
        assert result.thresholds.shape == (probabilities.shape[1],)
        predicted = result.predict(probabilities)
        assert predicted.shape == labels.shape
        expected = f1_score(labels, predicted, average=average, zero_division=0)
        assert compute_multiclass_metrics_from_labels(
            labels, predicted, metric="f1", average=average
        ) == pytest.approx(expected)

    @given(matched_multiclass_data(n_classes=4, min_size=16, max_size=24))
    @settings(max_examples=10)
    def test_multiclass_determinism(self, data):
        """Repeated fits produce the same thresholds, scores, and predictions."""
        labels, probabilities = data
        first = optimize_thresholds(labels, probabilities, metric="accuracy")
        second = optimize_thresholds(labels, probabilities, metric="accuracy")
        np.testing.assert_array_equal(first.thresholds, second.thresholds)
        np.testing.assert_array_equal(first.scores, second.scores)
        np.testing.assert_array_equal(
            first.predict(probabilities), second.predict(probabilities)
        )

    @given(
        data=matched_multiclass_data(n_classes=3, min_size=9, max_size=18),
        metric=st.sampled_from(["f1", "precision", "recall", "accuracy"]),
    )
    @settings(max_examples=15)
    def test_multiclass_metric_bounds(self, data, metric):
        """Check which one-vs-rest averages are defined for each metric."""
        labels, probabilities = data
        result = optimize_thresholds(
            labels, probabilities, metric=metric, method="independent"
        )
        cms = multiclass_confusion_matrices_at_thresholds(
            labels, probabilities, result.thresholds, comparison=">"
        )
        for average in ("macro", "weighted"):
            assert 0 <= multiclass_metric_ovr(cms, metric, average) <= 1
        if metric == "accuracy":
            with pytest.raises(ValueError, match="exclusive single-label"):
                multiclass_metric_ovr(cms, metric, "micro")
        else:
            assert 0 <= multiclass_metric_ovr(cms, metric, "micro") <= 1

    @given(matched_multiclass_data(n_classes=3, min_size=12, max_size=20))
    @settings(max_examples=10)
    def test_multiclass_comparison_operators(self, data):
        """Check the fitted comparison rule, including ties at each cutoff."""
        labels, probabilities = data
        for comparison, compare in ((">", np.greater), (">=", np.greater_equal)):
            result = optimize_thresholds(
                labels,
                probabilities,
                metric="f1",
                method="independent",
                comparison=comparison,
            )
            at_and_around = np.vstack([probabilities, result.thresholds])
            np.testing.assert_array_equal(
                result.predict(at_and_around), compare(at_and_around, result.thresholds)
            )


class TestMulticlassMathematicalProperties:
    """Test mathematical properties specific to multiclass classification."""

    @given(matched_multiclass_data(n_classes=3, min_size=15, max_size=25))
    @settings(max_examples=10)
    def test_multiclass_macro_micro_weighted_relationships(self, data):
        """Check equal class weights, support weights, and pooled counts."""
        labels, probabilities = data
        result = optimize_thresholds(
            labels, probabilities, metric="f1", method="independent"
        )
        cms = multiclass_confusion_matrices_at_thresholds(
            labels, probabilities, result.thresholds, comparison=">"
        )
        per_class = multiclass_metric_ovr(cms, "f1", "none")
        np.testing.assert_allclose(result.score, np.mean(per_class))
        assert multiclass_metric_ovr(cms, "f1", "macro") == pytest.approx(
            np.mean(per_class)
        )
        assert multiclass_metric_ovr(cms, "f1", "weighted") == pytest.approx(
            np.average(per_class, weights=np.bincount(labels))
        )
        tp, _, fp, fn = np.sum(cms, axis=0)
        assert multiclass_metric_ovr(cms, "f1", "micro") == pytest.approx(
            2 * tp / (2 * tp + fp + fn)
        )

    @given(matched_multiclass_data(n_classes=3, min_size=12, max_size=20))
    @settings(max_examples=10)
    def test_multiclass_threshold_vector_properties(self, data):
        """Check optimal per-class precision against every distinct cutoff."""
        labels, probabilities = data
        result = optimize_thresholds(
            labels, probabilities, metric="precision", method="independent"
        )
        for k, selected in enumerate(result.predict(probabilities).T):
            scores = probabilities[:, k]
            positives = labels == k
            counts = np.count_nonzero
            achieved = counts(positives & selected.astype(bool)) / max(
                1, counts(selected)
            )
            candidates = np.r_[0.0, np.unique(scores)]
            best = max(
                counts(positives & (scores > cut)) / max(1, counts(scores > cut))
                for cut in candidates
            )
            assert achieved == pytest.approx(best)


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
        assert np.nextafter(0.0, -np.inf) <= threshold <= np.nextafter(1.0, np.inf)

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
        assert np.nextafter(0.0, -np.inf) <= threshold <= np.nextafter(1.0, np.inf)

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

        for method in ["sort_scan", "minimize", "gradient"]:
            start = time.perf_counter()
            result = optimize_thresholds(
                labels, probabilities, metric="f1", method=method
            )
            elapsed = time.perf_counter() - start
            assert elapsed < 5.0, f"{method} took {elapsed:.2f}s"
            assert (
                np.nextafter(0.0, -np.inf)
                <= result.threshold
                <= np.nextafter(1.0, np.inf)
            )

    @given(matched_multiclass_data(n_classes=3, min_size=30, max_size=200))
    @settings(max_examples=5, deadline=15000)
    def test_multiclass_performance_scaling(self, data):
        """The independent solver performs one binary fit of sample length per class."""
        from unittest.mock import patch

        from optimal_cutoffs.binary import optimize_metric_binary

        labels, probabilities = data
        with patch(
            "optimal_cutoffs.binary.optimize_metric_binary",
            wraps=optimize_metric_binary,
        ) as fit:
            result = optimize_thresholds(
                labels, probabilities, metric="f1", method="independent"
            )

        assert fit.call_count == probabilities.shape[1]
        for k, call in enumerate(fit.call_args_list):
            np.testing.assert_array_equal(call.args[0], labels == k)
            np.testing.assert_array_equal(call.args[1], probabilities[:, k])
        assert result.thresholds.shape == (probabilities.shape[1],)

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
        assert np.nextafter(0.0, -np.inf) <= threshold <= np.nextafter(1.0, np.inf)

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
            start = time.perf_counter()
            result = optimize_thresholds(test_labels, test_probs, metric="accuracy")
            elapsed = time.perf_counter() - start
            assert elapsed < 1.0, f"{case_name} took {elapsed:.4f}s"
            assert (
                np.nextafter(0.0, -np.inf)
                <= result.threshold
                <= np.nextafter(1.0, np.inf)
            )

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
        threshold_gt = optimize_thresholds(
            labels, probabilities, metric="f1", comparison=">"
        ).threshold
        time_gt = time.time() - start_time

        start_time = time.time()
        threshold_gte = optimize_thresholds(
            labels, probabilities, metric="f1", comparison=">="
        ).threshold
        time_gte = time.time() - start_time

        # Both should be reasonably fast
        assert time_gt < 1.0, f"'>' comparison too slow: {time_gt:.4f}s"
        assert time_gte < 1.0, f"'>=' comparison too slow: {time_gte:.4f}s"

        # Times should be similar (within factor of 2)
        assert abs(time_gt - time_gte) < max(time_gt, time_gte), (
            f"Performance difference too large: {time_gt:.4f}s vs {time_gte:.4f}s"
        )

        # Results should be valid
        assert np.nextafter(0.0, -np.inf) <= threshold_gt <= np.nextafter(1.0, np.inf)
        assert np.nextafter(0.0, -np.inf) <= threshold_gte <= np.nextafter(1.0, np.inf)
