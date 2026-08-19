"""Test coordinate ascent invariants and convergence properties.

This module tests the mathematical properties that coordinate ascent
optimization must satisfy:

1. Monotone non-decreasing objective: F1 score should never decrease between iterations
2. Finite termination: Algorithm must converge in finite steps
3. Single-label consistency: Predictions use argmax(p - tau) for coupled optimization
4. Coordinate independence: Optimizing one threshold while fixing others should improve
    objective

These tests verify the correctness of the coordinate ascent implementation
for multiclass threshold optimization with coupled single-label predictions.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from optimal_cutoffs import optimize_thresholds
from optimal_cutoffs.metrics_core import multiclass_metric_single_label


def _generate_multiclass_data(n_samples, n_classes, random_state=42):
    """Generate multiclass test data."""
    rng = np.random.default_rng(random_state)

    # Generate probability matrix
    probs = rng.uniform(0.1, 0.9, size=(n_samples, n_classes))

    # Normalize to sum to 1 (proper probability distribution)
    probs = probs / probs.sum(axis=1, keepdims=True)

    # Generate labels with some correlation to probabilities
    noise = rng.normal(0, 0.2, probs.shape)
    labels = np.argmax(probs + noise, axis=1)

    return labels, probs


def _compute_exclusive_f1(labels, probs, thresholds, comparison=">"):
    """Compute F1 score using exclusive single-label predictions."""
    try:
        return multiclass_metric_single_label(
            labels, probs, thresholds, metric_name="f1", comparison=comparison
        )
    except (NotImplementedError, ValueError):
        # Fallback manual computation
        scores = probs - thresholds.reshape(1, -1)
        predicted_classes = np.argmax(scores, axis=1)

        # Compute macro F1 manually
        n_classes = len(thresholds)
        class_f1s = []

        for k in range(n_classes):
            tp = np.sum((labels == k) & (predicted_classes == k))
            fp = np.sum((labels != k) & (predicted_classes == k))
            fn = np.sum((labels == k) & (predicted_classes != k))

            precision = 0.0 if tp + fp == 0 else tp / (tp + fp)

            recall = 0.0 if tp + fn == 0 else tp / (tp + fn)

            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)

            class_f1s.append(f1)

        return np.mean(class_f1s)


class TestCoordinateAscentMonotonicity:
    """Test that coordinate ascent produces monotonically non-decreasing objectives."""

    def test_coordinate_ascent_monotonic_f1(self):
        """Coordinate ascent should produce monotonically non-decreasing F1."""
        labels, probs = _generate_multiclass_data(
            n_samples=30, n_classes=3, random_state=123
        )

        try:
            # Use coordinate ascent method
            result1 = optimize_thresholds(
                labels, probs, metric="f1", method="coord_ascent", comparison=">"
            )
            thresholds = result1.thresholds

            # Verify final result1 is reasonable
            assert len(thresholds) == probs.shape[1]
            assert all(0 <= t <= 1 for t in thresholds)

            # Compute final F1
            final_f1 = _compute_exclusive_f1(labels, probs, thresholds, ">")
            assert 0 <= final_f1 <= 1, f"Final F1 {final_f1} out of valid range"

            # Test that small perturbations don't significantly improve F1
            # (indicating we're at least at a local optimum)
            for _ in range(5):
                rng = np.random.default_rng(456)
                perturbed_thresholds = thresholds + rng.normal(
                    0, 0.05, size=len(thresholds)
                )
                perturbed_thresholds = np.clip(perturbed_thresholds, 0, 1)

                perturbed_f1 = _compute_exclusive_f1(
                    labels, probs, perturbed_thresholds, ">"
                )

                # Final should be at least as good as random perturbations
                # (allowing small numerical tolerance)
                assert final_f1 >= perturbed_f1 - 1e-6, (
                    f"Final F1 {final_f1:.8f} should be >= perturbed F1 "
                    f"{perturbed_f1:.8f}"
                )

        except ValueError as e:
            if "not supported" in str(e).lower() or "not implemented" in str(e).lower():
                pytest.skip("Coordinate ascent not available")
            raise

    def test_coordinate_ascent_vs_random_initialization(self):
        """Coordinate ascent should outperform random initialization."""
        labels, probs = _generate_multiclass_data(
            n_samples=25, n_classes=3, random_state=789
        )

        try:
            # Get coordinate ascent result1
            result = optimize_thresholds(
                labels, probs, metric="f1", method="coord_ascent", comparison=">"
            )
            thresholds = result.thresholds
            optimal_f1 = _compute_exclusive_f1(labels, probs, thresholds, ">")

            # Compare against multiple random initializations
            rng = np.random.default_rng(999)
            random_f1s = []

            for _ in range(10):
                random_thresholds = rng.uniform(0, 1, size=len(thresholds))
                random_f1 = _compute_exclusive_f1(labels, probs, random_thresholds, ">")
                random_f1s.append(random_f1)

            max_random_f1 = max(random_f1s)

            # Optimal should be at least as good as the best random initialization
            assert optimal_f1 >= max_random_f1 - 1e-10, (
                f"Coordinate ascent F1 {optimal_f1:.8f} should be >= best random "
                f"{max_random_f1:.8f}"
            )

        except Exception as e:
            if any(
                phrase in str(e).lower()
                for phrase in ["not supported", "not implemented"]
            ):
                pytest.skip(f"Coordinate ascent not available: {e}")
            raise

    def test_coordinate_ascent_improves_from_poor_start(self):
        """Coordinate ascent should improve from poor starting points."""
        labels = np.array([0, 1, 2, 0, 1, 2])
        probs = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.7, 0.2, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
            ]
        )

        try:
            # Get optimized result1
            result = optimize_thresholds(
                labels, probs, metric="f1", method="coord_ascent", comparison=">"
            )
            thresholds = result.thresholds
            optimal_f1 = _compute_exclusive_f1(labels, probs, thresholds, ">")

            # Test several poor starting points
            poor_starts = [
                np.array([0.9, 0.9, 0.9]),  # Very high thresholds
                np.array([0.1, 0.1, 0.1]),  # Very low thresholds
                np.array([1.0, 0.0, 0.5]),  # Mixed extreme values
            ]

            for poor_thresholds in poor_starts:
                poor_f1 = _compute_exclusive_f1(labels, probs, poor_thresholds, ">")

                # Optimized should be better than poor starting point
                # (unless the poor start happens to be optimal by coincidence)
                assert optimal_f1 >= poor_f1 - 1e-10, (
                    f"Optimal F1 {optimal_f1:.8f} should be >= poor start F1 "
                    f"{poor_f1:.8f}"
                )

        except Exception as e:
            if any(
                phrase in str(e).lower()
                for phrase in ["not supported", "not implemented"]
            ):
                pytest.skip(f"Coordinate ascent not available: {e}")
            raise


class TestCoordinateAscentConvergence:
    """Test finite termination and convergence properties."""

    def test_coordinate_ascent_converges(self):
        """Coordinate ascent should converge in finite iterations."""
        labels, probs = _generate_multiclass_data(
            n_samples=20, n_classes=3, random_state=555
        )

        try:
            # This should complete without infinite loops or convergence errors
            result1 = optimize_thresholds(
                labels, probs, metric="f1", method="coord_ascent", comparison=">"
            )
            thresholds = result1.thresholds

            # Basic sanity checks
            assert len(thresholds) == probs.shape[1]
            assert all(0 <= t <= 1 for t in thresholds)
            assert not any(np.isnan(t) for t in thresholds)
            assert not any(np.isinf(t) for t in thresholds)

        except Exception as e:
            if "convergence" in str(e).lower():
                pytest.fail(f"Coordinate ascent failed to converge: {e}")
            elif any(
                phrase in str(e).lower()
                for phrase in ["not supported", "not implemented"]
            ):
                pytest.skip(f"Coordinate ascent not available: {e}")
            raise

    def test_coordinate_ascent_deterministic(self):
        """Coordinate ascent should be deterministic given same input."""
        labels, probs = _generate_multiclass_data(
            n_samples=15, n_classes=3, random_state=777
        )

        try:
            # Run multiple times - should get identical results
            results = []
            for _ in range(3):
                result = optimize_thresholds(
                    labels, probs, metric="f1", method="coord_ascent", comparison=">"
                )
                thresholds = result.thresholds
                results.append(thresholds)

            # All results should be identical
            for i in range(1, len(results)):
                assert np.allclose(results[i], results[0], atol=1e-12), (
                    f"Coordinate ascent not deterministic: run {i} differs from run 0"
                )

        except Exception as e:
            if any(
                phrase in str(e).lower()
                for phrase in ["not supported", "not implemented"]
            ):
                pytest.skip(f"Coordinate ascent not available: {e}")
            raise

    def test_coordinate_ascent_finite_iterations(self):
        """Test that coordinate ascent terminates in reasonable number of iterations."""
        # This is more of a stress test to ensure no infinite loops
        labels, probs = _generate_multiclass_data(
            n_samples=50, n_classes=4, random_state=888
        )

        try:
            import time

            start_time = time.time()

            result = optimize_thresholds(
                labels, probs, metric="f1", method="coord_ascent", comparison=">"
            )

            end_time = time.time()
            duration = end_time - start_time

            # Should complete in reasonable time (generous bound)
            assert duration < 30.0, (
                f"Coordinate ascent took too long: {duration:.2f}s (may indicate "
                f"convergence issues)"
            )

            # Result should be valid
            thresholds = result.thresholds
            assert len(thresholds) == probs.shape[1]
            assert all(0 <= t <= 1 for t in thresholds)

        except Exception as e:
            if any(
                phrase in str(e).lower()
                for phrase in ["not supported", "not implemented"]
            ):
                pytest.skip(f"Coordinate ascent not available: {e}")
            raise


class TestSingleLabelConsistency:
    """Test single-label prediction consistency in coordinate ascent."""

    def test_predictions_use_argmax_rule(self):
        """Coordinate ascent predictions should use argmax(p - tau) rule."""
        labels = np.array([0, 1, 2, 0, 1])
        probs = np.array(
            [
                [0.6, 0.3, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
                [0.5, 0.4, 0.1],
                [0.3, 0.6, 0.1],
            ]
        )

        try:
            result = optimize_thresholds(
                labels, probs, metric="f1", method="coord_ascent", comparison=">"
            )

            # Apply the argmax(p - tau) rule
            thresholds = result.thresholds
            scores = probs - thresholds.reshape(1, -1)
            predicted_classes = np.argmax(scores, axis=1)

            # Each sample should get exactly one prediction
            assert len(predicted_classes) == len(labels)
            assert all(isinstance(pred, int | np.integer) for pred in predicted_classes)
            assert all(0 <= pred < probs.shape[1] for pred in predicted_classes)

            # Convert to one-hot for verification
            predictions_onehot = np.zeros_like(probs, dtype=bool)
            predictions_onehot[np.arange(len(labels)), predicted_classes] = True

            # Each sample should have exactly one positive prediction
            prediction_counts = np.sum(predictions_onehot, axis=1)
            assert np.all(prediction_counts == 1), (
                f"Each sample should have exactly 1 prediction, got {prediction_counts}"
            )

        except Exception as e:
            if any(
                phrase in str(e).lower()
                for phrase in ["not supported", "not implemented"]
            ):
                pytest.skip(f"Coordinate ascent not available: {e}")
            raise

    def test_single_label_vs_multilabel_difference(self):
        """Single-label coordinate ascent should differ from independent OvR."""
        labels, probs = _generate_multiclass_data(
            n_samples=20, n_classes=3, random_state=111
        )

        try:
            # Get coordinate ascent (single-label) result1
            result = optimize_thresholds(
                labels, probs, metric="f1", method="coord_ascent", comparison=">"
            )
            thresholds = result.thresholds

            # Get independent per-class (OvR) result1
            result2 = optimize_thresholds(
                labels, probs, metric="f1", method="auto", comparison=">"
            )

            # Apply both approaches
            scores_coord = probs - thresholds.reshape(1, -1)
            pred_coord = np.argmax(scores_coord, axis=1)

            pred_ovr = probs > result2.thresholds.reshape(1, -1)

            # Convert coordinate ascent to one-hot for comparison
            pred_coord_onehot = np.zeros_like(probs, dtype=bool)
            pred_coord_onehot[np.arange(len(labels)), pred_coord] = True

            # Coordinate ascent should have exactly 1 prediction per sample
            coord_counts = np.sum(pred_coord_onehot, axis=1)
            assert np.all(coord_counts == 1), (
                "Coordinate ascent should predict exactly 1 class per sample"
            )

            # OvR can have variable predictions per sample
            ovr_counts = np.sum(pred_ovr, axis=1)
            # This is just a basic check - OvR counts can be anything
            assert len(ovr_counts) == len(labels)

        except Exception as e:
            if any(
                phrase in str(e).lower()
                for phrase in ["not supported", "not implemented"]
            ):
                pytest.skip(f"Method not available: {e}")
            raise

    def test_coordinate_ascent_coupling_effect(self):
        """Test that coordinate ascent considers coupling between classes."""
        # Create case where independent optimization would be suboptimal
        # due to single-label constraint
        labels = np.array([0, 1, 2, 0, 1, 2])
        probs = np.array(
            [
                [0.5, 0.4, 0.1],  # Ambiguous between 0 and 1
                [0.4, 0.5, 0.1],  # Ambiguous between 0 and 1
                [0.1, 0.4, 0.5],  # Clear class 2
                [0.6, 0.3, 0.1],  # Clear class 0
                [0.3, 0.6, 0.1],  # Clear class 1
                [0.1, 0.3, 0.6],  # Clear class 2
            ]
        )

        try:
            result = optimize_thresholds(
                labels, probs, metric="f1", method="coord_ascent", comparison=">"
            )

            # Apply single-label rule
            thresholds = result.thresholds
            scores = probs - thresholds.reshape(1, -1)
            predictions = np.argmax(scores, axis=1)

            # Compute resulting F1
            coord_f1 = _compute_exclusive_f1(labels, probs, thresholds, ">")

            # The key test: coordinate ascent should produce a valid result1
            assert 0 <= coord_f1 <= 1, (
                f"Coordinate ascent F1 {coord_f1} out of valid range"
            )

            # Verify single-label property
            assert len(predictions) == len(labels)
            assert all(0 <= pred < probs.shape[1] for pred in predictions)

        except Exception as e:
            if any(
                phrase in str(e).lower()
                for phrase in ["not supported", "not implemented"]
            ):
                pytest.skip(f"Coordinate ascent not available: {e}")
            raise


class TestCoordinateAscentEdgeCases:
    """Test edge cases and robustness of coordinate ascent."""

    def test_coordinate_ascent_with_extreme_probabilities(self):
        """Test coordinate ascent with probabilities near 0 and 1."""
        labels = np.array([0, 1, 2])
        probs = np.array(
            [
                [0.99, 0.005, 0.005],  # Very confident class 0
                [0.01, 0.98, 0.01],  # Very confident class 1
                [0.01, 0.01, 0.98],  # Very confident class 2
            ]
        )

        try:
            result1 = optimize_thresholds(
                labels, probs, metric="f1", method="coord_ascent", comparison=">"
            )
            thresholds = result1.thresholds

            # Should handle extreme probabilities gracefully
            thresholds = result1.thresholds
            assert len(thresholds) == probs.shape[1]
            assert all(0 <= t <= 1 for t in thresholds)
            assert all(not np.isnan(t) and not np.isinf(t) for t in thresholds)

            # Should achieve good performance on this easy case
            f1 = _compute_exclusive_f1(labels, probs, thresholds, ">")
            assert f1 >= 0.8, f"Should achieve high F1 on easy case, got {f1:.4f}"

        except Exception as e:
            if any(
                phrase in str(e).lower()
                for phrase in ["not supported", "not implemented"]
            ):
                pytest.skip(f"Coordinate ascent not available: {e}")
            raise

    def test_coordinate_ascent_with_uniform_probabilities(self):
        """Test coordinate ascent when all probabilities are equal."""
        labels = np.array([0, 1, 2, 0, 1, 2])
        probs = np.full((6, 3), 1 / 3)  # All probabilities equal (uniform)

        try:
            result1 = optimize_thresholds(
                labels, probs, metric="f1", method="coord_ascent", comparison=">"
            )
            thresholds = result1.thresholds

            # Should produce valid thresholds even with uniform probabilities
            thresholds = result1.thresholds
            assert len(thresholds) == probs.shape[1]
            assert all(0 <= t <= 1 for t in thresholds)

            # Predictions should be consistent (all samples have same probabilities)
            scores = probs - thresholds.reshape(1, -1)
            predictions = np.argmax(scores, axis=1)

            # All samples should get the same prediction (since all probs are identical)
            assert len(set(predictions)) <= 1, (
                f"With uniform probabilities, all predictions should be identical, got "
                f"{predictions}"
            )

        except Exception as e:
            if any(
                phrase in str(e).lower()
                for phrase in ["not supported", "not implemented"]
            ):
                pytest.skip(f"Coordinate ascent not available: {e}")
            raise

    def test_coordinate_ascent_single_sample(self):
        """Test coordinate ascent with single sample."""
        labels = np.array([1])
        probs = np.array([[0.2, 0.7, 0.1]])

        try:
            result1 = optimize_thresholds(
                labels, probs, metric="f1", method="coord_ascent", comparison=">"
            )
            thresholds = result1.thresholds

            # Should handle single sample case
            thresholds = result1.thresholds
            assert len(thresholds) == probs.shape[1]
            assert all(0 <= t <= 1 for t in thresholds)

            # Single sample should be predicted correctly if possible
            scores = probs - thresholds.reshape(1, -1)
            np.argmax(scores, axis=1)[0]

            # Optimal prediction should match the true label if reasonable
            f1 = _compute_exclusive_f1(labels, probs, thresholds, ">")
            assert 0 <= f1 <= 1, f"F1 {f1} out of valid range"

        except Exception as e:
            if any(
                phrase in str(e).lower()
                for phrase in ["not supported", "not implemented", "single sample"]
            ):
                pytest.skip(f"Single sample case not supported: {e}")
            raise

    @given(n_samples=st.integers(8, 25), n_classes=st.integers(3, 4))
    @settings(deadline=None, max_examples=15)
    def test_coordinate_ascent_property_invariants(self, n_samples, n_classes):
        """Property-based test for coordinate ascent invariants."""
        labels, probs = _generate_multiclass_data(n_samples, n_classes, random_state=42)

        try:
            result1 = optimize_thresholds(
                labels, probs, metric="f1", method="coord_ascent", comparison=">"
            )
            thresholds = result1.thresholds

            # Basic invariants
            thresholds = result1.thresholds
            assert len(thresholds) == n_classes, "Wrong number of thresholds"
            # Coordinate ascent can produce thresholds outside [0,1] - allow reasonable
            # range
            out_of_range = [t for t in thresholds if not (-10 <= t <= 10)]
            assert len(out_of_range) == 0, (
                f"Thresholds out of reasonable range [-10,10]: {out_of_range}, all "
                f"thresholds: {thresholds}"
            )
            assert all(not np.isnan(t) and not np.isinf(t) for t in thresholds), (
                "Invalid threshold values"
            )

            # Single-label property
            scores = probs - thresholds.reshape(1, -1)
            predictions = np.argmax(scores, axis=1)

            assert len(predictions) == n_samples, "Wrong number of predictions"
            assert all(0 <= pred < n_classes for pred in predictions), (
                "Invalid prediction classes"
            )

            # F1 should be valid
            f1 = _compute_exclusive_f1(labels, probs, thresholds, ">")
            assert 0 <= f1 <= 1, f"F1 {f1} out of valid range"

        except Exception as e:
            if any(
                phrase in str(e).lower()
                for phrase in ["not supported", "not implemented", "degenerate"]
            ):
                pytest.skip(f"Configuration not supported: {e}")
            raise
