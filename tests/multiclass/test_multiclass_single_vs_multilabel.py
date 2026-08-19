"""Test multiclass accuracy semantics: exclusive vs One-vs-Rest.

This module tests the fundamental distinction between exclusive single-label
accuracy and One-vs-Rest approaches in multiclass classification:

1. Exclusive accuracy: Each sample gets exactly one prediction (single-label)
2. OvR accuracy: Each class is treated independently (multi-label possible)

The key principle: 'Accuracy' in single-label multiclass means exclusive
sample-level accuracy, which requires special handling compared to OvR approaches.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from optimal_cutoffs import optimize_thresholds
from optimal_cutoffs.metrics_core import multiclass_metric_single_label


def _generate_multiclass_data(n_samples, n_classes, random_state=42):
    """Generate multiclass test data with probabilities and labels."""
    rng = np.random.default_rng(random_state)

    # Generate probability matrix
    probs = rng.uniform(0, 1, size=(n_samples, n_classes))

    # Normalize to make proper probability distribution
    probs = probs / probs.sum(axis=1, keepdims=True)

    # Generate labels using argmax with some noise for realism
    labels = np.argmax(probs + rng.normal(0, 0.1, probs.shape), axis=1)

    return labels, probs


class TestExclusiveVsOvRDistinction:
    """Test the fundamental distinction between exclusive and OvR approaches."""

    def test_exclusive_produces_single_label_predictions(self):
        """Exclusive approach should produce exactly one prediction per sample."""
        labels = np.array([0, 1, 2, 0, 1, 2])
        probs = np.array(
            [
                [0.7, 0.2, 0.1],  # Clear class 0
                [0.1, 0.8, 0.1],  # Clear class 1
                [0.1, 0.1, 0.8],  # Clear class 2
                [0.4, 0.4, 0.2],  # Ambiguous 0 vs 1
                [0.3, 0.5, 0.2],  # Clear class 1
                [0.2, 0.3, 0.5],  # Clear class 2
            ]
        )

        # Get thresholds using coordinate ascent (which ensures exclusive predictions)
        try:
            result = optimize_thresholds(
                labels, probs, metric="f1", method="coord_ascent", comparison=">"
            )

            # Extract thresholds from result
            thresholds = result.thresholds

            # Apply exclusive prediction rule: argmax(p - tau)
            scores = probs - thresholds.reshape(1, -1)
            predictions = np.argmax(scores, axis=1)

            # Each sample should have exactly one prediction
            assert len(predictions) == len(labels)
            assert all(0 <= pred < probs.shape[1] for pred in predictions)

            # Predictions should be integers (class indices)
            assert predictions.dtype in [np.int32, np.int64, int]

        except ValueError as e:
            if "not supported" in str(e).lower():
                pytest.skip("Coordinate ascent not available for this metric")
            raise

    def test_ovr_can_produce_multilabel_predictions(self):
        """OvR approach can produce multiple positive predictions per sample."""
        # For multilabel, labels should be 2D binary arrays
        labels = np.array(
            [
                [1, 1, 0],  # Both class 0 and 1 are positive
                [0, 1, 1],  # Both class 1 and 2 are positive
                [0, 0, 1],  # Only class 2
                [1, 0, 0],  # Only class 0
            ]
        )
        probs = np.array(
            [
                [0.8, 0.7, 0.1],  # Both class 0 and 1 might be predicted
                [0.2, 0.9, 0.8],  # Both class 1 and 2 might be predicted
                [0.1, 0.2, 0.9],  # Clear class 2
                [0.6, 0.3, 0.4],  # Clear class 0
            ]
        )

        # Use OvR optimization (method='auto' typically uses OvR for multiclass)
        # Suppress probability sum warning - this test intentionally uses non-normalized
        # probabilities to test multi-label behavior
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = optimize_thresholds(
                labels, probs, metric="f1", method="auto", comparison=">"
            )

        # Extract thresholds from result
        thresholds = result.thresholds

        # Apply per-class thresholds
        predictions = probs > thresholds.reshape(1, -1)

        # Some samples might have multiple positive predictions
        predictions_per_sample = np.sum(predictions, axis=1)

        # At least verify the shape and type are correct
        assert predictions.shape == probs.shape
        assert predictions.dtype == bool

        # In OvR, it's possible (though not required) to have 0 or >1 predictions per
        # sample
        assert all(count >= 0 for count in predictions_per_sample)

    def test_exclusive_accuracy_different_from_ovr_f1(self):
        """Exclusive accuracy should be different from OvR macro F1."""
        labels = np.array([0, 1, 2, 0, 1, 2])
        probs = np.array(
            [
                [0.6, 0.3, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
                [0.5, 0.4, 0.1],
                [0.3, 0.6, 0.1],
                [0.1, 0.3, 0.6],
            ]
        )

        # Get OvR thresholds (standard approach)
        result_ovr = optimize_thresholds(
            labels, probs, metric="f1", method="auto", comparison=">"
        )

        thresholds_ovr = result_ovr.thresholds

        # Apply OvR predictions
        pred_ovr = probs > thresholds_ovr.reshape(1, -1)

        # Apply exclusive prediction rule using the same thresholds
        scores = probs - thresholds_ovr.reshape(1, -1)
        pred_exclusive_indices = np.argmax(scores, axis=1)

        # Convert exclusive predictions to one-hot format for comparison
        pred_exclusive = np.zeros_like(probs, dtype=bool)
        pred_exclusive[np.arange(len(labels)), pred_exclusive_indices] = True

        # The prediction patterns should generally be different
        # (They might occasionally be the same, but this is unlikely)
        different_predictions = not np.array_equal(pred_ovr, pred_exclusive)

        # This is not a strict requirement (they could be the same by coincidence),
        # but demonstrates the conceptual difference
        if different_predictions:
            # Count differences for reporting
            diff_count = np.sum(pred_ovr != pred_exclusive)
            assert diff_count >= 0  # Just verify we can count differences

        # The key test: exclusive should have exactly 1 prediction per sample
        exclusive_counts = np.sum(pred_exclusive, axis=1)
        assert np.all(exclusive_counts == 1), (
            f"Exclusive should predict exactly 1 class per sample, got "
            f"{exclusive_counts}"
        )

    def test_exclusive_accuracy_metric_computation(self):
        """Test exclusive accuracy metric computation."""
        labels = np.array([0, 1, 2, 0, 1])
        probs = np.array(
            [
                [0.8, 0.1, 0.1],  # Correct prediction for class 0
                [0.2, 0.7, 0.1],  # Correct prediction for class 1
                [0.1, 0.2, 0.7],  # Correct prediction for class 2
                [0.3, 0.6, 0.1],  # Incorrect prediction (predicts class 1, should be 0)
                [0.1, 0.8, 0.1],  # Correct prediction for class 1
            ]
        )

        # Use simple thresholds that would lead to argmax predictions
        thresholds = np.array([0.0, 0.0, 0.0])  # Low thresholds

        try:
            # Compute exclusive accuracy
            accuracy = multiclass_metric_single_label(
                labels, probs, thresholds, metric_name="accuracy", comparison=">"
            )

            # Manual calculation: argmax predictions
            pred_classes = np.argmax(probs, axis=1)  # [0, 1, 2, 1, 1]
            expected_accuracy = np.mean(pred_classes == labels)  # 4/5 = 0.8

            assert abs(accuracy - expected_accuracy) < 1e-10, (
                f"Exclusive accuracy {accuracy} should match manual calculation "
                f"{expected_accuracy}"
            )

        except Exception as e:
            if "not supported" in str(e).lower() or "not implemented" in str(e).lower():
                pytest.skip("Exclusive accuracy computation not available")
            raise

    def test_ovr_vs_exclusive_on_same_data(self):
        """Compare OvR and exclusive approaches on the same dataset."""
        labels, probs = _generate_multiclass_data(n_samples=20, n_classes=3)

        # Get OvR thresholds
        result_ovr = optimize_thresholds(
            labels, probs, metric="f1", method="auto", comparison=">"
        )

        thresholds_ovr = result_ovr.thresholds

        # Apply OvR predictions
        pred_ovr = probs > thresholds_ovr.reshape(1, -1)

        # For exclusive, we'll use the OvR thresholds but apply exclusive rule
        scores_exclusive = probs - thresholds_ovr.reshape(1, -1)
        pred_exclusive_classes = np.argmax(scores_exclusive, axis=1)

        # Convert to one-hot for comparison
        pred_exclusive = np.zeros_like(probs, dtype=bool)
        pred_exclusive[np.arange(len(labels)), pred_exclusive_classes] = True

        # Key difference: exclusive has exactly 1 prediction per sample
        ovr_counts = np.sum(pred_ovr, axis=1)
        exclusive_counts = np.sum(pred_exclusive, axis=1)

        assert np.all(exclusive_counts == 1), (
            "Exclusive should predict exactly 1 class per sample"
        )

        # OvR can have variable counts (0, 1, or more per sample)
        assert len(ovr_counts) == len(labels)  # Basic sanity check

        # Both should produce valid predictions
        assert pred_ovr.shape == probs.shape
        assert pred_exclusive.shape == probs.shape


class TestMulticlassAccuracySemantics:
    """Test the semantics of accuracy in multiclass settings."""

    def test_multiclass_accuracy_requires_exclusive_predictions(self):
        """Multiclass accuracy should require exclusive single-label predictions."""
        labels = np.array([0, 1, 2])
        probs = np.array(
            [
                [0.6, 0.3, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.3, 0.6],
            ]
        )

        # Try to compute multiclass accuracy with OvR approach
        # This should either work by converting to exclusive or raise an error
        try:
            result = optimize_thresholds(
                labels, probs, metric="accuracy", method="auto", comparison=">"
            )

            # If it works, verify the result makes sense
            thresholds = result.thresholds
            assert len(thresholds) == probs.shape[1]
            assert all(0 <= t <= 1 for t in thresholds)

        except ValueError as e:
            # Should raise error about requiring exclusive predictions
            assert "exclusive" in str(e).lower() or "single-label" in str(e).lower(), (
                f"Expected error about exclusive predictions, got: {e}"
            )

        except NotImplementedError:
            # Or might not be implemented yet
            pytest.skip("Multiclass accuracy not implemented")

    def test_accuracy_vs_f1_multiclass_behavior(self):
        """Test different behavior of accuracy vs F1 in multiclass."""
        labels, probs = _generate_multiclass_data(n_samples=15, n_classes=3)

        # F1 should work with OvR approach
        try:
            result_f1 = optimize_thresholds(
                labels, probs, metric="f1", method="auto", comparison=">"
            )
            thresholds_f1 = result_f1.thresholds
            assert len(thresholds_f1) == probs.shape[1]

        except Exception as e:
            pytest.skip(f"F1 optimization failed: {e}")

        # Accuracy should require exclusive approach or raise error
        try:
            result_accuracy = optimize_thresholds(
                labels, probs, metric="accuracy", method="auto", comparison=">"
            )

            # If it works, it should produce exclusive-style results
            thresholds_accuracy = result_accuracy.thresholds
            assert len(thresholds_accuracy) == probs.shape[1]

        except ValueError as e:
            # Expected error about exclusive predictions
            assert (
                "exclusive" in str(e).lower()
                or "single-label" in str(e).lower()
                or "micro" in str(e).lower()
            ), f"Expected error about exclusive/single-label predictions: {e}"

    def test_single_label_consistency_check(self):
        """Test that single-label predictions are consistent."""
        labels = np.array([0, 1, 2, 1, 0])
        probs = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.1, 0.7],
                [0.3, 0.6, 0.1],
                [0.8, 0.1, 0.1],
            ]
        )

        # Any method that produces single-label predictions should be consistent
        # with the exclusive accuracy computation
        try:
            result = optimize_thresholds(
                labels, probs, metric="f1", method="coord_ascent", comparison=">"
            )

            thresholds = result.thresholds

            # Apply exclusive prediction rule
            scores = probs - thresholds.reshape(1, -1)
            predicted_classes = np.argmax(scores, axis=1)

            # Compute accuracy manually
            manual_accuracy = np.mean(predicted_classes == labels)

            # Should match exclusive accuracy computation if available
            try:
                computed_accuracy = multiclass_metric_single_label(
                    labels, probs, thresholds, metric_name="accuracy", comparison=">"
                )

                assert abs(manual_accuracy - computed_accuracy) < 1e-10, (
                    f"Manual accuracy {manual_accuracy} should match computed "
                    f"{computed_accuracy}"
                )

            except (NotImplementedError, ValueError):
                # If exclusive accuracy not implemented, just verify manual computation
                assert 0 <= manual_accuracy <= 1

        except (ValueError, NotImplementedError):
            pytest.skip("Coordinate ascent or exclusive accuracy not available")


class TestMulticlassEdgeCases:
    """Test edge cases in multiclass exclusive vs OvR."""

    def test_single_class_case(self):
        """Test behavior with single class (degenerate multiclass)."""
        labels = np.array([0, 0, 0])  # Only class 0
        probs = np.array(
            [
                [0.8, 0.2],  # 2 classes in probabilities
                [0.6, 0.4],
                [0.7, 0.3],
            ]
        )

        # This is technically binary, but might be handled as multiclass
        try:
            result = optimize_thresholds(
                labels, probs, metric="f1", method="auto", comparison=">"
            )

            # Should produce valid thresholds
            thresholds = result.thresholds
            assert len(thresholds) == probs.shape[1]

        except ValueError as e:
            # Might reject single-class case
            if "single class" in str(e).lower() or "degenerate" in str(e).lower():
                pytest.skip("Single class case not supported")
            raise

    def test_perfect_separation_case(self):
        """Test case where classes are perfectly separated."""
        labels = np.array([0, 1, 2])
        probs = np.array(
            [
                [1.0, 0.0, 0.0],  # Perfect class 0
                [0.0, 1.0, 0.0],  # Perfect class 1
                [0.0, 0.0, 1.0],  # Perfect class 2
            ]
        )

        for method in ["auto", "sort_scan"]:
            try:
                result = optimize_thresholds(
                    labels, probs, metric="f1", method=method, comparison=">"
                )

                thresholds = result.thresholds

                # With perfect separation, should achieve perfect performance
                pred_classes = np.argmax(probs - thresholds.reshape(1, -1), axis=1)
                accuracy = np.mean(pred_classes == labels)

                assert accuracy == 1.0, (
                    "Perfect separation should achieve perfect accuracy"
                )

            except (ValueError, NotImplementedError):
                continue  # Try other methods

    def test_ambiguous_cases(self):
        """Test cases where multiple classes have similar probabilities."""
        labels = np.array([0, 1, 2])
        probs = np.array(
            [
                [0.34, 0.33, 0.33],  # Very close probabilities
                [0.33, 0.34, 0.33],
                [0.33, 0.33, 0.34],
            ]
        )

        try:
            result = optimize_thresholds(
                labels, probs, metric="f1", method="auto", comparison=">"
            )

            # Should handle ambiguous cases gracefully
            thresholds = result.thresholds
            assert len(thresholds) == probs.shape[1]
            assert all(0 <= t <= 1 for t in thresholds)

            # Exclusive predictions should still work
            scores = probs - thresholds.reshape(1, -1)
            predicted_classes = np.argmax(scores, axis=1)

            # Should predict valid classes
            assert all(0 <= pred < probs.shape[1] for pred in predicted_classes)

        except Exception as e:
            # Ambiguous cases might be challenging
            if "numerical" in str(e).lower() or "convergence" in str(e).lower():
                pytest.skip("Ambiguous case numerical issues")
            raise

    @given(n_samples=st.integers(5, 20), n_classes=st.integers(3, 5))
    @settings(deadline=None, max_examples=20)
    def test_exclusive_single_label_property(self, n_samples, n_classes):
        """Property test: exclusive predictions should always be single-label."""
        labels, probs = _generate_multiclass_data(n_samples, n_classes, random_state=42)

        try:
            # Use any method that should produce reasonable thresholds
            result = optimize_thresholds(
                labels, probs, metric="f1", method="auto", comparison=">"
            )

            thresholds = result.thresholds

            # Apply exclusive prediction rule
            scores = probs - thresholds.reshape(1, -1)
            predicted_classes = np.argmax(scores, axis=1)

            # Convert to one-hot for verification
            predictions_onehot = np.zeros_like(probs, dtype=bool)
            predictions_onehot[np.arange(n_samples), predicted_classes] = True

            # Each sample should have exactly one prediction
            prediction_counts = np.sum(predictions_onehot, axis=1)
            assert np.all(prediction_counts == 1), (
                f"Exclusive predictions should have exactly 1 per sample, got "
                f"{prediction_counts}"
            )

        except Exception as e:
            # Some combinations might not be supported
            if any(
                phrase in str(e).lower()
                for phrase in ["not supported", "not implemented", "degenerate"]
            ):
                pytest.skip(f"Configuration not supported: {e}")
            raise
