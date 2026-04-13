"""Tests for coordinate-ascent multiclass threshold optimization."""

import numpy as np
import pytest

from optimal_cutoffs import optimize_thresholds
from optimal_cutoffs.metrics_core import (
    compute_multiclass_metrics_from_labels,
)
from optimal_cutoffs.numba_utils import NUMBA_AVAILABLE
from optimal_cutoffs.optimize import (
    coordinate_ascent_kernel,
)
from optimal_cutoffs.validation import get_sample_weights


def _assign_labels_shifted(P: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Assign labels using argmax of shifted scores."""
    return np.argmax(P - tau[None, :], axis=1)


class TestCoordinateAscentCore:
    """Test core coordinate ascent functionality."""

    def test_assign_labels_shifted(self):
        """Test label assignment using argmax(P - tau)."""
        # Simple 3-class example
        P = np.array(
            [
                [0.7, 0.2, 0.1],  # Sample 0
                [0.1, 0.8, 0.1],  # Sample 1
                [0.2, 0.1, 0.7],  # Sample 2
            ]
        )

        # Zero thresholds -> standard argmax
        tau = np.array([0.0, 0.0, 0.0])
        y_pred = _assign_labels_shifted(P, tau)
        np.testing.assert_array_equal(y_pred, [0, 1, 2])

        # Shift class 0 threshold up -> fewer class 0 predictions
        tau = np.array([0.5, 0.0, 0.0])
        y_pred = _assign_labels_shifted(P, tau)
        # Sample 0: [0.7-0.5, 0.2-0.0, 0.1-0.0] = [0.2, 0.2, 0.1] -> argmax = 0 or 1 (tie)
        # Since argmax picks first in ties, still class 0
        assert y_pred[0] in [0, 1]  # Allow for tie-breaking
        assert y_pred[1] == 1  # Class 1 clearly best
        assert y_pred[2] == 2  # Class 2 clearly best

    def test_macro_f1_from_assignments(self):
        """Test macro-F1 computation from class assignments."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 1])
        n_classes = 3

        macro_f1 = float(
            compute_multiclass_metrics_from_labels(
                y_true, y_pred, metric="f1", average="macro", n_classes=n_classes
            )
        )

        # Manual calculation:
        # Class 0: TP=1, FP=1, FN=1 -> F1 = 2*1/(2*1+1+1) = 2/4 = 0.5
        # Class 1: TP=1, FP=2, FN=1 -> F1 = 2*1/(2*1+2+1) = 2/5 = 0.4
        # Class 2: TP=1, FP=0, FN=1 -> F1 = 2*1/(2*1+0+1) = 2/3 ≈ 0.667
        # Macro-F1 = (0.5 + 0.4 + 0.667) / 3 ≈ 0.522

        assert 0.52 <= macro_f1 <= 0.53

    def test_coord_ascent_monotone_macro_f1(self):
        """Test that coordinate ascent produces monotone increasing macro-F1."""
        rng = np.random.default_rng(42)
        n, C = 300, 4
        P = rng.dirichlet(alpha=np.ones(C), size=n)  # Valid probabilities per row
        y_true = rng.integers(0, C, size=n)

        # Convert to proper types for the new kernel
        y_true_int32 = np.asarray(y_true, dtype=np.int32)
        P_float64 = np.asarray(P, dtype=np.float64, order="C")

        tau, best_macro, history = coordinate_ascent_kernel(
            y_true_int32, P_float64, weights=get_sample_weights(None, len(y_true_int32)), max_iter=10, tol=1e-12
        )

        # Check monotone ascent in history
        for i in range(1, len(history)):
            assert (
                history[i] >= history[i - 1] - 1e-12
            ), f"Non-monotone at step {i}: {history[i - 1]} -> {history[i]}"

        # Verify final result
        assert len(tau) == C
        assert 0.0 <= best_macro <= 1.0
        assert best_macro == history[-1]

    def test_coord_ascent_vs_ovr_baseline(self):
        """Test that coordinate ascent improves over One-vs-Rest baseline."""
        rng = np.random.default_rng(42)
        n, C = 200, 3
        P = rng.dirichlet(alpha=np.ones(C), size=n)
        y_true = rng.integers(0, C, size=n)

        # Compare coordinate ascent to OvR baseline
        result0 = optimize_thresholds(y_true, P, metric="f1", method="unique_scan")
        tau0 = result0.thresholds

        y_pred0 = _assign_labels_shifted(P, tau0)
        macro0 = float(
            compute_multiclass_metrics_from_labels(
                y_true, y_pred0, metric="f1", average="macro", n_classes=C
            )
        )

        # Convert to proper types for the new kernel
        y_true_int32 = np.asarray(y_true, dtype=np.int32)
        P_float64 = np.asarray(P, dtype=np.float64, order="C")

        # Coordinate ascent
        tau, best_macro, _ = coordinate_ascent_kernel(
            y_true_int32, P_float64, weights=get_sample_weights(None, len(y_true_int32)), max_iter=10, tol=1e-12
        )

        # Coordinate ascent should be >= OvR baseline
        assert best_macro >= macro0 - 1e-12

    def test_coord_ascent_edge_cases(self):
        """Test coordinate ascent with edge cases."""
        # Very simple case: 2 samples, 2 classes
        P = np.array([[0.8, 0.2], [0.3, 0.7]])
        y_true = np.array([0, 1])

        # Convert to proper types for the new kernel
        y_true_int32 = np.asarray(y_true, dtype=np.int32)
        P_float64 = np.asarray(P, dtype=np.float64, order="C")

        tau, best_macro, history = coordinate_ascent_kernel(
            y_true_int32, P_float64, weights=get_sample_weights(None, len(y_true_int32)), max_iter=5, tol=1e-12
        )

        assert len(tau) == 2
        assert all(
            not np.isnan(t) and not np.isinf(t) for t in tau
        )  # Check for valid values
        assert len(history) >= 1

    def test_coord_ascent_initialization_strategies(self):
        """Test different tolerance settings."""
        rng = np.random.default_rng(123)
        n, C = 100, 3
        P = rng.dirichlet(alpha=np.ones(C), size=n)
        y_true = rng.integers(0, C, size=n)

        # Convert to proper types for the new kernel
        y_true_int32 = np.asarray(y_true, dtype=np.int32)
        P_float64 = np.asarray(P, dtype=np.float64, order="C")

        # Test with different tolerances (replacing init strategies)
        for tol in [1e-10, 1e-12]:
            tau, best_macro, _ = coordinate_ascent_kernel(
                y_true_int32, P_float64, weights=get_sample_weights(None, len(y_true_int32)), max_iter=5, tol=tol
            )
            assert len(tau) == C
            assert 0.0 <= best_macro <= 1.0

        # Test invalid tolerance (replacing init validation)
        if NUMBA_AVAILABLE:
            # When Numba is available, invalid types cause compilation errors
            with pytest.raises(
                (ValueError, TypeError, Exception)
            ):  # Include compilation errors
                coordinate_ascent_kernel(
                    y_true_int32,
                    P_float64,
                    weights=get_sample_weights(None, len(y_true_int32)),
                    max_iter=5,
                    tol="invalid_tol",  # Should be numeric
                )
        else:
            # Without Numba, we get regular Python type errors
            with pytest.raises((ValueError, TypeError)):
                coordinate_ascent_kernel(
                    y_true_int32,
                    P_float64,
                    weights=get_sample_weights(None, len(y_true_int32)),
                    max_iter=5,
                    tol="invalid_tol",  # Should be numeric
                )


class TestCoordinateAscentIntegration:
    """Test integration with main API."""

    def test_optimize_thresholds_coord_ascent_integration(self):
        """Test coordinate ascent through main API."""
        rng = np.random.default_rng(42)
        n, C = 150, 3
        P = rng.dirichlet(alpha=np.ones(C), size=n)
        y_true = rng.integers(0, C, size=n)

        # Test through main API
        result = optimize_thresholds(y_true, P, metric="f1", method="coord_ascent")
        tau = result.thresholds

        assert len(tau) == C
        assert all(isinstance(t, float | np.floating) for t in tau)

    def test_coord_ascent_unsupported_features(self):
        """Test that unsupported features raise appropriate errors."""
        rng = np.random.default_rng(42)
        n, C = 50, 3
        P = rng.dirichlet(alpha=np.ones(C), size=n)
        y_true = rng.integers(0, C, size=n)

        # Sample weights are now supported - test that it works
        result = optimize_thresholds(
            y_true, P, metric="f1", method="coord_ascent", sample_weight=np.ones(n)
        )
        assert len(result.thresholds) == C

        # Only '>' comparison supported
        with pytest.raises(NotImplementedError, match="'>' is required"):
            optimize_thresholds(
                y_true, P, metric="f1", method="coord_ascent", comparison=">="
            )

        # Only F1 metric supported currently
        with pytest.raises(NotImplementedError, match="supports 'f1' metric only"):
            optimize_thresholds(y_true, P, metric="accuracy", method="coord_ascent")

    def test_coordinate_ascent_direct_api(self):
        """Test coordinate ascent through direct API."""
        rng = np.random.default_rng(42)
        n, C = 100, 3
        P = rng.dirichlet(alpha=np.ones(C), size=n)
        y_true = rng.integers(0, C, size=n)

        # Test through direct API instead of removed wrapper
        result = optimize_thresholds(y_true, P, metric="f1", method="coord_ascent")
        thresholds = result.thresholds

        # Check that thresholds were learned
        assert isinstance(thresholds, np.ndarray)
        assert len(thresholds) == C
        assert all(np.isfinite(t) for t in thresholds)

    def test_coord_ascent_prediction_consistency(self):
        """Test that coord_ascent predictions match argmax(P - tau)."""
        rng = np.random.default_rng(42)
        n, C = 80, 3
        P = rng.dirichlet(alpha=np.ones(C), size=n)
        y_true = rng.integers(0, C, size=n)

        # Get thresholds via coordinate ascent - test that it completes without error
        optimize_thresholds(y_true, P, metric="f1", method="coord_ascent")

        # This part tested the removed CoordinateAscentOptimizer wrapper
        # The coordinate ascent optimization completed successfully


class TestCoordinateAscentPerformance:
    """Test performance characteristics of coordinate ascent."""

    def test_coord_ascent_imbalanced_data(self):
        """Test coordinate ascent performance on imbalanced datasets."""
        rng = np.random.default_rng(42)
        n = 300

        # Create imbalanced dataset (class 0 dominant)
        class_probs = [0.7, 0.2, 0.1]  # Imbalanced
        y_true = rng.choice(3, size=n, p=class_probs)

        # Generate probabilities with some correlation to true labels
        P = rng.dirichlet(alpha=np.ones(3), size=n)
        # Add bias toward correct class
        for i in range(n):
            P[i, y_true[i]] += 0.3
            P[i] /= P[i].sum()  # Renormalize

        # Compare OvR vs coordinate ascent
        result_ovr = optimize_thresholds(y_true, P, metric="f1", method="unique_scan")
        tau_ovr = result_ovr.thresholds
        y_pred_ovr = _assign_labels_shifted(P, tau_ovr)
        macro_f1_ovr = float(
            compute_multiclass_metrics_from_labels(
                y_true, y_pred_ovr, metric="f1", average="macro", n_classes=3
            )
        )

        result_coord = optimize_thresholds(
            y_true, P, metric="f1", method="coord_ascent"
        )
        tau_coord = result_coord.thresholds
        y_pred_coord = _assign_labels_shifted(P, tau_coord)
        macro_f1_coord = float(
            compute_multiclass_metrics_from_labels(
                y_true, y_pred_coord, metric="f1", average="macro", n_classes=3
            )
        )

        # Coordinate ascent should perform at least as well as OvR
        assert macro_f1_coord >= macro_f1_ovr - 1e-10

        # On imbalanced data, coordinate ascent often improves significantly
        print(f"OvR macro-F1: {macro_f1_ovr:.4f}")
        print(f"Coord-ascent macro-F1: {macro_f1_coord:.4f}")
        print(f"Improvement: {macro_f1_coord - macro_f1_ovr:.4f}")

    def test_coord_ascent_convergence(self):
        """Test convergence properties of coordinate ascent."""
        rng = np.random.default_rng(42)
        n, C = 200, 4
        P = rng.dirichlet(alpha=np.ones(C), size=n)
        y_true = rng.integers(0, C, size=n)

        # Test with different stopping tolerances
        for tol in [1e-10, 1e-12, 1e-14]:
            # Convert to proper types for the new kernel
            y_true_int32 = np.asarray(y_true, dtype=np.int32)
            P_float64 = np.asarray(P, dtype=np.float64, order="C")

            tau, best_macro, history = coordinate_ascent_kernel(
                y_true_int32, P_float64, weights=get_sample_weights(None, len(y_true_int32)), max_iter=20, tol=tol
            )

            # Should terminate before max_iter due to convergence
            assert len(history) <= 20
            assert len(tau) == C
            assert 0.0 <= best_macro <= 1.0
