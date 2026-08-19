"""Test comparison operator functionality for thresholding (inclusive vs exclusive)."""

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix

from optimal_cutoffs import optimize_thresholds

# from optimal_cutoffs.wrapper import ThresholdOptimizer  # Disabled - wrapper removed


def confusion_matrix_at_threshold(
    y_true, y_prob, threshold, sample_weight=None, comparison=">="
):
    """Compute confusion matrix at a threshold with comparison operator."""
    if comparison == ">=":
        y_pred = (y_prob >= threshold).astype(int)
    elif comparison == ">":
        y_pred = (y_prob > threshold).astype(int)
    else:
        raise ValueError(f"Invalid comparison operator: {comparison}")

    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        return tp, tn, fp, fn
    # Handle edge cases where not all classes are present
    return 0, 0, 0, 0


def multiclass_confusion_matrices_at_thresholds(
    y_true, y_prob, thresholds, comparison=">="
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
            y_true_binary, y_prob_class, threshold, comparison=comparison
        )
        cms.append((tp, tn, fp, fn))

    return cms


class TestComparisonOperators:
    """Test inclusive vs exclusive thresholding behavior."""

    def test_binary_confusion_matrix_comparison_operators(self):
        """Test that comparison operators affect confusion matrix calculations."""
        # Simple test case where comparison operator should make a difference
        true_labels = np.array([0, 1, 0, 1])
        pred_probs = np.array(
            [0.3, 0.5, 0.7, 0.5]
        )  # Note: two predictions exactly at 0.5
        threshold = 0.5

        # With ">" (exclusive), prob=0.5 predictions should be negative
        tp_gt, tn_gt, fp_gt, fn_gt = confusion_matrix_at_threshold(
            true_labels, pred_probs, threshold, comparison=">"
        )

        # With ">=" (inclusive), prob=0.5 predictions should be positive
        tp_gte, tn_gte, fp_gte, fn_gte = confusion_matrix_at_threshold(
            true_labels, pred_probs, threshold, comparison=">="
        )

        # Verify the results make sense
        # For ">": predictions are [0, 0, 1, 0] -> TP=0, TN=1, FP=1, FN=2
        assert tp_gt == 0
        assert tn_gt == 1
        assert fp_gt == 1
        assert fn_gt == 2

        # For ">=": predictions are [0, 1, 1, 1] -> TP=2, TN=1, FP=1, FN=0
        assert tp_gte == 2
        assert tn_gte == 1
        assert fp_gte == 1
        assert fn_gte == 0

        # Verify they are different
        assert (tp_gt, tn_gt, fp_gt, fn_gt) != (tp_gte, tn_gte, fp_gte, fn_gte)

    def test_binary_threshold_optimization_comparison_operators(self):
        """Test that comparison operators can affect optimal threshold selection."""
        # Create data where the optimal threshold might be different
        # depending on whether we use > or >=
        np.random.seed(42)
        true_labels = np.random.randint(0, 2, 100)
        pred_probs = np.random.rand(100)

        # Get optimal thresholds with both operators
        thresh_gt = optimize_thresholds(
            true_labels, pred_probs, metric="f1", comparison=">"
        )
        thresh_gte = optimize_thresholds(
            true_labels, pred_probs, metric="f1", comparison=">="
        )

        # Both should be valid thresholds
        assert 0 <= thresh_gt.threshold <= 1
        assert 0 <= thresh_gte.threshold <= 1

        # They might be the same or different, both are valid
        # The key is that the function accepts both operators without error

    def test_multiclass_confusion_matrix_comparison_operators(self):
        """Test comparison operators with multiclass confusion matrices."""
        true_labels = np.array([0, 1, 2, 0, 1, 2])
        pred_probs = np.array(
            [
                [0.7, 0.2, 0.1],  # class 0
                [0.1, 0.6, 0.3],  # class 1
                [0.2, 0.3, 0.5],  # class 2
                [0.5, 0.3, 0.2],  # class 0 (tie at threshold)
                [0.3, 0.5, 0.2],  # class 1 (tie at threshold)
                [0.1, 0.4, 0.5],  # class 2 (tie at threshold)
            ]
        )
        thresholds = np.array([0.5, 0.5, 0.5])  # All thresholds at 0.5

        # Get confusion matrices with both operators
        cms_gt = multiclass_confusion_matrices_at_thresholds(
            true_labels, pred_probs, thresholds, comparison=">"
        )
        cms_gte = multiclass_confusion_matrices_at_thresholds(
            true_labels, pred_probs, thresholds, comparison=">="
        )

        # Should have confusion matrices for 3 classes
        assert len(cms_gt) == 3
        assert len(cms_gte) == 3

        # The results should potentially be different due to tie-breaking
        # At minimum, the function should work without errors
        for i in range(3):
            tp_gt, tn_gt, fp_gt, fn_gt = cms_gt[i]
            tp_gte, tn_gte, fp_gte, fn_gte = cms_gte[i]

            # All values should be non-negative
            assert tp_gt >= 0
            assert tn_gt >= 0
            assert fp_gt >= 0
            assert fn_gt >= 0
            assert tp_gte >= 0
            assert tn_gte >= 0
            assert fp_gte >= 0
            assert fn_gte >= 0

            # Total should equal number of samples
            assert tp_gt + tn_gt + fp_gt + fn_gt == 6
            assert tp_gte + tn_gte + fp_gte + fn_gte == 6

    def test_multiclass_threshold_optimization_comparison_operators(self):
        """Test comparison operators with multiclass threshold optimization."""
        # Simple multiclass data
        np.random.seed(123)
        n_samples = 50
        n_classes = 3
        true_labels = np.random.randint(0, n_classes, n_samples)
        pred_probs = np.random.rand(n_samples, n_classes)
        # Normalize to make them proper probabilities
        pred_probs = pred_probs / pred_probs.sum(axis=1, keepdims=True)

        # Get optimal thresholds with both operators
        thresh_gt = optimize_thresholds(
            true_labels, pred_probs, metric="f1", comparison=">"
        )
        thresh_gte = optimize_thresholds(
            true_labels, pred_probs, metric="f1", comparison=">="
        )

        # Should return arrays of thresholds
        assert isinstance(thresh_gt.thresholds, np.ndarray)
        assert isinstance(thresh_gte.thresholds, np.ndarray)
        assert len(thresh_gt.thresholds) == n_classes
        assert len(thresh_gte.thresholds) == n_classes

        # All thresholds should be finite (coordinate ascent can produce thresholds
        # outside [0,1])
        assert np.all(np.isfinite(thresh_gt.thresholds)), "Thresholds should be finite"
        assert np.all(np.isfinite(thresh_gte.thresholds)), "Thresholds should be finite"

    def test_comparison_operator_validation(self):
        """Test that invalid comparison operators raise appropriate errors."""
        true_labels = np.array([0, 1, 0, 1])
        pred_probs = np.array([0.2, 0.8, 0.3, 0.7])

        # Invalid comparison operators should raise ValueError
        with pytest.raises(ValueError, match="Invalid comparison operator"):
            optimize_thresholds(true_labels, pred_probs, comparison="<")

        with pytest.raises(ValueError, match="Invalid comparison operator"):
            optimize_thresholds(true_labels, pred_probs, comparison="==")

        with pytest.raises(ValueError, match="Invalid comparison operator"):
            confusion_matrix_at_threshold(true_labels, pred_probs, 0.5, comparison="!=")

    def test_edge_cases_with_comparison_operators(self):
        """Test edge cases that might behave differently with different operators."""
        # Case where all probabilities equal the threshold
        true_labels = np.array([0, 1, 0, 1])
        pred_probs = np.array([0.5, 0.5, 0.5, 0.5])  # All equal to threshold
        threshold = 0.5

        # With ">", all predictions should be negative (0)
        tp_gt, tn_gt, fp_gt, fn_gt = confusion_matrix_at_threshold(
            true_labels, pred_probs, threshold, comparison=">"
        )
        # Expected: predictions=[0,0,0,0], so TP=0, TN=2, FP=0, FN=2
        assert tp_gt == 0
        assert tn_gt == 2
        assert fp_gt == 0
        assert fn_gt == 2

        # With ">=", all predictions should be positive (1)
        tp_gte, tn_gte, fp_gte, fn_gte = confusion_matrix_at_threshold(
            true_labels, pred_probs, threshold, comparison=">="
        )
        # Expected: predictions=[1,1,1,1], so TP=2, TN=0, FP=2, FN=0
        assert tp_gte == 2
        assert tn_gte == 0
        assert fp_gte == 2
        assert fn_gte == 0

    def test_comparison_operators_with_sample_weights(self):
        """Test that comparison operators work correctly with sample weights."""
        true_labels = np.array([0, 1, 0, 1])
        pred_probs = np.array([0.3, 0.5, 0.7, 0.5])  # Two at threshold 0.5
        sample_weights = np.array([1.0, 2.0, 1.0, 3.0])  # Different weights
        threshold = 0.5

        # Test with both comparison operators
        tp_gt, tn_gt, fp_gt, fn_gt = confusion_matrix_at_threshold(
            true_labels, pred_probs, threshold, sample_weights, comparison=">"
        )
        tp_gte, tn_gte, fp_gte, fn_gte = confusion_matrix_at_threshold(
            true_labels, pred_probs, threshold, sample_weights, comparison=">="
        )

        # Results should be floats when using sample weights
        assert isinstance(tp_gt, float)
        assert isinstance(tp_gte, float)

        # The weighted results should be different
        assert (tp_gt, tn_gt, fp_gt, fn_gt) != (tp_gte, tn_gte, fp_gte, fn_gte)

        # Sanity check: total weights should be preserved
        total_weight = np.sum(sample_weights)
        assert abs((tp_gt + tn_gt + fp_gt + fn_gt) - total_weight) < 1e-10
        assert abs((tp_gte + tn_gte + fp_gte + fn_gte) - total_weight) < 1e-10
