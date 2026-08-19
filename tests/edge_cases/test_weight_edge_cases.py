"""Tests for sample weight functionality."""

import numpy as np
import pytest

from optimal_cutoffs import optimize_thresholds
from optimal_cutoffs.cv import cross_validate, nested_cross_validate
from optimal_cutoffs.metrics_core import (
    confusion_matrix_at_threshold,
    needs_probability_scores,
    register_metric,
    should_maximize_metric,
)


def multiclass_confusion_matrices_at_thresholds(
    y_true, y_prob, thresholds, sample_weight=None
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
            y_true_binary, y_prob_class, threshold, sample_weight
        )
        cms.append((tp, tn, fp, fn))

    return cms


def test_confusion_matrix_with_sample_weights():
    """Test weighted confusion matrix calculation."""
    true_labs = np.array([0, 1, 1, 0, 1])
    pred_prob = np.array([0.2, 0.8, 0.7, 0.3, 0.4])
    threshold = 0.5
    sample_weight = np.array([1.0, 2.0, 1.5, 0.5, 2.0])  # Different weights

    # Test without weights
    tp, tn, fp, fn = confusion_matrix_at_threshold(true_labs, pred_prob, threshold)
    assert (tp, tn, fp, fn) == (2, 2, 0, 1)

    # Test with weights
    tp_w, tn_w, fp_w, fn_w = confusion_matrix_at_threshold(
        true_labs, pred_prob, threshold, sample_weight
    )

    # Manual calculation:
    # Sample 0: true=0, pred=0, weight=1.0 -> TN
    # Sample 1: true=1, pred=1, weight=2.0 -> TP
    # Sample 2: true=1, pred=1, weight=1.5 -> TP
    # Sample 3: true=0, pred=0, weight=0.5 -> TN
    # Sample 4: true=1, pred=0, weight=2.0 -> FN
    expected_tp = 2.0 + 1.5  # = 3.5
    expected_tn = 1.0 + 0.5  # = 1.5
    expected_fp = 0.0
    expected_fn = 2.0

    assert tp_w == pytest.approx(expected_tp)
    assert tn_w == pytest.approx(expected_tn)
    assert fp_w == pytest.approx(expected_fp)
    assert fn_w == pytest.approx(expected_fn)

    # Ensure they are float values when using sample weights
    assert isinstance(tp_w, float)
    assert isinstance(tn_w, float)
    assert isinstance(fp_w, float)
    assert isinstance(fn_w, float)


def test_multiclass_confusion_matrix_with_sample_weights():
    """Test weighted multiclass confusion matrix calculation."""
    true_labs = np.array([0, 1, 2, 0, 1, 2])
    pred_prob = np.array(
        [
            [0.8, 0.1, 0.1],  # True: 0
            [0.2, 0.7, 0.1],  # True: 1
            [0.1, 0.2, 0.7],  # True: 2
            [0.6, 0.3, 0.1],  # True: 0
            [0.3, 0.6, 0.1],  # True: 1
            [0.2, 0.2, 0.6],  # True: 2
        ]
    )
    thresholds = np.array([0.5, 0.5, 0.5])
    sample_weight = np.array([1.0, 2.0, 1.5, 0.5, 2.0, 1.0])

    cms = multiclass_confusion_matrices_at_thresholds(
        true_labs, pred_prob, thresholds, sample_weight
    )

    assert len(cms) == 3
    for cm in cms:
        assert len(cm) == 4
        # Check that we got weighted values (should be floats when using weights)
        assert all(isinstance(x, float) for x in cm)


def test_optimal_threshold_with_sample_weights():
    """Test threshold optimization with sample weights."""
    np.random.seed(42)
    true_labs = np.array([0, 0, 0, 1, 1, 1])
    pred_prob = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])

    # Equal weights should give same result as no weights
    equal_weights = np.ones(len(true_labs))

    result_no_weights = optimize_thresholds(true_labs, pred_prob, metric="f1")
    result_equal_weights = optimize_thresholds(
        true_labs, pred_prob, metric="f1", sample_weight=equal_weights
    )
    threshold_no_weights = result_no_weights.threshold
    threshold_equal_weights = result_equal_weights.threshold

    assert np.isclose(threshold_no_weights, threshold_equal_weights, rtol=1e-10)

    # Different weights should potentially give different result
    # Weight the positive class more heavily
    heavy_positive_weights = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])
    result_weighted = optimize_thresholds(
        true_labs, pred_prob, metric="f1", sample_weight=heavy_positive_weights
    )
    threshold_weighted = result_weighted.threshold

    # Should be a valid threshold
    assert 0 <= threshold_weighted <= 1


def test_multiclass_optimal_thresholds_with_sample_weights():
    """Test multiclass threshold optimization with sample weights."""
    np.random.seed(42)
    true_labs = np.array([0, 1, 2, 0, 1, 2])
    pred_prob = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.6, 0.3, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )

    sample_weight = np.array([1.0, 2.0, 1.5, 0.5, 2.0, 1.0])

    result = optimize_thresholds(
        true_labs, pred_prob, metric="f1", sample_weight=sample_weight
    )
    thresholds = result.thresholds

    assert len(thresholds) == 3
    # Note: With coordinate ascent optimization, thresholds can be outside [0,1]
    # This is mathematically correct for margin-based decision rules
    assert all(np.isfinite(t) for t in thresholds), "Thresholds should be finite"


def test_direct_api_with_sample_weights():
    """Test direct API with sample weights."""
    true_labs = np.array([0, 0, 1, 1, 1])
    pred_prob = np.array([0.2, 0.3, 0.6, 0.7, 0.8])
    sample_weight = np.array([1.0, 1.0, 2.0, 2.0, 2.0])  # Weight positive class more

    # Use direct API instead of removed wrapper
    result = optimize_thresholds(
        true_labs, pred_prob, metric="f1", sample_weight=sample_weight
    )

    threshold = result.threshold
    assert isinstance(threshold, float | np.number) or (
        isinstance(threshold, np.ndarray) and threshold.size == 1
    )
    assert 0 <= threshold <= 1


def test_cv_with_sample_weights():
    """Test cross-validation with sample weights."""
    np.random.seed(42)
    true_labs = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    pred_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    sample_weight = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])

    thresholds, scores = cross_validate(
        true_labs, pred_prob, metric="f1", cv=3, sample_weight=sample_weight
    )

    assert len(thresholds) == 3
    assert len(scores) == 3
    assert all(0 <= t <= 1 for t in thresholds)
    assert all(0 <= s <= 1 for s in scores)


def test_nested_cv_with_sample_weights():
    """Test nested cross-validation with sample weights."""
    np.random.seed(42)
    true_labs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    pred_prob = np.array([0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.7, 0.75, 0.8, 0.9])
    sample_weight = np.array([1.0] * 5 + [2.0] * 5)

    thresholds, scores = nested_cross_validate(
        true_labs,
        pred_prob,
        metric="f1",
        inner_cv=3,
        outer_cv=2,
        sample_weight=sample_weight,
    )

    assert len(thresholds) == 2
    assert len(scores) == 2
    assert all(0 <= t <= 1 for t in thresholds)
    assert all(0 <= s <= 1 for s in scores)


def test_metric_registry_metadata():
    """Test new metric registry metadata functionality."""
    # Test built-in metrics have correct metadata
    assert should_maximize_metric("f1") is True
    assert should_maximize_metric("accuracy") is True
    assert should_maximize_metric("precision") is True
    assert should_maximize_metric("recall") is True

    assert needs_probability_scores("f1") is False
    assert needs_probability_scores("accuracy") is False
    assert needs_probability_scores("precision") is False
    assert needs_probability_scores("recall") is False

    # Test unknown metric defaults
    assert should_maximize_metric("unknown_metric") is True  # Default
    assert needs_probability_scores("unknown_metric") is False  # Default

    # Test registering a metric that should be minimized
    @register_metric("mse", maximize=False, needs_proba=True)
    def mean_squared_error(tp, tn, fp, fn):
        """Dummy MSE metric for testing."""
        total = tp + tn + fp + fn
        errors = fp + fn  # Simplified error count
        return errors / total if total > 0 else 0.0

    assert should_maximize_metric("mse") is False
    assert needs_probability_scores("mse") is True


def test_sample_weight_validation():
    """Test sample weight validation."""
    true_labs = np.array([0, 1, 1, 0])
    pred_prob = np.array([0.2, 0.8, 0.7, 0.3])

    # Wrong length should raise error
    wrong_length_weights = np.array([1.0, 2.0])  # Length 2 vs 4 samples

    with pytest.raises(ValueError, match="Length mismatch"):
        confusion_matrix_at_threshold(true_labs, pred_prob, 0.5, wrong_length_weights)


def test_sample_weights_piecewise_optimization():
    """Test that piecewise optimization works with sample weights."""
    np.random.seed(42)
    true_labs = np.array([0, 0, 1, 1, 1])
    pred_prob = np.array([0.2, 0.3, 0.6, 0.7, 0.8])
    sample_weight = np.array([1.0, 1.0, 2.0, 2.0, 2.0])

    # F1 is piecewise, so should use the fast algorithm
    result = optimize_thresholds(
        true_labs,
        pred_prob,
        metric="f1",
        method="unique_scan",
        sample_weight=sample_weight,
    )

    threshold = result.threshold
    assert 0 <= threshold <= 1

    # Verify it's actually using the piecewise optimization by checking it gives a
    # reasonable result
    # The threshold should be reasonable for the probability distribution
    # With enhanced piecewise optimization, thresholds can be midpoints or edge values
    unique_probs = np.unique(pred_prob)
    candidate_values = list(unique_probs)

    # Add midpoints between adjacent unique values as valid candidates
    sorted_probs = np.sort(unique_probs)
    for i in range(len(sorted_probs) - 1):
        midpoint = 0.5 * (sorted_probs[i] + sorted_probs[i + 1])
        candidate_values.append(midpoint)

    # Also add edge cases (slightly above max, slightly below min)
    if len(sorted_probs) > 0:
        candidate_values.append(min(1.0, sorted_probs[-1] + 0.01))  # Above max
        candidate_values.append(max(0.0, sorted_probs[0] - 0.01))  # Below min

    min_distance = min(abs(threshold - p) for p in candidate_values)
    assert min_distance < 0.02, (
        f"Threshold {threshold} not close to any candidate value. Candidates: "
        f"{sorted(set(candidate_values))}"
    )


if __name__ == "__main__":
    pytest.main([__file__])
