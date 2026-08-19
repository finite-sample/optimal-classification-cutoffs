"""Tests for score-based workflows and new piecewise features."""

import time

import numpy as np
import pytest

from optimal_cutoffs.metrics import get_metric_function
from optimal_cutoffs.piecewise import (
    _compute_threshold_midpoint,
    optimal_threshold_sortscan,
)
from optimal_cutoffs.validation import validate_binary_classification


class TestScoreBasedWorkflows:
    """Test score-based workflows with require_proba=False."""

    def test_validate_inputs_scores(self):
        """Test validation with score inputs outside [0,1]."""
        y_true = [0, 1, 0, 1]
        scores = [-2.5, 1.3, -0.8, 3.2]  # Logits/scores outside [0,1]

        # Should fail with default require_proba=True
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            validate_binary_classification(y_true, scores)

        # Should work with require_proba=False
        y, s, _ = validate_binary_classification(y_true, scores, require_proba=False)
        assert y.dtype == np.int8
        assert s.dtype == np.float64
        np.testing.assert_array_equal(y, [0, 1, 0, 1])
        np.testing.assert_array_almost_equal(s, [-2.5, 1.3, -0.8, 3.2])

    def test_optimal_threshold_scores(self):
        """Test optimization with score inputs."""
        y_true = [0, 0, 1, 1]
        scores = [-1.5, 0.2, 1.1, 2.8]  # Logits

        result = optimal_threshold_sortscan(y_true, scores, "f1", require_proba=False)
        threshold = result.threshold
        score = result.score
        result.diagnostics.get("k_star", 0) if hasattr(
            result, "diagnostics"
        ) and result.diagnostics else 0

        # Should achieve perfect F1 = 1.0
        assert abs(score - 1.0) < 1e-10
        # Threshold should be between 0.2 and 1.1 (midpoint ~0.65)
        assert 0.2 < threshold < 1.1

    def test_threshold_no_clamping_scores(self):
        """Test that thresholds are not clamped when using scores."""
        y_true = [0, 1]
        scores = [-5.0, 10.0]  # Extreme scores

        result = optimal_threshold_sortscan(y_true, scores, "f1", require_proba=False)
        threshold = result.threshold

        # Threshold should be between -5 and 10, not clamped to [0,1]
        assert -5.0 < threshold < 10.0
        # Specifically should be around the midpoint (2.5)
        assert 2.0 < threshold < 3.0

    def test_threshold_midpoint_scores(self):
        """Test threshold midpoint computation for scores."""
        scores_sorted = np.array([5.0, 2.0, -1.0, -3.0])

        # No clamping should occur for midpoints
        threshold = _compute_threshold_midpoint(scores_sorted, 2, False)
        expected = (2.0 + (-1.0)) / 2.0  # Midpoint = 0.5
        assert abs(threshold - expected) < 1e-10

        # Test with inclusive operator
        threshold_inclusive = _compute_threshold_midpoint(scores_sorted, 2, True)
        # Should still be the same midpoint regardless of inclusive flag
        assert abs(threshold_inclusive - expected) < 1e-10

    def test_edge_cases_with_scores(self):
        """Test edge cases with score inputs."""
        # Single sample with negative score
        y_true = [1]
        scores = [-2.5]

        result = optimal_threshold_sortscan(y_true, scores, "f1", require_proba=False)
        threshold = result.threshold
        score = result.score
        k = (
            result.diagnostics.get("k_star", 0)
            if hasattr(result, "diagnostics") and result.diagnostics
            else 0
        )

        assert abs(threshold - (-2.5)) < 1e-10  # Should be very close to -2.5
        assert score == 1.0  # Perfect F1 for single positive
        # Note: k_star uses 0-based indexing, diagnostic may vary by implementation
        assert k in [
            0,
            1,
        ]  # Should predict the single positive sample, modulo index differences

        # All negatives with positive scores
        y_true = [0, 0, 0]
        scores = [1.1, 2.5, 3.9]

        result = optimal_threshold_sortscan(y_true, scores, "f1", require_proba=False)
        threshold = result.threshold
        score = result.score
        k = (
            result.diagnostics.get("k_star", 0)
            if hasattr(result, "diagnostics") and result.diagnostics
            else 0
        )

        # Should set threshold >= max score to predict all negative
        assert threshold >= 3.9
        assert k == 0

    def test_weighted_scores(self):
        """Test weighted optimization with scores."""
        y_true = [0, 1, 0, 1]
        scores = [-1.0, 0.5, 1.2, 2.8]
        weights = [1.0, 3.0, 1.0, 2.0]  # Weight positive samples more

        result = optimal_threshold_sortscan(
            y_true,
            scores,
            "f1",
            sample_weight=weights,
            require_proba=False,
        )
        threshold = result.threshold
        score = result.score
        result.diagnostics.get("k_star", 0) if hasattr(
            result, "diagnostics"
        ) and result.diagnostics else 0

        # Should be valid threshold and score
        assert -1.0 <= threshold <= 2.8
        assert 0.0 <= score <= 1.0


class TestNewVectorizedMetrics:
    """Test new IoU and specificity metrics."""

    def test_iou_vectorized(self):
        """Test vectorized IoU/Jaccard computation."""
        tp = np.array([2, 0, 1])
        tn = np.array([1, 2, 3])
        fp = np.array([1, 0, 2])
        fn = np.array([0, 1, 0])

        iou_vectorized = get_metric_function("iou")
        iou_scores = iou_vectorized(tp, tn, fp, fn)

        # Case 0: IoU = 2/(2+1+0) = 2/3
        assert abs(iou_scores[0] - 2 / 3) < 1e-10

        # Case 1: IoU = 0/(0+0+1) = 0.0
        assert abs(iou_scores[1] - 0.0) < 1e-10

        # Case 2: IoU = 1/(1+2+0) = 1/3
        assert abs(iou_scores[2] - 1 / 3) < 1e-10

    def test_iou_zero_denominator(self):
        """Test IoU with zero denominator (no positives predicted or actual)."""
        tp = np.array([0])
        tn = np.array([5])
        fp = np.array([0])
        fn = np.array([0])

        iou_vectorized = get_metric_function("iou")
        iou_scores = iou_vectorized(tp, tn, fp, fn)

        # IoU = 0/(0+0+0) = 0.0 (handled by np.where)
        assert abs(iou_scores[0] - -0.0) < 1e-10

    def test_specificity_vectorized(self):
        """Test vectorized specificity computation."""
        tp = np.array([2, 1, 0])
        tn = np.array([3, 0, 4])
        fp = np.array([1, 2, 1])
        fn = np.array([0, 1, 2])

        specificity_vectorized = get_metric_function("specificity")
        spec_scores = specificity_vectorized(tp, tn, fp, fn)

        # Case 0: Specificity = 3/(3+1) = 3/4 = 0.75
        assert abs(spec_scores[0] - 0.75) < 1e-10

        # Case 1: Specificity = 0/(0+2) = 0.0
        assert abs(spec_scores[1] - 0.0) < 1e-10

        # Case 2: Specificity = 4/(4+1) = 4/5 = 0.8
        assert abs(spec_scores[2] - 0.8) < 1e-10

    def test_specificity_zero_denominator(self):
        """Test specificity with zero denominator (no actual negatives)."""
        tp = np.array([2])
        tn = np.array([0])
        fp = np.array([0])
        fn = np.array([1])

        specificity_vectorized = get_metric_function("specificity")
        spec_scores = specificity_vectorized(tp, tn, fp, fn)

        # Specificity = 0/(0+0) = 0.0 (handled by np.where)
        assert abs(spec_scores[0] - 0.0) < 1e-10

    def test_get_vectorized_metric_new_metrics(self):
        """Test that new metrics are available through the registry."""
        iou_func = get_metric_function("iou")
        assert callable(iou_func)

        jaccard_func = get_metric_function("jaccard")
        assert callable(jaccard_func)  # Should be alias

        spec_func = get_metric_function("specificity")
        assert callable(spec_func)

    def test_optimization_with_new_metrics(self):
        """Test optimization using new metrics."""
        y_true = [0, 0, 1, 1]
        pred_prob = [0.1, 0.3, 0.7, 0.9]

        # Test IoU optimization
        result_iou = optimal_threshold_sortscan(y_true, pred_prob, "iou")
        threshold_iou = result_iou.threshold
        score_iou = result_iou.score

        # Test specificity optimization
        result_spec = optimal_threshold_sortscan(y_true, pred_prob, "specificity")
        threshold_spec = result_spec.threshold
        score_spec = result_spec.score

        # Should produce valid results
        assert 0.0 <= threshold_iou <= 1.0
        assert 0.0 <= threshold_spec <= 1.0
        assert 0.0 <= score_iou <= 1.0
        assert 0.0 <= score_spec <= 1.0


class TestTieHandlingImprovements:
    """Test improved tie handling with local nudges."""

    def test_tied_probabilities_deterministic(self):
        """Test that tied probabilities produce deterministic results."""
        y_true = [0, 1, 0, 1, 0, 1]
        pred_prob = [0.3, 0.5, 0.5, 0.5, 0.5, 0.7]  # Many ties at 0.5

        # Run multiple times to check determinism
        results = []
        for _ in range(5):
            result = optimal_threshold_sortscan(y_true, pred_prob, "f1")
            threshold = result.threshold
            score = result.score
            k = (
                result.diagnostics.get("k_star", 0)
                if hasattr(result, "diagnostics") and result.diagnostics
                else 0
            )
            results.append((threshold, score, k))

        # All results should be identical
        for i in range(1, len(results)):
            assert abs(results[i][0] - results[0][0]) < 1e-12
            assert abs(results[i][1] - results[0][1]) < 1e-12
            assert results[i][2] == results[0][2]

    def test_local_nudge_effectiveness(self):
        """Test that local nudging finds better solutions for ties."""
        # Create a case where local nudging should help
        y_true = [0, 0, 1, 1, 1]
        pred_prob = [0.2, 0.5, 0.5, 0.5, 0.8]  # Ties at decision boundary

        result = optimal_threshold_sortscan(y_true, pred_prob, "f1")
        threshold = result.threshold
        score = result.score
        result.diagnostics.get("k_star", 0) if hasattr(
            result, "diagnostics"
        ) and result.diagnostics else 0

        # Should achieve reasonable performance
        assert score > 0.5  # Better than random
        assert 0.0 <= threshold <= 1.0

    def test_performance_with_many_ties(self):
        """Test that performance is good even with many tied values."""
        n = 1000
        y_true = np.random.randint(0, 2, n)
        # Create many ties by using only a few unique values
        pred_prob = np.random.choice([0.1, 0.3, 0.5, 0.7, 0.9], size=n)

        start_time = time.time()
        result = optimal_threshold_sortscan(y_true, pred_prob, "f1")
        threshold = result.threshold
        score = result.score
        result.diagnostics.get("k_star", 0) if hasattr(
            result, "diagnostics"
        ) and result.diagnostics else 0
        end_time = time.time()

        duration = end_time - start_time

        # Should still be fast despite many ties
        assert duration < 0.5  # Less than 500ms
        assert 0.0 <= threshold <= 1.0
        assert 0.0 <= score <= 1.0
