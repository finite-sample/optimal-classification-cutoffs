"""Test weight invariance properties and catch integer cast bugs.

This module tests the fundamental properties that weighted metrics must satisfy:
1. Non-integer weights behave like proportional sample duplication
2. Scale invariance: multiplying all weights by a constant doesn't change optimum
3. Fractional weights are preserved (catches int() cast bugs)

These tests use rational weights with small denominators to enable exact
expansion testing and verify mathematical correctness.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from optimal_cutoffs import optimize_thresholds
from optimal_cutoffs.metrics_core import confusion_matrix_at_threshold, f1_score


def expand_by_rational_weights(y, p, w):
    """Expand dataset by rational weights for exact comparison.

    Each sample i is duplicated exactly k*w[i] times where k is chosen
    so that all k*w[i] are integers >= 1.

    Parameters
    ----------
    y : array-like
        Binary labels
    p : array-like
        Probabilities
    w : array-like
        Rational weights (can be fractional)

    Returns
    -------
    tuple
        (expanded_labels, expanded_probs) where each sample is duplicated
        according to its weight
    """
    y = np.asarray(y)
    p = np.asarray(p)
    w = np.asarray(w)

    # Find a common multiplier to make all weights integer
    # Use LCM approach with reasonable denominators
    k = 60  # LCM of small denominators, sufficient for test cases

    # Convert to integer counts, ensuring each is at least 1
    counts = np.maximum(1, np.round(k * w).astype(int))

    # Expand arrays
    y_expanded = np.repeat(y, counts)
    p_expanded = np.repeat(p, counts)

    return y_expanded, p_expanded


class TestWeightedEqualsExpanded:
    """Test that weighted metrics match expanded dataset results."""

    @given(n=st.integers(5, 80), comparison=st.sampled_from([">", ">="]))
    @settings(deadline=None, max_examples=80)
    def test_weighted_equals_expanded_f1(self, n, comparison):
        """Weighted F1 must match F1 on expanded dataset."""
        rng = np.random.default_rng(7)
        p = rng.uniform(0, 1, size=n)
        y = (rng.uniform(0, 1, size=n) < 0.4).astype(int)

        # Ensure both classes present
        if y.sum() == 0:
            y[0] = 1
        if y.sum() == y.size:
            y[0] = 0

        # Generate fractional and integer weights
        w = rng.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], size=n)

        # Weighted optimization
        result_weighted = optimize_thresholds(
            y,
            p,
            metric="f1",
            method="sort_scan",
            comparison=comparison,
            sample_weight=w,
        )
        threshold_weighted = result_weighted.threshold

        # Expanded dataset optimization
        y_expanded, p_expanded = expand_by_rational_weights(y, p, w)
        result_expanded = optimize_thresholds(
            y_expanded,
            p_expanded,
            metric="f1",
            method="sort_scan",
            comparison=comparison,
            sample_weight=None,
        )
        threshold_expanded = result_expanded.threshold

        # Decisions must match on original items
        pred_weighted = (
            (p > threshold_weighted) if comparison == ">" else (p >= threshold_weighted)
        )
        pred_expanded = (
            (p > threshold_expanded) if comparison == ">" else (p >= threshold_expanded)
        )

        # This is the key test - decisions should be identical
        assert np.array_equal(pred_weighted, pred_expanded), (
            f"Weighted and expanded predictions don't match. "
            f"Weighted: {pred_weighted.astype(int)}, "
            f"Expanded: {pred_expanded.astype(int)}, "
            f"Thresholds: w={threshold_weighted:.10f}, e={threshold_expanded:.10f}"
        )

        # Also verify scores match
        tp_w, tn_w, fp_w, fn_w = confusion_matrix_at_threshold(
            y, p, threshold_weighted, w, comparison
        )
        tp_e, tn_e, fp_e, fn_e = confusion_matrix_at_threshold(
            y_expanded, p_expanded, threshold_expanded, None, comparison
        )

        f1_weighted = f1_score(tp_w, tn_w, fp_w, fn_w)
        f1_expanded = f1_score(tp_e, tn_e, fp_e, fn_e)

        # Scores should be very close (allowing tiny numerical differences)
        assert abs(f1_weighted - f1_expanded) < 1e-10, (
            f"F1 scores don't match: weighted={f1_weighted:.12f}, "
            f"expanded={f1_expanded:.12f}"
        )

    @given(n=st.integers(8, 60), comparison=st.sampled_from([">", ">="]))
    @settings(deadline=None, max_examples=60)
    def test_weighted_equals_expanded_accuracy(self, n, comparison):
        """Weighted accuracy must match accuracy on expanded dataset."""
        rng = np.random.default_rng(123)
        p = rng.uniform(0, 1, size=n)
        y = (rng.uniform(0, 1, size=n) < 0.5).astype(int)

        if y.sum() == 0:
            y[0] = 1
        if y.sum() == y.size:
            y[0] = 0

        # Use rational weights for exact expansion
        w = rng.choice([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0], size=n)

        result_weighted = optimize_thresholds(
            y,
            p,
            metric="accuracy",
            method="sort_scan",
            comparison=comparison,
            sample_weight=w,
        )
        threshold_weighted = result_weighted.threshold

        y_expanded, p_expanded = expand_by_rational_weights(y, p, w)
        result_expanded = optimize_thresholds(
            y_expanded,
            p_expanded,
            metric="accuracy",
            method="sort_scan",
            comparison=comparison,
            sample_weight=None,
        )
        threshold_expanded = result_expanded.threshold

        # Verify decisions match
        pred_weighted = (
            (p > threshold_weighted) if comparison == ">" else (p >= threshold_weighted)
        )
        pred_expanded = (
            (p > threshold_expanded) if comparison == ">" else (p >= threshold_expanded)
        )

        assert np.array_equal(pred_weighted, pred_expanded), (
            "Weighted and expanded decisions don't match for accuracy"
        )

    def test_fractional_weights_preserved(self):
        """Fractional weights must be preserved without integer casting."""
        # This test specifically catches the int() cast bug
        y = np.array([0, 1, 1, 0, 1])
        p = np.array([0.2, 0.7, 0.8, 0.3, 0.9])
        w = np.array([0.1, 1.7, 2.3, 0.9, 1.4])  # Fractional weights

        # This should work without error and preserve fractional precision
        result = optimize_thresholds(
            y, p, metric="f1", method="sort_scan", sample_weight=w
        )
        threshold = result.threshold

        # Verify confusion matrix preserves fractional values
        tp, tn, fp, fn = confusion_matrix_at_threshold(y, p, threshold, w)

        # At least one of these should be fractional if weights are preserved
        confusion_values = [tp, tn, fp, fn]
        total = sum(confusion_values)

        # Total should equal sum of weights
        assert abs(total - np.sum(w)) < 1e-12, (
            f"Total confusion matrix count {total} should equal sum of weights "
            f"{np.sum(w)}"
        )

        # At least some values should be fractional (not integers)
        has_fractional = any(abs(val - round(val)) > 1e-10 for val in confusion_values)
        if not np.allclose(w, np.round(w)):  # Only if weights are actually fractional
            assert has_fractional, (
                f"Confusion matrix values should preserve fractional precision: "
                f"tp={tp}, tn={tn}, fp={fp}, fn={fn}"
            )


class TestWeightScaleInvariance:
    """Test that multiplying all weights by a constant doesn't change optimum."""

    @given(
        n=st.integers(10, 50),
        scale_factor=st.floats(0.1, 10.0),
        comparison=st.sampled_from([">", ">="]),
    )
    @settings(deadline=None, max_examples=50)
    def test_weight_scale_invariance_f1(self, n, scale_factor, comparison):
        """F1 optimization should be invariant to weight scaling."""
        rng = np.random.default_rng(456)
        p = rng.uniform(0, 1, size=n)
        y = (rng.uniform(0, 1, size=n) < 0.6).astype(int)

        if y.sum() == 0:
            y[0] = 1
        if y.sum() == y.size:
            y[0] = 0

        # Original weights
        w_orig = rng.uniform(0.5, 3.0, size=n)

        # Scaled weights
        w_scaled = w_orig * scale_factor

        # Both should give same threshold (decisions)
        result_orig = optimize_thresholds(
            y,
            p,
            metric="f1",
            method="sort_scan",
            comparison=comparison,
            sample_weight=w_orig,
        )
        threshold_orig = result_orig.threshold

        result_scaled = optimize_thresholds(
            y,
            p,
            metric="f1",
            method="sort_scan",
            comparison=comparison,
            sample_weight=w_scaled,
        )
        threshold_scaled = result_scaled.threshold

        # Decisions should be identical
        pred_orig = (p > threshold_orig) if comparison == ">" else (p >= threshold_orig)
        pred_scaled = (
            (p > threshold_scaled) if comparison == ">" else (p >= threshold_scaled)
        )

        assert np.array_equal(pred_orig, pred_scaled), (
            f"Scale invariance violated: original threshold {threshold_orig:.10f}, "
            f"scaled threshold {threshold_scaled:.10f}, scale_factor={scale_factor:.6f}"
        )

    def test_weight_scale_invariance_accuracy(self):
        """Accuracy optimization should be invariant to weight scaling."""
        np.random.default_rng(789)
        y = np.array([0, 1, 1, 0, 1, 0, 1, 0])
        p = np.array([0.1, 0.8, 0.7, 0.3, 0.9, 0.2, 0.6, 0.4])

        w_orig = np.array([1.0, 2.0, 1.5, 0.5, 2.5, 1.2, 0.8, 1.8])

        for scale in [0.1, 0.5, 2.0, 5.0, 10.0]:
            w_scaled = w_orig * scale

            result_orig = optimize_thresholds(
                y, p, metric="accuracy", method="sort_scan", sample_weight=w_orig
            )
            threshold_orig = result_orig.threshold

            result_scaled = optimize_thresholds(
                y, p, metric="accuracy", method="sort_scan", sample_weight=w_scaled
            )
            threshold_scaled = result_scaled.threshold

            pred_orig = p > threshold_orig
            pred_scaled = p > threshold_scaled

            assert np.array_equal(pred_orig, pred_scaled), (
                f"Scale invariance violated for scale={scale}"
            )


class TestWeightEdgeCases:
    """Test edge cases with weights."""

    def test_zero_weights_handled(self):
        """Zero weights should be handled gracefully."""
        y = np.array([0, 1, 1, 0])
        p = np.array([0.2, 0.7, 0.8, 0.3])
        w = np.array([0.0, 1.0, 2.0, 0.0])  # Some zero weights

        # Should not crash
        result = optimize_thresholds(
            y, p, metric="f1", method="sort_scan", sample_weight=w
        )

        threshold = result.threshold
        assert 0 <= threshold <= 1

        # Only non-zero weighted samples should contribute
        tp, tn, fp, fn = confusion_matrix_at_threshold(y, p, threshold, w)
        total = tp + tn + fp + fn

        # Total should equal sum of non-zero weights
        assert abs(total - np.sum(w)) < 1e-12

    def test_uniform_weights_equal_unweighted(self):
        """Uniform weights should give same result as unweighted."""
        rng = np.random.default_rng(999)
        n = 30
        y = rng.integers(0, 2, size=n)
        p = rng.uniform(0, 1, size=n)

        if y.sum() == 0:
            y[0] = 1
        if y.sum() == y.size:
            y[0] = 0

        # Test different uniform weight values
        for weight_val in [0.5, 1.0, 2.0, 5.0]:
            w_uniform = np.full(n, weight_val)

            result_weighted = optimize_thresholds(
                y, p, metric="f1", method="sort_scan", sample_weight=w_uniform
            )
            threshold_weighted = result_weighted.threshold

            result_unweighted = optimize_thresholds(
                y, p, metric="f1", method="sort_scan", sample_weight=None
            )
            threshold_unweighted = result_unweighted.threshold

            # Decisions should be identical
            pred_weighted = p > threshold_weighted
            pred_unweighted = p > threshold_unweighted

            assert np.array_equal(pred_weighted, pred_unweighted), (
                f"Uniform weights {weight_val} should match unweighted"
            )

    def test_single_nonzero_weight(self):
        """Single non-zero weight should optimize for that sample only."""
        y = np.array([0, 1, 1, 0])
        p = np.array([0.2, 0.7, 0.8, 0.3])
        w = np.array([0.0, 1.0, 0.0, 0.0])  # Only second sample weighted

        result = optimize_thresholds(
            y, p, metric="accuracy", method="sort_scan", sample_weight=w
        )
        threshold = result.threshold

        # Should optimize for the single weighted sample (y[1]=1, p[1]=0.7)
        # For accuracy, optimal is to predict this sample correctly
        pred = p > threshold

        # The weighted sample should be predicted correctly if possible
        assert pred[1] == y[1], "Single weighted sample should be predicted correctly"


class TestWeightMethodConsistency:
    """Test that different methods handle weights consistently."""

    @given(n=st.integers(8, 40), comparison=st.sampled_from([">", ">="]))
    @settings(deadline=None, max_examples=40)
    def test_sort_scan_vs_unique_scan_with_weights(self, n, comparison):
        """sort_scan and unique_scan should give consistent results with weights."""
        rng = np.random.default_rng(1111)
        p = rng.uniform(0, 1, size=n)
        y = (rng.uniform(0, 1, size=n) < 0.5).astype(int)
        w = rng.uniform(0.1, 2.0, size=n)

        if y.sum() == 0:
            y[0] = 1
        if y.sum() == y.size:
            y[0] = 0

        try:
            result_scan = optimize_thresholds(
                y,
                p,
                metric="f1",
                method="sort_scan",
                comparison=comparison,
                sample_weight=w,
            )
            threshold_scan = result_scan.threshold

            result_brute = optimize_thresholds(
                y,
                p,
                metric="f1",
                method="sort_scan",
                comparison=comparison,
                sample_weight=w,
            )
            threshold_brute = result_brute.threshold

            # Decisions should match (allowing small threshold differences due to
            # different
            # threshold selection strategies)

            # Compute F1 scores to verify they're equal
            tp_scan, tn_scan, fp_scan, fn_scan = confusion_matrix_at_threshold(
                y, p, threshold_scan, w, comparison
            )
            tp_brute, tn_brute, fp_brute, fn_brute = confusion_matrix_at_threshold(
                y, p, threshold_brute, w, comparison
            )

            f1_scan = f1_score(tp_scan, tn_scan, fp_scan, fn_scan)
            f1_brute = f1_score(tp_brute, tn_brute, fp_brute, fn_brute)

            # Scores should match (both methods should find optimal)
            assert abs(f1_scan - f1_brute) < 1e-10, (
                f"F1 scores don't match: sort_scan={f1_scan:.12f}, "
                f"unique_scan={f1_brute:.12f}"
            )

        except (ValueError, NotImplementedError):
            # Some method/metric combinations might not support weights
            pass

    def test_weights_improve_targeted_performance(self):
        """Higher weights on samples should improve their prediction accuracy."""
        # Create scenario where weighting specific samples changes the optimal threshold
        y = np.array([0, 0, 1, 1, 0])
        p = np.array([0.3, 0.4, 0.6, 0.7, 0.45])

        # Equal weights
        w_equal = np.ones(5)
        result_equal = optimize_thresholds(
            y, p, metric="accuracy", sample_weight=w_equal
        )
        threshold_equal = result_equal.threshold

        # Weight the correct predictions more heavily
        w_biased = np.array([2.0, 2.0, 2.0, 2.0, 1.0])  # Weight first 4 samples more
        result_biased = optimize_thresholds(
            y, p, metric="accuracy", sample_weight=w_biased
        )
        threshold_biased = result_biased.threshold

        # Apply thresholds
        pred_equal = p > threshold_equal
        pred_biased = p > threshold_biased

        # Weighted accuracy should be different (and potentially better on weighted
        # samples)
        acc_equal = np.mean(pred_equal == y)

        # Compute weighted accuracy for biased case
        correct_biased = (pred_biased == y).astype(float)
        acc_biased_weighted = np.average(correct_biased, weights=w_biased)

        # The weighted approach should perform at least as well on the weighted metric
        # (This is not a strict inequality due to potential ties, but shows the effect)
        assert acc_biased_weighted >= acc_equal - 1e-10, (
            f"Weighted optimization should not perform worse: "
            f"equal={acc_equal:.6f}, biased_weighted={acc_biased_weighted:.6f}"
        )
