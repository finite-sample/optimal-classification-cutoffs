"""Test Dinkelbach method mathematical properties for expected F-beta optimization.

The Dinkelbach method optimizes the EXPECTED F-beta score under the assumption
of perfect calibration, meaning it depends only on predicted probabilities (p),
not on realized labels (y). This module tests these fundamental properties:

1. Label independence: threshold depends only on p, not y
2. Calibration behavior: reasonable performance on calibrated data
3. Tie handling follows comparison operator semantics
4. Expected vs empirical F-beta relationship
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from optimal_cutoffs import optimize_thresholds
from optimal_cutoffs.metrics_core import confusion_matrix_at_threshold, f1_score
from tests.fixtures.hypothesis_strategies import beta_bernoulli_calibrated


class TestDinkelbachLabelIndependence:
    """Test that Dinkelbach depends only on probabilities, not labels."""

    @given(n=st.integers(20, 200))
    @settings(deadline=None, max_examples=50)
    def test_dinkelbach_independent_of_labels(self, n):
        """Dinkelbach threshold is identical for one probability set, any labels."""
        rng = np.random.default_rng(99)
        probs = rng.uniform(0, 1, size=n)

        # Generate different label sets with same probabilities
        labels_calibrated = (rng.uniform(0, 1, size=n) < probs).astype(
            int
        )  # Labels match probs
        labels_random = rng.integers(0, 2, size=n)  # Random unrelated labels
        labels_anticorrelated = 1 - labels_calibrated  # Opposite of calibrated

        for comparison in [">", ">="]:
            # Get thresholds for all three label sets
            result_calibrated = optimize_thresholds(
                labels_calibrated,
                probs,
                metric="f1",
                mode="expected",
                comparison=comparison,
            )
            result_random = optimize_thresholds(
                labels_random,
                probs,
                metric="f1",
                mode="expected",
                comparison=comparison,
            )
            result_anticorrelated = optimize_thresholds(
                labels_anticorrelated,
                probs,
                metric="f1",
                mode="expected",
                comparison=comparison,
            )

            # Extract thresholds
            threshold_calibrated = result_calibrated.threshold
            threshold_random = result_random.threshold
            threshold_anticorrelated = result_anticorrelated.threshold

            # All thresholds should be identical (Dinkelbach ignores labels)
            assert abs(threshold_calibrated - threshold_random) < 1e-10, (
                f"Calibrated vs random labels gave different thresholds: "
                f"{threshold_calibrated:.10f} vs {threshold_random:.10f}"
            )
            assert abs(threshold_calibrated - threshold_anticorrelated) < 1e-10, (
                f"Calibrated vs anticorrelated labels gave different thresholds: "
                f"{threshold_calibrated:.10f} vs {threshold_anticorrelated:.10f}"
            )
            assert abs(threshold_random - threshold_anticorrelated) < 1e-10, (
                f"Random vs anticorrelated labels gave different thresholds: "
                f"{threshold_random:.10f} vs {threshold_anticorrelated:.10f}"
            )

    def test_dinkelbach_depends_only_on_probabilities(self):
        """Explicit test that changing labels doesn't affect Dinkelbach threshold."""
        probs = np.array([0.2, 0.4, 0.6, 0.8, 0.3, 0.7])

        # Test different label patterns
        label_patterns = [
            np.array([0, 0, 1, 1, 0, 1]),  # Mixed
            np.array([1, 1, 0, 0, 1, 0]),  # Opposite pattern
            np.array([0, 1, 0, 1, 0, 1]),  # Alternating
            np.array([1, 1, 1, 0, 0, 0]),  # First half positive
            np.array([0, 0, 0, 0, 0, 0]),  # All negative (edge case)
            np.array([1, 1, 1, 1, 1, 1]),  # All positive (edge case)
        ]

        thresholds = []
        for labels in label_patterns:
            result = optimize_thresholds(
                labels, probs, mode="expected", metric="f1", comparison=">"
            )
            thresholds.append(result.threshold)

        # All thresholds should be identical
        for i in range(1, len(thresholds)):
            assert abs(thresholds[i] - thresholds[0]) < 1e-10, (
                f"Pattern {i} gave different threshold: {thresholds[i]} vs "
                f"{thresholds[0]}"
            )

    def test_dinkelbach_sum_probabilities_not_labels(self):
        """Verify Dinkelbach uses sum(p) not sum(y) in expected F-beta calculation."""
        # Create case where sum(p) != sum(y) due to miscalibration
        probs = np.array([0.1, 0.9, 0.1, 0.9, 0.5])  # sum = 2.5
        labels = np.array([1, 0, 1, 0, 0])  # sum = 2 (different from probs)

        # Two datasets with same probs but different labels
        labels2 = np.array([0, 1, 0, 1, 1])  # sum = 3 (also different from probs)

        result1 = optimize_thresholds(
            labels, probs, mode="expected", metric="f1", comparison=">="
        )
        result2 = optimize_thresholds(
            labels2, probs, mode="expected", metric="f1", comparison=">="
        )

        # Extract thresholds from results
        threshold1 = result1.threshold
        threshold2 = result2.threshold
        # Should be identical despite different label sums
        assert abs(threshold1 - threshold2) < 1e-10, (
            f"Different label sums should not affect Dinkelbach: {threshold1} vs "
            f"{threshold2}"
        )

        # Neither should equal a threshold based on label statistics
        label_based_fraction = np.sum(labels) / len(labels)  # 0.4
        assert abs(threshold1 - label_based_fraction) > 1e-6, (
            f"Threshold {threshold1} suspiciously close to label fraction "
            f"{label_based_fraction}"
        )


class TestDinkelbachCalibratedPerformance:
    """Test Dinkelbach performance on calibrated data."""

    @given(beta_bernoulli_calibrated(min_size=100, max_size=400))
    @settings(deadline=None, max_examples=25)
    def test_calibrated_data_reasonable_performance(self, calibrated_data):
        """Dinkelbach should give reasonable F1 on calibrated Beta-Bernoulli data."""
        labels, probs, alpha, beta_param = calibrated_data

        # Skip degenerate cases
        if np.sum(labels) <= 1 or np.sum(labels) >= len(labels) - 1:
            return

        for comparison in [">", ">="]:
            result = optimize_thresholds(
                labels, probs, mode="expected", metric="f1", comparison=comparison
            )

            # Extract threshold from tuple
            threshold, expected_f1 = result.threshold, result.score

            # Should produce reasonable threshold
            threshold = result.threshold
            assert 0 <= threshold <= 1, f"Threshold {threshold} out of bounds"
            assert 0 <= expected_f1 <= 1, f"Expected F1 {expected_f1} out of bounds"

            # Compute empirical F1 with this threshold
            tp, tn, fp, fn = confusion_matrix_at_threshold(
                labels, probs, threshold, comparison=comparison
            )
            empirical_f1 = f1_score(tp, tn, fp, fn)

            # For calibrated data, F1 should be reasonable (not random performance)
            # Exact bound depends on data characteristics, but should be > 0.1 for
            # non-degenerate cases
            assert 0 <= empirical_f1 <= 1, f"F1 {empirical_f1} out of valid range"

            if len(labels) > 50 and 0.1 < np.mean(labels) < 0.9:
                # For larger, non-extreme datasets, expect decent performance
                assert empirical_f1 > 0.05, (
                    f"F1 {empirical_f1:.4f} too low for calibrated data "
                    f"(alpha={alpha:.2f}, beta={beta_param:.2f}, n={len(labels)})"
                )

    def test_dinkelbach_vs_other_methods_on_calibrated_data(self):
        """Compare Dinkelbach to other methods on well-calibrated data."""
        # Generate well-calibrated data
        rng = np.random.default_rng(2025)
        n = 200
        probs = rng.beta(2, 2, size=n)  # Symmetric beta
        labels = rng.binomial(1, probs)

        # Skip if degenerate
        if np.sum(labels) <= 1 or np.sum(labels) >= n - 1:
            pytest.skip("Degenerate calibrated data")

        # Get thresholds from different methods
        result_dinkelbach = optimize_thresholds(
            labels, probs, mode="expected", metric="f1", comparison=">"
        )
        threshold_dinkelbach = result_dinkelbach.threshold
        result_sort_scan = optimize_thresholds(
            labels, probs, metric="f1", method="sort_scan", comparison=">"
        )
        threshold_sort_scan = result_sort_scan.threshold

        # Compute F1 scores
        f1_dinkelbach = f1_score(
            *confusion_matrix_at_threshold(
                labels, probs, threshold_dinkelbach, comparison=">"
            )
        )
        f1_sort_scan = f1_score(
            *confusion_matrix_at_threshold(
                labels, probs, threshold_sort_scan, comparison=">"
            )
        )

        # Both should be reasonable
        assert 0 <= f1_dinkelbach <= 1
        assert 0 <= f1_sort_scan <= 1

        # sort_scan optimizes empirical F1, so should be >= Dinkelbach on this dataset
        # (allowing small numerical tolerance)
        assert f1_sort_scan >= f1_dinkelbach - 1e-10, (
            f"sort_scan F1 {f1_sort_scan:.6f} should be >= Dinkelbach F1 "
            f"{f1_dinkelbach:.6f} "
            f"on empirical data"
        )

    def test_dinkelbach_expected_vs_empirical_relationship(self):
        """Test relationship between expected and empirical F1 for Dinkelbach."""
        # Create perfectly calibrated data
        rng = np.random.default_rng(1234)
        n = 150
        probs = rng.uniform(0.1, 0.9, size=n)  # Avoid extreme probabilities
        labels = rng.binomial(1, probs)  # Perfectly calibrated

        if np.sum(labels) <= 1 or np.sum(labels) >= n - 1:
            pytest.skip("Degenerate case")

        result = optimize_thresholds(
            labels, probs, mode="expected", metric="f1", comparison=">"
        )
        threshold = result.threshold

        # Compute empirical F1
        empirical_f1 = f1_score(
            *confusion_matrix_at_threshold(labels, probs, threshold, comparison=">")
        )

        # Compute expected F1 using the probabilities
        pred_probs = probs > threshold
        expected_tp = np.sum(probs[pred_probs])  # Expected true positives
        expected_fp = np.sum((1 - probs)[pred_probs])  # Expected false positives
        expected_fn = np.sum(probs[~pred_probs])  # Expected false negatives

        if expected_tp + expected_fp > 0 and expected_tp + expected_fn > 0:
            expected_precision = expected_tp / (expected_tp + expected_fp)
            expected_recall = expected_tp / (expected_tp + expected_fn)
            expected_f1 = (
                2
                * expected_precision
                * expected_recall
                / (expected_precision + expected_recall)
            )
        else:
            expected_f1 = 0.0

        # For large samples, empirical should be close to expected (stochastic
        # relationship)
        # We mainly test that both are reasonable, not exact equality
        assert 0 <= expected_f1 <= 1, f"Expected F1 {expected_f1} out of range"
        assert abs(empirical_f1 - expected_f1) < 0.5, (
            f"Empirical F1 {empirical_f1:.4f} very different from expected "
            f"{expected_f1:.4f}"
        )


class TestDinkelbachTieHandling:
    """Test Dinkelbach tie handling with comparison operators."""

    def test_dinkelbach_tie_behavior_follows_comparison(self):
        """Dinkelbach tie handling should follow comparison operator semantics."""
        # Create data with ties that might affect the optimal threshold
        probs = np.array([0.2, 0.5, 0.5, 0.5, 0.8])
        labels = np.array([0, 1, 0, 1, 1])  # Mixed labels (irrelevant for Dinkelbach)

        result_exclusive = optimize_thresholds(
            labels, probs, mode="expected", metric="f1", comparison=">"
        )
        result_inclusive = optimize_thresholds(
            labels, probs, mode="expected", metric="f1", comparison=">="
        )

        # Extract thresholds from results
        threshold_exclusive = result_exclusive.threshold
        threshold_inclusive = result_inclusive.threshold
        # Both should be valid
        assert 0 <= threshold_exclusive <= 1
        assert 0 <= threshold_inclusive <= 1

        # Test prediction behavior when threshold equals tied probabilities
        if abs(threshold_exclusive - 0.5) < 1e-12:
            # Threshold exactly at tie - test exclusion
            pred_exclusive = probs > threshold_exclusive
            tied_indices = np.isclose(probs, 0.5, atol=1e-10)
            assert not pred_exclusive[tied_indices].any(), (
                "Exclusive '>' should not include tied probabilities"
            )

        if abs(threshold_inclusive - 0.5) < 1e-12:
            # Threshold exactly at tie - test inclusion
            pred_inclusive = probs >= threshold_inclusive
            tied_indices = np.isclose(probs, 0.5, atol=1e-10)
            assert pred_inclusive[tied_indices].all(), (
                "Inclusive '>=' should include tied probabilities"
            )

    def test_dinkelbach_all_probabilities_tied(self):
        """Test Dinkelbach when all probabilities are identical."""
        prob_val = 0.4
        probs = np.full(20, prob_val)
        labels = np.random.default_rng(42).integers(0, 2, size=20)  # Random labels

        for comparison in [">", ">="]:
            result = optimize_thresholds(
                labels, probs, mode="expected", metric="f1", comparison=comparison
            )
            threshold = result.threshold

            # Should produce valid threshold
            assert 0 <= threshold <= 1

            # Predictions should be all same (since all probs identical)
            pred = (probs > threshold) if comparison == ">" else (probs >= threshold)
            assert len(np.unique(pred)) == 1, (
                f"All probs equal, so predictions should be identical with {comparison}"
            )

    def test_dinkelbach_threshold_selection_with_ties(self):
        """Test Dinkelbach threshold selection strategy with tied probabilities."""
        # Create scenario where optimal Dinkelbach cut might fall at tied probabilities
        probs = np.array([0.1, 0.3, 0.6, 0.6, 0.6, 0.9])
        labels = np.array([0, 0, 1, 1, 0, 1])  # Labels don't matter for Dinkelbach

        for comparison in [">", ">="]:
            result = optimize_thresholds(
                labels, probs, mode="expected", metric="f1", comparison=comparison
            )
            threshold = result.threshold

            # Verify threshold produces consistent behavior
            pred = (probs > threshold) if comparison == ">" else (probs >= threshold)

            # Count predictions at tied probabilities (0.6)
            tied_mask = np.isclose(probs, 0.6, atol=1e-10)
            tied_predictions = pred[tied_mask]

            if len(tied_predictions) > 0 and abs(threshold - 0.6) < 1e-10:
                # If threshold is exactly at tie value
                if comparison == ">":
                    assert not tied_predictions.any(), (
                        "Tied values should be excluded with '>'"
                    )
                else:  # '>='
                    assert tied_predictions.all(), (
                        "Tied values should be included with '>='"
                    )


class TestDinkelbachEdgeCases:
    """Test Dinkelbach edge cases and robustness."""

    def test_dinkelbach_extreme_probabilities(self):
        """Test Dinkelbach with probabilities at 0 and 1."""
        probs = np.array([0.0, 0.2, 0.8, 1.0])
        labels = np.array([0, 0, 1, 1])  # Labels don't affect Dinkelbach

        for comparison in [">", ">="]:
            result = optimize_thresholds(
                labels, probs, mode="expected", metric="f1", comparison=comparison
            )
            threshold = result.threshold

            assert 0 <= threshold <= 1, (
                f"Threshold {threshold} out of bounds with extreme probs"
            )

            # Verify behavior at boundaries
            pred = (probs > threshold) if comparison == ">" else (probs >= threshold)

            # Check specific boundary behaviors
            if threshold == 0.0:
                if comparison == ">":
                    assert not pred[0], "0.0 > 0.0 should be False"
                else:  # '>='
                    assert pred[0], "0.0 >= 0.0 should be True"

            if threshold == 1.0:
                if comparison == ">":
                    assert not pred[3], "1.0 > 1.0 should be False"
                else:  # '>='
                    assert pred[3], "1.0 >= 1.0 should be True"

    def test_dinkelbach_single_probability(self):
        """Test Dinkelbach with single unique probability."""
        probs = np.array([0.7, 0.7, 0.7])
        labels = np.array([0, 1, 0])  # Mixed labels

        for comparison in [">", ">="]:
            result = optimize_thresholds(
                labels, probs, mode="expected", metric="f1", comparison=comparison
            )
            threshold = result.threshold

            assert 0 <= threshold <= 1

            # With single unique probability, decision is binary
            pred = (probs > threshold) if comparison == ">" else (probs >= threshold)

            # All predictions should be identical
            assert len(np.unique(pred)) == 1, (
                f"Single probability should give uniform predictions with {comparison}"
            )

    def test_dinkelbach_reproducibility(self):
        """Test that Dinkelbach gives identical results on repeated calls."""
        rng = np.random.default_rng(555)
        probs = rng.uniform(0, 1, size=50)
        labels = rng.integers(0, 2, size=50)  # Random labels (don't affect result)

        for comparison in [">", ">="]:
            # Multiple calls should give identical results
            thresholds = []
            for _ in range(5):
                result = optimize_thresholds(
                    labels,
                    probs,
                    metric="f1",
                    mode="expected",
                    comparison=comparison,
                )
                threshold = result.threshold
                thresholds.append(threshold)

            # All should be identical
            for i in range(1, len(thresholds)):
                assert abs(thresholds[i] - thresholds[0]) < 1e-12, (
                    f"Dinkelbach not reproducible: call {i} gave {thresholds[i]} vs "
                    f"{thresholds[0]}"
                )

    def test_dinkelbach_supports_sample_weights(self):
        """Test that Dinkelbach correctly handles sample weights."""
        probs = np.array([0.2, 0.5, 0.8])
        labels = np.array([0, 1, 1])
        weights = np.array([1.0, 2.0, 1.5])

        # Should not raise an error with sample weights
        result = optimize_thresholds(
            labels, probs, mode="expected", metric="f1", sample_weight=weights
        )

        # Extract threshold from tuple and verify it's valid
        threshold = result.threshold
        expected_f1 = result.score
        assert 0 <= threshold <= 1
        assert 0 <= expected_f1 <= 1

    def test_dinkelbach_supports_multiple_metrics(self):
        """Test that Dinkelbach supports F1 and other F-beta related metrics."""
        probs = np.array([0.3, 0.7])
        labels = np.array([0, 1])

        # Should work with F1
        result_f1 = optimize_thresholds(labels, probs, mode="expected", metric="f1")
        threshold_f1 = result_f1.threshold
        assert 0 <= threshold_f1 <= 1

        # Currently, expected mode only supports F1 metric
        # (fbeta is mentioned in code but not in metric registry)

        # Test that other metrics work or handle gracefully
        for metric in ["accuracy", "recall"]:
            try:
                result = optimize_thresholds(
                    labels, probs, metric=metric, mode="expected"
                )
                threshold = result.threshold
                score = result.score
                assert 0 <= threshold <= 1
                assert 0 <= score <= 1
            except (ValueError, NotImplementedError):
                # If not supported, that's also acceptable
                pass
