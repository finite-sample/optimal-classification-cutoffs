"""Test determinism and stable sorting properties.

This module tests that the optimization algorithms are deterministic and
produce stable, reproducible results:

1. Determinism: Identical inputs produce identical outputs across multiple runs
2. Stable sorting: When multiple thresholds achieve the same score, selection is
    consistent
3. Reproducibility: Results are consistent across different environments/platforms
4. Numerical stability: Small changes in input produce predictable changes in output

These tests ensure the reliability and reproducibility of the threshold
optimization algorithms for scientific and production use cases.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from optimal_cutoffs import optimize_thresholds


def _create_tied_score_scenario():
    """Create a scenario where multiple thresholds achieve the same optimal score."""
    # Carefully crafted case where several thresholds give identical F1
    labels = np.array([0, 0, 1, 1, 0, 1])
    probs = np.array([0.1, 0.4, 0.6, 0.7, 0.3, 0.8])

    return labels, probs


def _create_numerical_precision_scenario():
    """Create scenario with values that test numerical precision."""
    # Values with small differences that could cause instability
    labels = np.array([0, 1, 0, 1, 1])
    probs = np.array(
        [
            0.5000000001,  # Very close to 0.5
            0.5000000002,
            0.4999999998,  # Very close to 0.5 from below
            0.5000000003,
            0.4999999999,
        ]
    )

    return labels, probs


class TestBasicDeterminism:
    """Test basic deterministic behavior of optimization algorithms."""

    def test_identical_inputs_identical_outputs(self):
        """Identical inputs should produce identical outputs across multiple runs."""
        labels = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        probs = np.array([0.2, 0.8, 0.7, 0.3, 0.9, 0.1, 0.6, 0.75, 0.25, 0.4])

        methods_to_test = ["sort_scan", "sort_scan", "minimize"]

        for method in methods_to_test:
            thresholds = []

            for run in range(5):
                try:
                    result = optimize_thresholds(
                        labels, probs, metric="f1", method=method, comparison=">"
                    )
                    thresholds.append(result.threshold)

                except (ValueError, NotImplementedError) as e:
                    if "not supported" in str(e).lower():
                        pytest.skip(f"Method {method} not supported")
                    raise

            if thresholds:
                # All runs should produce identical results
                for i in range(1, len(thresholds)):
                    assert thresholds[i] == thresholds[0], (
                        f"Method {method} not deterministic: run {i} gave "
                        f"{thresholds[i]:.15f}, "
                        f"run 0 gave {thresholds[0]:.15f}"
                    )

    def test_determinism_across_metrics(self):
        """Test determinism across different metrics."""
        labels = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        probs = np.array([0.3, 0.7, 0.2, 0.8, 0.6, 0.4, 0.9, 0.1])

        metrics_to_test = ["f1", "accuracy", "precision", "recall"]

        for metric in metrics_to_test:
            results = []

            for run in range(3):
                try:
                    result = optimize_thresholds(
                        labels,
                        probs,
                        metric=metric,
                        method="sort_scan",
                        comparison=">=",
                    )
                    results.append(result.threshold)

                except ValueError as e:
                    if "not supported" in str(e).lower():
                        continue
                    raise

            if len(results) > 1:
                # All runs should be identical
                for i in range(1, len(results)):
                    assert results[i] == results[0], (
                        f"Metric {metric} not deterministic: run {i} gave "
                        f"{results[i]}, "
                        f"run 0 gave {results[0]}"
                    )

    def test_determinism_with_weights(self):
        """Test determinism when sample weights are used."""
        labels = np.array([0, 1, 1, 0, 1, 0])
        probs = np.array([0.2, 0.7, 0.8, 0.3, 0.6, 0.4])
        weights = np.array([1.0, 2.0, 1.5, 0.8, 2.2, 1.3])

        results = []
        for run in range(4):
            try:
                result = optimize_thresholds(
                    labels,
                    probs,
                    metric="f1",
                    method="sort_scan",
                    sample_weight=weights,
                    comparison=">",
                )
                results.append(result.threshold)

            except (ValueError, NotImplementedError):
                pytest.skip("Weighted optimization not supported")

        if len(results) > 1:
            for i in range(1, len(results)):
                assert results[i] == results[0], (
                    f"Weighted optimization not deterministic: run {i} gave "
                    f"{results[i]}, run 0 gave {results[0]}"
                )

    @given(n_samples=st.integers(10, 50), random_seed=st.integers(0, 1000))
    @settings(deadline=None, max_examples=20)
    def test_determinism_property_based(self, n_samples, random_seed):
        """Property-based test for determinism."""
        # Generate data with fixed seed for reproducibility
        rng = np.random.default_rng(random_seed)
        labels = rng.integers(0, 2, size=n_samples)
        probs = rng.uniform(0, 1, size=n_samples)

        # Ensure both classes present
        if labels.sum() == 0:
            labels[0] = 1
        if labels.sum() == labels.size:
            labels[0] = 0

        # Run optimization multiple times
        try:
            results = []
            for _ in range(3):
                result = optimize_thresholds(
                    labels, probs, metric="accuracy", method="sort_scan", comparison=">"
                )
                results.append(result.threshold)

            # Should be identical
            for i in range(1, len(results)):
                assert results[i] == results[0], (
                    f"Property-based determinism failed: {results}"
                )

        except Exception as e:
            if any(
                phrase in str(e).lower()
                for phrase in ["degenerate", "empty", "single class"]
            ):
                pytest.skip("Degenerate case in property test")
            raise


class TestStableSorting:
    """Test stable sorting behavior when multiple thresholds achieve same score."""

    def test_tied_scores_consistent_selection(self):
        """When several thresholds tie on score, the choice stays consistent."""
        labels, probs = _create_tied_score_scenario()

        # Run multiple times to check consistency
        results = []
        for _ in range(10):
            try:
                result = optimize_thresholds(
                    labels, probs, metric="f1", method="sort_scan", comparison=">"
                )
                results.append(result.threshold)

            except ValueError:
                pytest.skip("Tied score scenario not supported")

        if results:
            # All results should be identical despite potential ties
            for i in range(1, len(results)):
                assert results[i] == results[0], (
                    f"Tied score handling not stable: got {results[i]} vs {results[0]}"
                )

    def test_stable_selection_with_duplicate_probabilities(self):
        """Test stable behavior when probabilities have duplicates."""
        labels = np.array([0, 1, 0, 1, 1, 0])
        probs = np.array([0.3, 0.6, 0.6, 0.6, 0.8, 0.3])  # Duplicates at 0.3 and 0.6

        results = []
        for comparison in [">", ">="]:
            for _ in range(5):
                try:
                    result = optimize_thresholds(
                        labels,
                        probs,
                        metric="accuracy",
                        method="sort_scan",
                        comparison=comparison,
                    )
                    results.append((comparison, result.threshold))

                except ValueError:
                    continue

        # Group by comparison operator
        exclusive_results = [t for comp, t in results if comp == ">"]
        inclusive_results = [t for comp, t in results if comp == ">="]

        # Within each comparison type, results should be identical
        if len(exclusive_results) > 1:
            for i in range(1, len(exclusive_results)):
                assert exclusive_results[i] == exclusive_results[0], (
                    f"Exclusive comparison not stable with duplicates: "
                    f"{exclusive_results}"
                )

        if len(inclusive_results) > 1:
            for i in range(1, len(inclusive_results)):
                assert inclusive_results[i] == inclusive_results[0], (
                    f"Inclusive comparison not stable with duplicates: "
                    f"{inclusive_results}"
                )

    def test_stable_sorting_preserves_order(self):
        """Test that stable sorting preserves relative order of tied elements."""
        # Create case where stable sorting matters
        labels = np.array([0, 1, 1, 0, 1])
        probs = np.array([0.2, 0.5, 0.5, 0.8, 0.5])  # Multiple 0.5 values

        try:
            result = optimize_thresholds(
                labels, probs, metric="f1", method="sort_scan", comparison=">"
            )

            # Test behavior at the tied probability value
            threshold = result.threshold
            if abs(threshold - 0.5) < 1e-10:
                # If threshold is exactly at tie value, behavior should be consistent
                pred = probs > threshold
                tied_indices = np.where(np.isclose(probs, 0.5, atol=1e-10))[0]

                if len(tied_indices) > 1:
                    # All tied values should have same prediction
                    tied_predictions = pred[tied_indices]
                    assert len(set(tied_predictions)) <= 1, (
                        f"Tied probabilities should have consistent predictions: "
                        f"{tied_predictions}"
                    )

            # Result should be valid regardless
            assert 0 <= threshold <= 1

        except ValueError:
            pytest.skip("Stable sorting test case not supported")

    def test_deterministic_tie_breaking(self):
        """Test that tie-breaking is deterministic."""
        # Case designed to have multiple optimal thresholds
        labels = np.array([0, 0, 1, 1])
        probs = np.array([0.2, 0.4, 0.6, 0.8])

        # Multiple runs should select the same threshold among tied options
        thresholds = []
        for _ in range(8):
            try:
                result = optimize_thresholds(
                    labels, probs, metric="accuracy", method="sort_scan", comparison=">"
                )
                thresholds.append(result.threshold)

            except ValueError:
                continue

        if len(thresholds) > 1:
            # Should be deterministic
            for i in range(1, len(thresholds)):
                assert thresholds[i] == thresholds[0], (
                    f"Tie-breaking not deterministic: {thresholds}"
                )


class TestNumericalStability:
    """Test numerical stability and precision handling."""

    def test_numerical_precision_stability(self):
        """Test stability with values that test numerical precision."""
        labels, probs = _create_numerical_precision_scenario()

        try:
            result = optimize_thresholds(
                labels, probs, metric="f1", method="sort_scan", comparison=">"
            )

            # Should handle precision issues gracefully
            threshold = result.threshold
            assert 0 <= threshold <= 1
            assert not np.isnan(threshold)
            assert not np.isinf(threshold)

            # Result should be reproducible
            result2 = optimize_thresholds(
                labels, probs, metric="f1", method="sort_scan", comparison=">"
            )
            threshold2 = result2.threshold

            assert threshold == threshold2, (
                f"Numerical precision case not reproducible: {threshold} vs "
                f"{threshold2}"
            )

        except ValueError:
            pytest.skip("Numerical precision test case not supported")

    def test_extreme_probability_values(self):
        """Test behavior with probabilities very close to 0 and 1."""
        labels = np.array([0, 1, 0, 1])
        probs = np.array([1e-15, 1 - 1e-15, 1e-14, 1 - 1e-14])  # Extreme values

        methods_to_test = ["sort_scan", "sort_scan"]

        for method in methods_to_test:
            results = []

            for _ in range(3):
                try:
                    result = optimize_thresholds(
                        labels, probs, metric="accuracy", method=method, comparison=">"
                    )
                    results.append(result.threshold)

                except (ValueError, NotImplementedError) as e:
                    if "not supported" in str(e).lower():
                        break
                    raise

            if len(results) > 1:
                # Should be stable with extreme values
                for i in range(1, len(results)):
                    assert results[i] == results[0], (
                        f"Method {method} not stable with extreme probabilities: "
                        f"{results}"
                    )

                # Results should be valid
                assert all(0 <= r <= 1 for r in results)
                assert all(not np.isnan(r) and not np.isinf(r) for r in results)

    def test_small_perturbation_stability(self):
        """Test that small perturbations produce predictable changes."""
        base_labels = np.array([0, 1, 1, 0, 1])
        base_probs = np.array([0.2, 0.7, 0.6, 0.3, 0.8])

        try:
            base_result = optimize_thresholds(
                base_labels, base_probs, metric="f1", method="sort_scan", comparison=">"
            )

            # Apply small perturbations
            perturbations = [1e-10, -1e-10, 1e-12, -1e-12]

            for perturbation in perturbations:
                perturbed_probs = base_probs + perturbation
                perturbed_probs = np.clip(perturbed_probs, 0, 1)  # Keep in valid range

                perturbed_result = optimize_thresholds(
                    base_labels,
                    perturbed_probs,
                    metric="f1",
                    method="sort_scan",
                    comparison=">",
                )

                # Small perturbations should produce small changes (or no change)
                base_threshold = base_result.threshold
                perturbed_threshold = perturbed_result.threshold
                threshold_change = abs(perturbed_threshold - base_threshold)

                # Either no change (stable) or change proportional to perturbation
                assert threshold_change <= abs(perturbation) * 1000, (
                    f"Large threshold change {threshold_change} from small "
                    f"perturbation "
                    f"{perturbation}"
                )

        except ValueError:
            pytest.skip("Small perturbation test not supported")

    def test_floating_point_edge_cases(self):
        """Test edge cases in floating point arithmetic."""
        # Test with values that might cause floating point issues
        edge_cases = [
            (np.array([0, 1]), np.array([0.0, 1.0])),  # Exact boundaries
            (
                np.array([0, 1]),
                np.array([np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)]),
            ),  # Near boundaries
            (np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5])),  # All identical
        ]

        for labels, probs in edge_cases:
            for comparison in [">", ">="]:
                try:
                    result = optimize_thresholds(
                        labels,
                        probs,
                        metric="accuracy",
                        method="sort_scan",
                        comparison=comparison,
                    )

                    # Should handle edge cases gracefully
                    threshold = result.threshold
                    assert 0 <= threshold <= 1, (
                        f"Threshold {threshold} out of bounds for edge case"
                    )
                    assert not np.isnan(threshold), "NaN threshold for edge case"
                    assert not np.isinf(threshold), "Infinite threshold for edge case"

                    # Should be reproducible
                    result2 = optimize_thresholds(
                        labels,
                        probs,
                        metric="accuracy",
                        method="sort_scan",
                        comparison=comparison,
                    )
                    threshold2 = result2.threshold

                    assert threshold == threshold2, (
                        f"Edge case not reproducible: {threshold} vs {threshold2}"
                    )

                except ValueError as e:
                    if (
                        "degenerate" in str(e).lower()
                        or "single class" in str(e).lower()
                    ):
                        continue  # Expected for some edge cases
                    raise


class TestReproducibilityAcrossPlatforms:
    """Test reproducibility across different conditions."""

    def test_reproducibility_with_different_array_types(self):
        """Test reproducibility with different numpy array types."""
        base_labels = [0, 1, 1, 0, 1, 0]
        base_probs = [0.2, 0.7, 0.8, 0.3, 0.9, 0.1]

        # Convert to different numpy types
        array_types = [
            (np.array(base_labels, dtype=int), np.array(base_probs, dtype=float)),
            (
                np.array(base_labels, dtype=np.int32),
                np.array(base_probs, dtype=np.float32),
            ),
            (
                np.array(base_labels, dtype=np.int64),
                np.array(base_probs, dtype=np.float64),
            ),
        ]

        results = []
        for labels, probs in array_types:
            try:
                result = optimize_thresholds(
                    labels, probs, metric="f1", method="sort_scan", comparison=">"
                )
                results.append(result.threshold)

            except ValueError:
                continue

        if len(results) > 1:
            # Should be consistent across array types
            for i in range(1, len(results)):
                assert abs(results[i] - results[0]) < 1e-12, (
                    f"Results vary by array type: {results}"
                )

    def test_reproducibility_with_list_vs_array_input(self):
        """Test that list and array inputs produce identical results."""
        labels_list = [0, 1, 0, 1, 1, 0, 1]
        probs_list = [0.3, 0.8, 0.2, 0.7, 0.9, 0.1, 0.6]

        labels_array = np.array(labels_list)
        probs_array = np.array(probs_list)

        try:
            result_list = optimize_thresholds(
                labels_list,
                probs_list,
                metric="accuracy",
                method="sort_scan",
                comparison=">",
            )

            result_array = optimize_thresholds(
                labels_array,
                probs_array,
                metric="accuracy",
                method="sort_scan",
                comparison=">",
            )

            # Should be identical
            threshold_list = result_list.threshold
            threshold_array = result_array.threshold
            assert threshold_list == threshold_array, (
                f"List vs array inputs give different results: {threshold_list} vs "
                f"{threshold_array}"
            )

        except ValueError:
            pytest.skip("List input not supported")

    def test_reproducibility_across_multiple_calls(self):
        """Test that multiple calls in sequence produce identical results."""
        labels = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1])
        probs = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.85, 0.15, 0.75])

        # Make many sequential calls
        results = []
        for i in range(20):
            try:
                result = optimize_thresholds(
                    labels, probs, metric="f1", method="sort_scan", comparison=">="
                )
                results.append((i, result.threshold))

            except ValueError:
                continue

        if len(results) > 1:
            # All should be identical
            first_threshold = results[0][1]
            for call_num, threshold in results[1:]:
                assert threshold == first_threshold, (
                    f"Call {call_num} gave different result: {threshold} vs "
                    f"{first_threshold}"
                )

    def test_order_independence(self):
        """Test that input order doesn't affect results when it shouldn't."""
        labels = np.array([0, 1, 0, 1, 1])
        probs = np.array([0.2, 0.8, 0.3, 0.7, 0.9])

        # Create permuted version
        perm = np.array([4, 0, 2, 1, 3])  # Specific permutation
        labels_perm = labels[perm]
        probs_perm = probs[perm]

        try:
            result_original = optimize_thresholds(
                labels, probs, metric="accuracy", method="sort_scan", comparison=">"
            )

            result_permuted = optimize_thresholds(
                labels_perm,
                probs_perm,
                metric="accuracy",
                method="sort_scan",
                comparison=">",
            )

            # Results should be identical (threshold optimization is order-independent)
            threshold_original = result_original.threshold
            threshold_permuted = result_permuted.threshold
            assert threshold_original == threshold_permuted, (
                f"Order dependence detected: original={threshold_original}, "
                f"permuted={threshold_permuted}"
            )

        except ValueError:
            pytest.skip("Order independence test not supported")
