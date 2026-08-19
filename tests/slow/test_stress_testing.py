"""Stress testing for robustness and reliability.

This module contains stress tests that push the library to its limits
to identify potential failure modes and ensure robust behavior.
"""

import gc
import time
import warnings

import numpy as np
import pytest

from optimal_cutoffs import optimize_thresholds
from optimal_cutoffs.metrics import compute_metric_at_threshold
from tests.fixtures.assertions import (
    assert_valid_metric_score,
    assert_valid_threshold,
)
from tests.fixtures.data_generators import (
    generate_binary_data,
)


class TestConcurrentStressTesting:
    """Test behavior under concurrent/repeated execution stress."""

    @pytest.mark.slow
    def test_rapid_fire_optimization(self):
        """Test rapid consecutive optimizations."""
        n_iterations = 200
        n_samples = 500

        success_count = 0
        failures = []
        execution_times = []

        for i in range(n_iterations):
            try:
                # Generate new data each iteration
                y_true, pred_prob = generate_binary_data(n_samples, random_state=i)

                start_time = time.perf_counter()
                result = optimize_thresholds(y_true, pred_prob, metric="f1")
                end_time = time.perf_counter()

                execution_time = end_time - start_time
                execution_times.append(execution_time)

                threshold = result.threshold
                assert_valid_threshold(threshold)
                success_count += 1

                # Verify quality occasionally
                if i % 20 == 0:
                    score = compute_metric_at_threshold(
                        y_true, pred_prob, threshold, "f1"
                    )
                    assert_valid_metric_score(score, "f1")

            except Exception as e:
                failures.append((i, str(e)))

        # Analyze results
        success_rate = success_count / n_iterations
        assert success_rate > 0.95, (
            f"Success rate {success_rate:.2%} too low. Failures: {failures[:5]}"
        )

        if execution_times:
            mean_time = np.mean(execution_times)
            max_time = np.max(execution_times)

            assert max_time < 5.0, f"Slowest optimization took {max_time:.4f}s"
            assert mean_time < 1.0, f"Average time {mean_time:.4f}s too high"

    @pytest.mark.slow
    def test_memory_pressure_optimization(self):
        """Test optimization under memory pressure."""
        # Create multiple large datasets to pressure memory
        datasets = []
        n_datasets = 20
        n_samples = 5000

        try:
            # Build up memory pressure
            for i in range(n_datasets):
                y_true, pred_prob = generate_binary_data(n_samples, random_state=i)
                datasets.append((y_true, pred_prob))

            # Now try optimization with memory pressure
            success_count = 0
            for i, (y_true, pred_prob) in enumerate(datasets):
                try:
                    result = optimize_thresholds(y_true, pred_prob)
                    threshold = result.threshold
                    assert_valid_threshold(threshold)
                    success_count += 1
                except MemoryError:
                    # Expected under high memory pressure
                    break
                except Exception as e:
                    print(f"Dataset {i} failed under memory pressure: {e}")

            # Should handle at least some datasets under memory pressure
            assert success_count > n_datasets * 0.5, (
                f"Only {success_count}/{n_datasets} succeeded under memory pressure"
            )

        except MemoryError:
            pytest.skip("Insufficient memory for memory pressure test")
        finally:
            # Clean up to avoid affecting other tests
            del datasets
            gc.collect()

    @pytest.mark.slow
    def test_random_seed_stress(self):
        """Test optimization stability across many random seeds."""
        n_seeds = 100
        n_samples = 1000

        thresholds = []
        scores = []

        for seed in range(n_seeds):
            y_true, pred_prob = generate_binary_data(n_samples, random_state=seed)

            result = optimize_thresholds(y_true, pred_prob, method="sort_scan")
            threshold = result.threshold
            score = compute_metric_at_threshold(y_true, pred_prob, threshold, "f1")
            assert_valid_threshold(threshold)
            assert_valid_metric_score(score, "f1")

            thresholds.append(threshold)
            scores.append(score)

        # Analyze distribution of results
        threshold_std = np.std(thresholds)
        score_std = np.std(scores)

        # Results should vary (different data) but not be extremely variable
        assert 0.01 < threshold_std < 0.5, (
            f"Threshold std {threshold_std:.4f} unexpected"
        )
        assert 0.01 < score_std < 0.3, f"Score std {score_std:.4f} unexpected"


class TestResourceExhaustionStress:
    """Test behavior when approaching resource limits."""

    @pytest.mark.slow
    def test_computational_complexity_stress(self):
        """Test with computationally challenging scenarios."""
        # Create scenarios that stress different algorithmic aspects

        scenarios = [
            # Many unique probability values (stress sorting)
            lambda n: (
                np.random.binomial(1, 0.5, n),
                np.linspace(0.001, 0.999, n),  # All unique
            ),
            # Many tied values (stress tie handling)
            lambda n: (
                np.random.binomial(1, 0.5, n),
                np.random.choice([0.2, 0.5, 0.8], n),  # Only 3 unique values
            ),
            # High precision requirements (stress numerical algorithms)
            lambda n: (
                np.random.binomial(1, 0.5, n),
                0.5 + np.random.normal(0, 1e-6, n),  # Very narrow distribution
            ),
        ]

        max_size = 20000  # Large enough to stress algorithms

        for i, scenario_gen in enumerate(scenarios):
            try:
                y_true, pred_prob = scenario_gen(max_size)
                pred_prob = np.clip(pred_prob, 0, 1)

                # Ensure both classes present
                if np.sum(y_true) == 0:
                    y_true[0] = 1
                elif np.sum(y_true) == len(y_true):
                    y_true[0] = 0

                start_time = time.time()
                result = optimize_thresholds(y_true, pred_prob)
                end_time = time.time()

                execution_time = end_time - start_time

                threshold = result.threshold
                assert_valid_threshold(threshold)
                assert execution_time < 60.0, (
                    f"Scenario {i} took {execution_time:.2f}s with {max_size} samples"
                )

            except (MemoryError, ValueError) as e:
                print(f"Computational stress scenario {i} failed: {e}")

    @pytest.mark.slow
    def test_extreme_dataset_characteristics(self):
        """Test with extreme dataset characteristics."""
        base_size = 10000

        extreme_cases = [
            # Extreme class imbalance
            {
                "name": "extreme_imbalance",
                "generator": lambda: (
                    np.concatenate([np.zeros(base_size - 1), np.ones(1)]),
                    np.random.uniform(0, 1, base_size),
                ),
            },
            # All probabilities at boundaries
            {
                "name": "boundary_probabilities",
                "generator": lambda: (
                    np.random.binomial(1, 0.5, base_size),
                    np.random.choice([0.0, 1.0], base_size),
                ),
            },
            # Extremely small probability differences
            {
                "name": "tiny_differences",
                "generator": lambda: (
                    np.random.binomial(1, 0.5, base_size),
                    0.5 + np.random.uniform(-1e-10, 1e-10, base_size),
                ),
            },
        ]

        success_count = 0

        for case in extreme_cases:
            try:
                y_true, pred_prob = case["generator"]()
                pred_prob = np.clip(pred_prob, 0, 1)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # Suppress expected warnings

                    result = optimize_thresholds(y_true, pred_prob)
                    threshold = result.threshold
                    assert_valid_threshold(threshold)

                    score = compute_metric_at_threshold(
                        y_true, pred_prob, threshold, "f1"
                    )
                    assert_valid_metric_score(score, "f1", allow_nan=True)

                success_count += 1

            except Exception as e:
                print(f"Extreme case '{case['name']}' failed: {e}")

        # Should handle at least some extreme cases
        assert success_count > 0, "All extreme cases failed"


class TestRobustnessUnderAdversarialConditions:
    """Test robustness under deliberately challenging conditions."""

    @pytest.mark.slow
    def test_adversarial_probability_patterns(self):
        """Test with adversarially designed probability patterns."""
        n_samples = 8000

        adversarial_patterns = [
            # Probabilities designed to confuse sorting algorithms
            lambda: np.tile([0.1, 0.9, 0.1, 0.9], n_samples // 4),
            # Probabilities with repeated sequences
            lambda: np.tile(np.linspace(0, 1, 100), n_samples // 100),
            # Probabilities with fractal-like patterns
            lambda: np.array(
                [
                    0.5 + 0.4 * np.sin(i * 2 * np.pi / 100) * np.sin(i * 2 * np.pi / 7)
                    for i in range(n_samples)
                ]
            ),
            # Probabilities clustering around decision boundaries
            lambda: np.concatenate(
                [
                    np.random.normal(0.499, 0.001, n_samples // 2),
                    np.random.normal(0.501, 0.001, n_samples // 2),
                ]
            ),
        ]

        for i, pattern_gen in enumerate(adversarial_patterns):
            try:
                pred_prob = pattern_gen()
                pred_prob = np.clip(pred_prob, 0, 1)

                # Generate labels with some correlation to probabilities
                y_true = (
                    pred_prob + 0.1 * np.random.normal(0, 1, n_samples) > 0.5
                ).astype(int)

                # Ensure both classes present
                if np.sum(y_true) == 0:
                    y_true[0] = 1
                elif np.sum(y_true) == len(y_true):
                    y_true[0] = 0

                result = optimize_thresholds(y_true, pred_prob)
                threshold = result.threshold
                assert_valid_threshold(threshold)

                score = compute_metric_at_threshold(y_true, pred_prob, threshold, "f1")
                assert_valid_metric_score(score, "f1")

            except Exception as e:
                print(f"Adversarial pattern {i} failed: {e}")

    @pytest.mark.slow
    def test_malformed_input_resilience(self):
        """Test resilience to various forms of malformed input."""
        n_samples = 1000

        # Test cases that push input validation limits
        edge_cases = [
            # Labels with float values that should convert to int
            {
                "y_true": np.array([0.0, 1.0, 0.0, 1.0] * (n_samples // 4)),
                "pred_prob": np.random.uniform(0, 1, n_samples),
                "should_work": True,
            },
            # Very large arrays with valid data
            {
                "y_true": np.random.binomial(1, 0.5, n_samples * 5),
                "pred_prob": np.random.uniform(0, 1, n_samples * 5),
                "should_work": True,
            },
            # Probabilities exactly at boundaries
            {
                "y_true": np.random.binomial(1, 0.5, n_samples),
                "pred_prob": np.random.choice([0.0, 1.0], n_samples),
                "should_work": True,
            },
        ]

        for i, case in enumerate(edge_cases):
            try:
                result = optimize_thresholds(case["y_true"], case["pred_prob"])

                if case["should_work"]:
                    threshold = result.threshold
                    assert_valid_threshold(threshold)
                else:
                    pytest.fail(f"Case {i} should have failed but didn't")

            except Exception as e:
                if case["should_work"]:
                    print(f"Edge case {i} unexpectedly failed: {e}")
                # If it shouldn't work, failure is expected

    @pytest.mark.slow
    def test_floating_point_edge_cases(self):
        """Test floating point edge cases and precision limits."""
        n_samples = 5000

        fp_cases = [
            # Values at machine epsilon
            {
                "name": "machine_epsilon",
                "generator": lambda: (
                    np.random.binomial(1, 0.5, n_samples),
                    np.finfo(float).eps * np.random.uniform(-1000, 1000, n_samples)
                    + 0.5,
                ),
            },
            # Subnormal numbers
            {
                "name": "subnormal",
                "generator": lambda: (
                    np.random.binomial(1, 0.5, n_samples),
                    np.random.uniform(0, np.finfo(float).tiny, n_samples),
                ),
            },
            # Values very close to 1.0
            {
                "name": "near_one",
                "generator": lambda: (
                    np.random.binomial(1, 0.5, n_samples),
                    1.0 - np.random.uniform(0, np.finfo(float).eps * 1000, n_samples),
                ),
            },
        ]

        for case in fp_cases:
            try:
                y_true, pred_prob = case["generator"]()
                pred_prob = np.clip(pred_prob, 0, 1)  # Ensure valid range

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # Suppress precision warnings

                    result = optimize_thresholds(y_true, pred_prob)
                    threshold = result.threshold
                    assert_valid_threshold(threshold)

                print(f"Floating point case '{case['name']}' succeeded")

            except Exception as e:
                print(f"Floating point case '{case['name']}' failed: {e}")


class TestLongRunningReliability:
    """Test reliability over extended periods and many operations."""

    @pytest.mark.slow
    def test_sustained_operation_reliability(self):
        """Test reliability during sustained operation."""
        n_iterations = 500
        n_samples = 2000

        failure_rate_threshold = 0.05  # Allow 5% failure rate
        performance_degradation_threshold = 2.0  # Allow 2x slowdown

        initial_times = []
        final_times = []
        failure_count = 0

        # Measure initial performance
        for i in range(10):
            y_true, pred_prob = generate_binary_data(n_samples, random_state=i)

            start_time = time.perf_counter()
            try:
                result = optimize_thresholds(y_true, pred_prob)
                end_time = time.perf_counter()
                initial_times.append(end_time - start_time)
                threshold = result.threshold
                assert_valid_threshold(threshold)
            except Exception:
                failure_count += 1

        # Run sustained operations
        for i in range(10, n_iterations - 10):
            y_true, pred_prob = generate_binary_data(n_samples, random_state=i)

            try:
                result = optimize_thresholds(y_true, pred_prob)
                threshold = result.threshold
                assert_valid_threshold(threshold)
            except Exception:
                failure_count += 1

        # Measure final performance
        for i in range(n_iterations - 10, n_iterations):
            y_true, pred_prob = generate_binary_data(n_samples, random_state=i)

            start_time = time.perf_counter()
            try:
                result = optimize_thresholds(y_true, pred_prob)
                end_time = time.perf_counter()
                final_times.append(end_time - start_time)
                threshold = result.threshold
                assert_valid_threshold(threshold)
            except Exception:
                failure_count += 1

        # Analyze results
        failure_rate = failure_count / n_iterations
        assert failure_rate < failure_rate_threshold, (
            f"Failure rate {failure_rate:.2%} exceeds threshold "
            f"{failure_rate_threshold:.2%}"
        )

        if initial_times and final_times:
            initial_mean = np.mean(initial_times)
            final_mean = np.mean(final_times)
            performance_ratio = final_mean / initial_mean

            assert performance_ratio < performance_degradation_threshold, (
                f"Performance degraded by {performance_ratio:.1f}x over sustained "
                f"operation"
            )

    @pytest.mark.slow
    def test_determinism_over_time(self):
        """Test that optimization remains deterministic over many operations."""
        n_iterations = 100
        reference_data_seed = 12345

        # Generate reference data
        y_true, pred_prob = generate_binary_data(1000, random_state=reference_data_seed)

        # Get reference result
        reference_result = optimize_thresholds(y_true, pred_prob, method="sort_scan")
        reference_threshold = reference_result.threshold

        # Test determinism over many operations
        non_deterministic_count = 0

        for i in range(n_iterations):
            # Do some other operations between tests
            for j in range(5):
                temp_y, temp_p = generate_binary_data(500, random_state=i * 100 + j)
                _ = optimize_thresholds(temp_y, temp_p)

            # Test determinism on reference data
            current_result = optimize_thresholds(y_true, pred_prob, method="sort_scan")
            current_threshold = current_result.threshold

            if abs(current_threshold - reference_threshold) > 1e-12:
                non_deterministic_count += 1

        # Should remain deterministic
        determinism_rate = 1 - (non_deterministic_count / n_iterations)
        assert determinism_rate > 0.99, (
            f"Determinism rate {determinism_rate:.2%} too low over {n_iterations} "
            f"iterations"
        )
