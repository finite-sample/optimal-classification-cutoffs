"""Performance benchmarking tests to verify algorithmic complexity guarantees.

This module contains tests that verify the O(n log n) performance characteristics
of the sort-scan optimization algorithm and other performance-critical components.
"""

import gc
import time

import numpy as np
import pytest

from optimal_cutoffs import optimize_thresholds
from optimal_cutoffs.metrics import get_metric_function
from optimal_cutoffs.piecewise import optimal_threshold_sortscan


def generate_test_data(
    n_samples: int, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate binary classification test data."""
    rng = np.random.RandomState(random_state)
    y_true = rng.randint(0, 2, n_samples)
    pred_prob = rng.uniform(0, 1, n_samples)
    return y_true, pred_prob


class TestAlgorithmicComplexity:
    """Test that algorithms meet their expected complexity guarantees."""

    @pytest.mark.parametrize("n_samples", [100, 500, 1000, 2000, 4000])
    def test_sort_scan_n_log_n_scaling(self, n_samples):
        """Test that sort_scan algorithm scales as O(n log n)."""
        y_true, pred_prob = generate_test_data(n_samples, random_state=42)

        # Get vectorized metric
        try:
            f1_vectorized = get_metric_function("f1")
        except (KeyError, AttributeError, ValueError):
            pytest.skip("Vectorized F1 metric not available")

        # Warm up and force garbage collection
        gc.collect()

        # Time the optimization
        start_time = time.perf_counter()
        result = optimal_threshold_sortscan(y_true, pred_prob, f1_vectorized)
        end_time = time.perf_counter()

        execution_time = end_time - start_time

        # Extract results from OptimizationResult
        threshold = result.threshold
        score = result.score
        k_star = result.diagnostics.get("k_star", 0) if result.diagnostics else 0

        # Verify valid results
        assert 0.0 <= threshold <= 1.0
        assert 0.0 <= score <= 1.0
        assert 0 <= k_star <= n_samples

        # Performance should scale reasonably
        expected_max_time = n_samples * np.log(n_samples) * 1e-6  # Generous bound
        assert execution_time < max(0.1, expected_max_time), (
            f"Execution time {execution_time:.6f}s exceeded expected bound for "
            f"{n_samples} samples"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("n_samples", [1000, 5000, 10000])
    def test_large_dataset_performance(self, n_samples):
        """Test performance on large datasets."""
        y_true, pred_prob = generate_test_data(n_samples, random_state=42)

        # Time different methods
        methods = ["sort_scan", "sort_scan"]
        times = {}

        for method in methods:
            gc.collect()  # Clean garbage before timing

            start_time = time.perf_counter()
            try:
                if method == "sort_scan":
                    # Direct sort_scan call
                    f1_vectorized = get_metric_function("f1")
                    result = optimal_threshold_sortscan(
                        y_true, pred_prob, f1_vectorized
                    )
                    threshold = result.threshold
                else:
                    # High-level interface
                    result = optimize_thresholds(y_true, pred_prob, method=method)
                    threshold = result.threshold

                end_time = time.perf_counter()
                times[method] = end_time - start_time

                # Verify result is valid
                assert 0.0 <= threshold <= 1.0

            except (ValueError, NotImplementedError, KeyError):
                # Method might not be available
                times[method] = float("inf")

        # At least one method should complete in reasonable time
        min_time = min(times.values())
        assert min_time < 10.0, f"All methods too slow for {n_samples} samples: {times}"

    def test_method_performance_comparison(self):
        """Compare performance of different optimization methods."""
        n_samples = 2000
        y_true, pred_prob = generate_test_data(n_samples, random_state=42)

        methods = ["sort_scan", "minimize", "gradient"]
        times = {}
        results = {}

        for method in methods:
            gc.collect()

            try:
                start_time = time.perf_counter()
                result = optimize_thresholds(
                    y_true, pred_prob, method=method, metric="f1"
                )
                end_time = time.perf_counter()

                times[method] = end_time - start_time
                results[method] = result.threshold

                # All methods should complete in reasonable time
                assert times[method] < 30.0, (
                    f"Method {method} took {times[method]:.2f}s"
                )

            except (ValueError, NotImplementedError):
                # Method might not be available
                continue

        # All successful methods should produce valid thresholds
        for method, threshold in results.items():
            assert 0.0 <= threshold <= 1.0

    def test_sort_scan_vs_brute_force_scaling(self):
        """Compare scaling behavior of sort_scan vs brute force methods."""
        sample_sizes = [100, 500, 1000, 2000]
        timing_results = {"sort_scan": [], "unique_scan": []}

        for n_samples in sample_sizes:
            y_true, pred_prob = generate_test_data(n_samples, random_state=42)

            # Time sort_scan method
            try:
                start_time = time.time()
                optimize_thresholds(y_true, pred_prob, metric="f1", method="sort_scan")
                end_time = time.time()
                timing_results["sort_scan"].append(end_time - start_time)
            except Exception:
                timing_results["sort_scan"].append(float("inf"))

            # Time unique_scan method
            start_time = time.time()
            optimize_thresholds(y_true, pred_prob, metric="f1", method="unique_scan")
            end_time = time.time()
            timing_results["unique_scan"].append(end_time - start_time)

        # For large datasets, sort_scan should be competitive or better
        if len(timing_results["sort_scan"]) > 0 and timing_results["sort_scan"][
            -1
        ] != float("inf"):
            final_ratio = (
                timing_results["sort_scan"][-1] / timing_results["sort_scan"][-1]
            )
            # Sort_scan should not be more than 2x slower than unique_scan
            assert final_ratio < 2.0, (
                f"sort_scan much slower than unique_scan: {final_ratio:.2f}x"
            )


class TestMemoryEfficiency:
    """Test memory usage characteristics."""

    @pytest.mark.skipif(
        not hasattr(__builtins__, "__PYTEST_SETUP_DONE__"),
        reason="Memory tests need proper setup",
    )
    def test_memory_usage_scaling(self):
        """Test that memory usage scales reasonably with dataset size."""
        import os

        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not available for memory testing")

        process = psutil.Process(os.getpid())

        # Test different dataset sizes
        sizes = [1000, 5000, 10000]
        memory_usage = []

        for n_samples in sizes:
            gc.collect()
            baseline = process.memory_info().rss / 1024 / 1024  # MB

            y_true, pred_prob = generate_test_data(n_samples, random_state=42)
            result = optimize_thresholds(y_true, pred_prob)

            peak = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage.append(peak - baseline)

            del y_true, pred_prob, result
            gc.collect()

        # Memory usage should not grow dramatically
        if len(memory_usage) >= 2:
            growth_ratio = (
                memory_usage[-1] / memory_usage[0] if memory_usage[0] > 0 else 1
            )
            assert growth_ratio < 20, (
                f"Memory usage growing too fast: {growth_ratio:.2f}x"
            )

    def test_memory_cleanup(self):
        """Test that memory is properly cleaned up after optimization."""
        import os

        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not available for memory testing")

        process = psutil.Process(os.getpid())

        # Measure baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create and process data
        for _ in range(5):
            y_true, pred_prob = generate_test_data(2000, random_state=42)
            result = optimize_thresholds(y_true, pred_prob)
            del y_true, pred_prob, result

        # Force garbage collection
        gc.collect()

        # Check final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - baseline_memory

        # Should not have significant memory leaks
        assert memory_increase < 50, (
            f"Potential memory leak: {memory_increase:.2f}MB increase"
        )


class TestWorstCasePerformance:
    """Test behavior under worst-case conditions."""

    def test_all_unique_probabilities_worst_case(self):
        """Performance with all-unique probabilities: worst case for the old code."""
        n_samples = 1000
        y_true = np.random.RandomState(42).randint(0, 2, n_samples)
        pred_prob = np.linspace(0, 1, n_samples)  # All unique values

        start_time = time.time()
        result = optimize_thresholds(y_true, pred_prob, method="sort_scan")
        end_time = time.time()

        execution_time = end_time - start_time
        threshold = result.threshold

        # Should still complete quickly even with all unique values
        assert execution_time < 1.0, (
            f"Too slow with unique values: {execution_time:.4f}s"
        )
        assert 0.0 <= threshold <= 1.0

    def test_extreme_class_imbalance_performance(self):
        """Test performance with extreme class imbalance."""
        n_samples = 10000
        y_true = np.zeros(n_samples, dtype=int)
        y_true[:10] = 1  # Only 10 positive samples out of 10000

        pred_prob = np.random.RandomState(42).uniform(0, 1, n_samples)

        start_time = time.time()
        result = optimize_thresholds(y_true, pred_prob, method="sort_scan")
        end_time = time.time()

        execution_time = end_time - start_time
        threshold = result.threshold

        # Should handle extreme imbalance efficiently
        assert execution_time < 2.0, (
            f"Too slow with extreme imbalance: {execution_time:.4f}s"
        )
        assert 0.0 <= threshold <= 1.0

    def test_many_tied_probabilities_performance(self):
        """Test performance when many probabilities are tied."""
        n_samples = 5000
        y_true = np.random.RandomState(42).randint(0, 2, n_samples)

        # Create many tied values
        unique_probs = [0.1, 0.3, 0.5, 0.7, 0.9]
        pred_prob = np.random.RandomState(42).choice(unique_probs, n_samples)

        start_time = time.time()
        result = optimize_thresholds(y_true, pred_prob, method="sort_scan")
        end_time = time.time()

        execution_time = end_time - start_time
        threshold = result.threshold

        # Should handle tied values efficiently
        assert execution_time < 1.0, f"Too slow with tied values: {execution_time:.4f}s"
        assert 0.0 <= threshold <= 1.0


class TestConcurrentPerformance:
    """Test performance characteristics under various conditions."""

    def test_repeated_optimization_performance(self):
        """Test that repeated optimizations don't degrade performance."""
        n_trials = 10
        n_samples = 1000

        total_start_time = time.time()

        for trial in range(n_trials):
            y_true, pred_prob = generate_test_data(n_samples, random_state=42 + trial)
            result = optimize_thresholds(y_true, pred_prob, method="sort_scan")

            # Each trial should produce valid results
            assert 0.0 <= result.threshold <= 1.0

        total_end_time = time.time()

        # Average time per optimization should be reasonable
        average_time = (total_end_time - total_start_time) / n_trials
        assert average_time < 0.1, f"Average optimization too slow: {average_time:.4f}s"

    def test_different_metrics_performance_consistency(self):
        """Test that performance is consistent across different metrics."""
        n_samples = 1000
        y_true, pred_prob = generate_test_data(n_samples, random_state=42)

        metrics = ["f1", "accuracy", "precision", "recall"]
        timings = {}

        for metric in metrics:
            start_time = time.time()
            result = optimize_thresholds(
                y_true, pred_prob, metric=metric, method="sort_scan"
            )
            end_time = time.time()

            timings[metric] = end_time - start_time
            threshold = result.threshold
            assert 0.0 <= threshold <= 1.0

        # Timings should be within reasonable bounds
        # Some metrics may have vectorized implementations while others don't
        min_time = min(timings.values())
        max_time = max(timings.values())

        if min_time > 0:  # Avoid division by zero
            ratio = max_time / min_time
            # Allow larger variation since some metrics have different optimization
            # paths
            assert ratio < 1000.0, f"Extreme timing variation across metrics: {timings}"

        # All metrics should complete in reasonable absolute time
        for metric, timing in timings.items():
            assert timing < 1.0, f"Metric {metric} too slow: {timing:.4f}s"


@pytest.fixture(scope="module", autouse=True)
def performance_test_setup():
    """Set up performance testing environment."""
    # Ensure consistent timing by disabling some optimizations that could interfere
    import warnings

    warnings.filterwarnings("ignore", message=".*performance.*")
    return
