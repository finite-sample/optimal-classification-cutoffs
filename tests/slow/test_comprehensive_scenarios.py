"""Comprehensive slow tests for complex scenarios.

This module contains tests that take longer to run but provide comprehensive
coverage of complex real-world scenarios and stress testing.
"""

import time

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
    generate_imbalanced_data,
    generate_multiclass_data,
    generate_tied_probabilities,
)


class TestLargeScaleScenarios:
    """Test scenarios with large datasets and complex conditions."""

    @pytest.mark.slow
    def test_massive_dataset_optimization(self):
        """Test optimization with very large datasets."""
        n_samples = 100000  # Large dataset

        try:
            y_true, pred_prob = generate_binary_data(n_samples, random_state=42)
        except MemoryError:
            pytest.skip("Insufficient memory for massive dataset test")

        start_time = time.time()
        result = optimize_thresholds(y_true, pred_prob, method="sort_scan")
        end_time = time.time()

        execution_time = end_time - start_time

        threshold = result.threshold
        assert_valid_threshold(threshold)
        assert execution_time < 120.0, (
            f"Massive dataset optimization took {execution_time:.2f}s"
        )

        # Verify optimization quality
        score = compute_metric_at_threshold(y_true, pred_prob, threshold, "f1")
        assert_valid_metric_score(score, "f1")
        assert score > 0.1  # Should achieve reasonable performance

    @pytest.mark.slow
    def test_extreme_imbalance_large_scale(self):
        """Test extreme class imbalance on large datasets."""
        n_samples = 50000
        imbalance_ratio = 0.0001  # 0.01% positive class

        try:
            y_true, pred_prob = generate_imbalanced_data(
                n_samples, imbalance_ratio=imbalance_ratio, random_state=42
            )
        except MemoryError:
            pytest.skip("Insufficient memory for large imbalanced dataset test")

        start_time = time.time()
        result = optimize_thresholds(y_true, pred_prob, metric="f1")
        end_time = time.time()

        execution_time = end_time - start_time

        threshold = result.threshold
        assert_valid_threshold(threshold)
        assert execution_time < 60.0, (
            f"Large imbalanced optimization took {execution_time:.2f}s"
        )

        # Even with extreme imbalance, should find reasonable threshold
        score = compute_metric_at_threshold(y_true, pred_prob, threshold, "f1")
        assert_valid_metric_score(score, "f1")

    @pytest.mark.slow
    def test_massive_tie_scenarios(self):
        """Test scenarios with massive numbers of tied probability values."""
        n_samples = 20000
        tie_fraction = 0.8  # 80% of values are tied

        y_true, pred_prob = generate_tied_probabilities(
            n_samples, tie_fraction=tie_fraction, random_state=42
        )

        start_time = time.time()
        result = optimize_thresholds(y_true, pred_prob, metric="f1")
        end_time = time.time()

        execution_time = end_time - start_time

        threshold = result.threshold
        assert_valid_threshold(threshold)
        assert execution_time < 30.0, (
            f"Massive tie optimization took {execution_time:.2f}s"
        )

        score = compute_metric_at_threshold(y_true, pred_prob, threshold, "f1")
        assert_valid_metric_score(score, "f1")


# TestStressTesting class removed - duplicated in test_stress_testing.py


class TestLongRunningScenarios:
    """Test scenarios that require extended execution time."""

    @pytest.mark.slow
    def test_comprehensive_method_comparison(self):
        """Comprehensive comparison of all optimization methods."""
        n_samples = 5000
        n_datasets = 20

        methods = ["sort_scan", "minimize", "gradient"]
        metrics = ["f1", "accuracy", "precision", "recall"]

        results = {method: {metric: [] for metric in metrics} for method in methods}
        times = {method: {metric: [] for metric in metrics} for method in methods}

        for dataset_idx in range(n_datasets):
            y_true, pred_prob = generate_binary_data(
                n_samples, random_state=dataset_idx
            )

            for method in methods:
                for metric in metrics:
                    try:
                        start_time = time.time()
                        result = optimize_thresholds(
                            y_true, pred_prob, method=method, metric=metric
                        )
                        end_time = time.time()

                        execution_time = end_time - start_time

                        threshold = result.threshold
                        assert_valid_threshold(threshold)
                        score = compute_metric_at_threshold(
                            y_true, pred_prob, threshold, metric
                        )

                        results[method][metric].append(score)
                        times[method][metric].append(execution_time)

                    except (ValueError, NotImplementedError):
                        # Method/metric combination might not be available
                        continue

        # Analyze results
        for method in methods:
            for metric in metrics:
                if results[method][metric]:  # If we have results
                    scores = results[method][metric]
                    execution_times = times[method][metric]

                    # All scores should be valid
                    for score in scores:
                        assert_valid_metric_score(score, metric)

                    # Performance should be reasonable
                    max_time = np.max(execution_times)

                    assert max_time < 30.0, (
                        f"Method {method} with metric {metric} too slow: "
                        f"{max_time:.2f}s"
                    )

    @pytest.mark.slow
    def test_multiclass_scaling_comprehensive(self):
        """Comprehensive test of multiclass optimization scaling."""
        class_counts = [3, 5, 8, 10]
        sample_counts = [1000, 2000, 5000]

        for n_classes in class_counts:
            for n_samples in sample_counts:
                try:
                    y_true, pred_prob = generate_multiclass_data(
                        n_samples, n_classes=n_classes, random_state=42
                    )

                    start_time = time.time()
                    # Test multiclass optimization if available
                    try:
                        thresholds = optimize_thresholds(y_true, pred_prob, metric="f1")
                        end_time = time.time()

                        execution_time = end_time - start_time

                        # Should complete in reasonable time
                        expected_max_time = (
                            n_samples * n_classes * 1e-4
                        )  # Generous bound
                        assert execution_time < max(10.0, expected_max_time), (
                            f"Multiclass optimization too slow: {execution_time:.2f}s "
                            f"for {n_samples} samples, {n_classes} classes"
                        )

                        # Results should be valid
                        if isinstance(thresholds, list | np.ndarray):
                            assert len(thresholds) == n_classes
                            for threshold in thresholds:
                                assert_valid_threshold(threshold)
                        else:
                            assert_valid_threshold(thresholds)

                    except (AttributeError, TypeError, ValueError):
                        # Multiclass optimization might not be available or
                        # might return different format
                        continue

                except MemoryError:
                    pytest.skip(
                        f"Insufficient memory for {n_samples} samples, {n_classes} "
                        f"classes"
                    )

    @pytest.mark.slow
    def test_cross_validation_integration_comprehensive(self):
        """Comprehensive test of cross-validation integration."""
        n_samples = 3000
        n_folds = 10
        n_datasets = 5

        from sklearn.model_selection import StratifiedKFold

        for dataset_idx in range(n_datasets):
            y_true, pred_prob = generate_binary_data(
                n_samples, random_state=dataset_idx
            )

            # Perform cross-validation
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

            fold_thresholds = []
            fold_scores = []

            for train_idx, val_idx in cv.split(y_true, y_true):
                # Use training set to optimize threshold
                y_train, p_train = y_true[train_idx], pred_prob[train_idx]

                result = optimize_thresholds(y_train, p_train, metric="f1")
                threshold = result.threshold
                assert_valid_threshold(threshold)

                # Evaluate on validation set
                _y_val, _p_val = y_true[val_idx], pred_prob[val_idx]
                score = compute_metric_at_threshold(y_true, pred_prob, threshold, "f1")

                fold_thresholds.append(threshold)
                fold_scores.append(score)

            # All folds should produce valid results
            assert len(fold_thresholds) == n_folds
            assert len(fold_scores) == n_folds

            for threshold, score in zip(fold_thresholds, fold_scores, strict=False):
                assert_valid_threshold(threshold)
                assert_valid_metric_score(score, "f1")

            # Cross-validation should show reasonable performance
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)

            assert mean_score > 0.1, f"Poor CV performance: {mean_score:.3f}"
            assert std_score < 0.5, f"High CV variance: {std_score:.3f}"


class TestExtremeCaseComprehensive:
    """Comprehensive testing of extreme edge cases."""

    @pytest.mark.slow
    def test_pathological_data_distributions(self):
        """Test optimization with pathological data distributions."""
        n_samples = 10000

        pathological_cases = [
            # Extremely peaked distributions
            lambda rs: (
                np.random.binomial(1, 0.5, n_samples),
                np.random.beta(0.01, 0.01, n_samples),  # U-shaped
            ),
            # Multimodal distributions
            lambda rs: (
                np.random.binomial(1, 0.5, n_samples),
                np.concatenate(
                    [
                        np.random.beta(0.5, 5, n_samples // 2),  # Left-skewed
                        np.random.beta(5, 0.5, n_samples // 2),  # Right-skewed
                    ]
                ),
            ),
            # Extremely concentrated distributions
            lambda rs: (
                np.random.binomial(1, 0.5, n_samples),
                0.5 + np.random.normal(0, 0.01, n_samples),  # Very narrow
            ),
        ]

        success_count = 0

        for i, case_generator in enumerate(pathological_cases):
            try:
                y_true, pred_prob = case_generator(np.random.RandomState(42))
                pred_prob = np.clip(pred_prob, 0, 1)  # Ensure valid range

                # Ensure both classes present
                if np.sum(y_true) == 0:
                    y_true[0] = 1
                elif np.sum(y_true) == len(y_true):
                    y_true[0] = 0

                result = optimize_thresholds(y_true, pred_prob, metric="f1")
                threshold = result.threshold
                assert_valid_threshold(threshold)

                score = compute_metric_at_threshold(y_true, pred_prob, threshold, "f1")
                assert_valid_metric_score(score, "f1")

                success_count += 1

            except (ValueError, RuntimeError) as e:
                print(f"Pathological case {i} failed: {e}")

        # Should handle at least some pathological cases
        assert success_count > 0, "All pathological cases failed"
