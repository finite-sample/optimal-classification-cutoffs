"""Custom assertion helpers for test consistency.

This module provides standardized assertion functions to ensure
consistent validation across all test modules.
"""

import numpy as np


def assert_valid_threshold(
    threshold: float | np.ndarray,
    n_classes: int | None = None,
    allow_infinite_range: bool = True,
) -> None:
    """Assert that threshold(s) are valid.

    Parameters
    ----------
    threshold : float or array-like
        Threshold value(s) to validate
    n_classes : int, optional
        Expected number of classes for multiclass thresholds
    allow_infinite_range : bool, default=True
        Whether to allow thresholds outside [0,1]. For coordinate ascent and
        margin-based optimization, thresholds can be outside [0,1] and are
        mathematically correct.
    """
    threshold = np.asarray(threshold)

    # Check for NaN/inf - thresholds must be finite
    assert np.all(np.isfinite(threshold)), (
        f"Threshold {threshold} contains non-finite values"
    )

    # Check range - for traditional methods, thresholds should be in [0,1]
    # For margin-based methods (coordinate ascent), they can be outside [0,1]
    if not allow_infinite_range:
        assert np.all(threshold >= 0.0), f"Threshold {threshold} contains values < 0"
        assert np.all(threshold <= 1.0), f"Threshold {threshold} contains values > 1"

    # Check dimensions for multiclass
    if n_classes is not None:
        if threshold.ndim == 0:
            # Scalar threshold for multiclass is only valid if n_classes=1
            assert n_classes == 1, f"Scalar threshold for {n_classes} classes"
        else:
            assert len(threshold) == n_classes, (
                f"Expected {n_classes} thresholds, got {len(threshold)}"
            )


def assert_valid_confusion_matrix(
    tp: int | float,
    tn: int | float,
    fp: int | float,
    fn: int | float,
    total_samples: int | None = None,
    total_weight: float | None = None,
    tolerance: float = 1e-10,
) -> None:
    """Assert that confusion matrix values are valid.

    Parameters
    ----------
    tp, tn, fp, fn : int or float
        Confusion matrix components
    total_samples : int, optional
        Expected total number of samples
    total_weight : float, optional
        Expected total weight (for weighted confusion matrices)
    tolerance : float
        Tolerance for floating point comparisons
    """
    # Non-negative values
    assert tp >= -tolerance, f"True positives {tp} < 0"
    assert tn >= -tolerance, f"True negatives {tn} < 0"
    assert fp >= -tolerance, f"False positives {fp} < 0"
    assert fn >= -tolerance, f"False negatives {fn} < 0"

    # Finite values
    assert np.isfinite(tp), f"True positives {tp} not finite"
    assert np.isfinite(tn), f"True negatives {tn} not finite"
    assert np.isfinite(fp), f"False positives {fp} not finite"
    assert np.isfinite(fn), f"False negatives {fn} not finite"

    # Check totals if provided
    total = tp + tn + fp + fn
    if total_samples is not None:
        assert abs(total - total_samples) < tolerance, (
            f"Total {total} != expected {total_samples}"
        )

    if total_weight is not None:
        assert abs(total - total_weight) < tolerance, (
            f"Total weight {total} != expected {total_weight}"
        )


def assert_valid_metric_score(
    score: float,
    metric_name: str,
    expected_range: tuple = (0.0, 1.0),
    allow_nan: bool = False,
) -> None:
    """Assert that a metric score is valid.

    Parameters
    ----------
    score : float
        Metric score to validate
    metric_name : str
        Name of the metric for error messages
    expected_range : tuple
        Expected (min, max) range for the metric
    allow_nan : bool
        Whether NaN values are acceptable (e.g., for undefined metrics)
    """
    if allow_nan and np.isnan(score):
        return

    assert np.isfinite(score), f"{metric_name} score {score} is not finite"

    min_val, max_val = expected_range
    assert min_val <= score <= max_val, (
        f"{metric_name} score {score} outside range [{min_val}, {max_val}]"
    )


def assert_monotonic_increase(
    values: np.ndarray, strict: bool = False, tolerance: float = 1e-10
) -> None:
    """Assert that values are monotonically increasing.

    Parameters
    ----------
    values : array-like
        Values to check for monotonicity
    strict : bool
        Whether to require strict monotonicity (no equal consecutive values)
    tolerance : float
        Tolerance for floating point comparisons
    """
    values = np.asarray(values)
    diffs = np.diff(values)

    if strict:
        assert np.all(diffs > tolerance), f"Values not strictly increasing: {values}"
    else:
        assert np.all(diffs >= -tolerance), (
            f"Values not monotonically increasing: {values}"
        )


def assert_arrays_close(
    actual: np.ndarray,
    expected: np.ndarray,
    rtol: float = 1e-7,
    atol: float = 1e-10,
    description: str = "arrays",
) -> None:
    """Assert that two arrays are close within tolerance.

    Parameters
    ----------
    actual, expected : array-like
        Arrays to compare
    rtol : float
        Relative tolerance
    atol : float
        Absolute tolerance
    description : str
        Description for error messages
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)

    assert actual.shape == expected.shape, (
        f"{description} shapes differ: {actual.shape} vs {expected.shape}"
    )

    try:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as e:
        raise AssertionError(f"{description} not close enough: {e}") from e


def assert_probability_matrix_valid(
    probs: np.ndarray,
    n_classes: int | None = None,
    require_normalized: bool = True,
    tolerance: float = 1e-6,
) -> None:
    """Assert that probability matrix is valid.

    Parameters
    ----------
    probs : array-like
        Probability matrix to validate
    n_classes : int, optional
        Expected number of classes
    require_normalized : bool
        Whether rows should sum to 1
    tolerance : float
        Tolerance for sum-to-1 check
    """
    probs = np.asarray(probs)

    # Check 2D
    assert probs.ndim == 2, f"Probability matrix must be 2D, got shape {probs.shape}"

    # Check range [0, 1]
    assert np.all(probs >= 0.0), "Probabilities contain negative values"
    assert np.all(probs <= 1.0), "Probabilities contain values > 1"

    # Check finite
    assert np.all(np.isfinite(probs)), "Probabilities contain non-finite values"

    # Check number of classes
    if n_classes is not None:
        assert probs.shape[1] == n_classes, (
            f"Expected {n_classes} classes, got {probs.shape[1]}"
        )

    # Check normalization
    if require_normalized:
        row_sums = probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=tolerance), (
            f"Probability rows don't sum to 1: {row_sums}"
        )


def assert_labels_valid(
    labels: np.ndarray, n_classes: int | None = None, require_consecutive: bool = False
) -> None:
    """Assert that labels are valid.

    Parameters
    ----------
    labels : array-like
        Labels to validate
    n_classes : int, optional
        Expected number of classes
    require_consecutive : bool
        Whether labels must be consecutive integers from 0
    """
    labels = np.asarray(labels)

    # Check 1D
    assert labels.ndim == 1, f"Labels must be 1D, got shape {labels.shape}"

    # Check integer type
    assert np.issubdtype(labels.dtype, np.integer), (
        f"Labels must be integers, got {labels.dtype}"
    )

    # Check non-negative
    assert np.all(labels >= 0), "Labels contain negative values"

    # Check finite
    assert np.all(np.isfinite(labels)), "Labels contain non-finite values"

    unique_labels = np.unique(labels)

    if n_classes is not None:
        assert np.max(labels) < n_classes, (
            f"Labels {np.max(labels)} >= n_classes {n_classes}"
        )

    if require_consecutive:
        expected_labels = np.arange(len(unique_labels))
        assert np.array_equal(unique_labels, expected_labels), (
            f"Labels must be consecutive from 0, got {unique_labels}"
        )


def assert_optimization_successful(
    threshold: float | np.ndarray,
    metric_score: float,
    metric_name: str,
    min_score: float = 0.0,
    allow_infinite_range: bool = True,
) -> None:
    """Assert that optimization produced reasonable results.

    Parameters
    ----------
    threshold : float or array-like
        Optimized threshold(s)
    metric_score : float
        Achieved metric score
    metric_name : str
        Name of the optimized metric
    min_score : float
        Minimum acceptable score
    allow_infinite_range : bool, default=True
        Whether to allow thresholds outside [0,1]
    """
    assert_valid_threshold(threshold, allow_infinite_range=allow_infinite_range)
    assert_valid_metric_score(metric_score, metric_name)
    assert metric_score >= min_score, (
        f"{metric_name} score {metric_score} below minimum {min_score}"
    )


def assert_method_consistency(
    result1: float | np.ndarray,
    result2: float | np.ndarray,
    method1: str,
    method2: str,
    tolerance: float = 1e-5,
    score_tolerance: float = 0.05,
) -> None:
    """Assert that two optimization methods give consistent results.

    Parameters
    ----------
    result1, result2 : float or array-like
        Results from two methods (thresholds or scores)
    method1, method2 : str
        Names of the methods
    tolerance : float
        Tolerance for threshold comparison
    score_tolerance : float
        Tolerance for score comparison
    """
    result1 = np.asarray(result1)
    result2 = np.asarray(result2)

    # For thresholds, use tight tolerance
    # For scores, use looser tolerance since different methods may find different local
    # optima
    tol = (
        score_tolerance
        if "score" in method1.lower() or "score" in method2.lower()
        else tolerance
    )

    try:
        np.testing.assert_allclose(result1, result2, atol=tol, rtol=tol)
    except AssertionError:
        # Allow for reasonable differences between methods
        max_diff = np.max(np.abs(result1 - result2))
        if max_diff > 2 * tol:
            raise AssertionError(
                f"Methods {method1} and {method2} differ by {max_diff:.6f} "
                f"(tolerance: {tol:.6f}): {result1} vs {result2}"
            ) from None
