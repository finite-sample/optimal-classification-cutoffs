"""validation.py - Simple, direct validation with fail-fast semantics."""

import logging

import numpy as np
from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger(__name__)

# ============================================================================
# Core Validation - Simple functions that return clean arrays
# ============================================================================


def validate_binary_labels(labels: ArrayLike) -> NDArray[np.int8]:
    """Validate and return binary labels as int8 array.

    Args:
        labels: Input labels in {0, 1}

    Returns:
        Validated binary labels

    Raises:
        ValueError: If labels are not binary or array is invalid, contain NaN/inf
            values,
            or cannot be converted to int8
    """
    # Handle None input gracefully
    if labels is None:
        raise ValueError("Labels cannot be None")

    # First check for NaN/inf in input before conversion
    temp_arr = np.asarray(labels, dtype=float)
    if np.any(np.isnan(temp_arr)):
        raise ValueError("cannot convert float NaN to integer")
    if np.any(np.isinf(temp_arr)):
        raise ValueError("cannot convert float inf to integer")

    # Try conversion to int8, but catch potential overflow/conversion errors
    try:
        arr = np.asarray(labels, dtype=np.int8)
    except (ValueError, OverflowError) as e:
        raise ValueError(f"Cannot convert labels to int8: {e}") from e

    if arr.ndim != 1:
        raise ValueError(f"Labels must be 1D, got shape {arr.shape}")

    if arr.size == 0:
        raise ValueError("Labels cannot be empty")

    # Check for any NaN or inf values after conversion
    if not np.isfinite(arr).all():
        raise ValueError("Labels contain non-finite values (NaN or inf)")

    unique = np.unique(arr)
    if not (
        np.array_equal(unique, [0, 1])
        or np.array_equal(unique, [0])
        or np.array_equal(unique, [1])
    ):
        raise ValueError(f"Labels must be binary (0 or 1), got unique values: {unique}")

    return arr


def validate_multiclass_labels(
    labels: ArrayLike, n_classes: int | None = None
) -> NDArray[np.int32]:
    """Validate and return multiclass labels as int32 array.

    Args:
        labels: Input labels (non-negative integers)
        n_classes: If provided, validate that labels are in [0, n_classes)

    Returns:
        Validated labels

    Raises:
        ValueError: If labels are invalid
    """
    arr = np.asarray(labels, dtype=np.int32)

    if arr.ndim != 1:
        raise ValueError(f"Labels must be 1D, got shape {arr.shape}")

    if arr.size == 0:
        raise ValueError("Labels cannot be empty")

    if np.any(arr < 0):
        raise ValueError(f"Labels must be non-negative, got min {arr.min()}")

    # Check against n_classes if provided
    if n_classes is not None:
        max_label = np.max(arr)
        if max_label >= n_classes:
            raise ValueError(
                f"Found label {max_label} but n_classes={n_classes}. "
                f"Labels must be in [0, {n_classes})"
            )

    return arr


def validate_probabilities(
    probs: ArrayLike, binary: bool = False, require_proba: bool = True
) -> NDArray[np.float64]:
    """Validate and return probabilities or scores as float64 array.

    Args:
        probs: Probabilities or scores
        binary: If True, require 1D array
        require_proba: If True, enforce [0,1] range. If False, allow arbitrary scores.

    Returns:
        Validated probabilities or scores

    Raises:
        ValueError: If probabilities/scores are invalid
    """
    arr = np.asarray(probs, dtype=np.float64)

    if arr.size == 0:
        raise ValueError("Probabilities cannot be empty")

    # Check for NaN/inf
    if not np.all(np.isfinite(arr)):
        if np.any(np.isnan(arr)):
            raise ValueError("Probabilities contains NaN values")
        if np.any(np.isinf(arr)):
            raise ValueError("Probabilities contains infinite values")
        raise ValueError("Probabilities must be finite")

    # Check shape
    if binary:
        if arr.ndim != 1:
            raise ValueError(f"Binary probabilities must be 1D, got shape {arr.shape}")
    else:
        if arr.ndim not in {1, 2}:
            raise ValueError(f"Probabilities must be 1D or 2D, got {arr.ndim}D")

    # Check range [0, 1] if probabilities are required
    if require_proba and (np.any(arr < 0) or np.any(arr > 1)):
        raise ValueError(
            f"Probabilities must be in [0, 1], got range "
            f"[{arr.min():.3f}, {arr.max():.3f}]"
        )

    # For multiclass, warn if rows don't sum to 1
    if arr.ndim == 2 and arr.shape[1] > 1:
        row_sums = np.sum(arr, axis=1)
        if not np.allclose(row_sums, 1.0, rtol=1e-3):
            logger.warning(
                "Probability rows don't sum to 1 (range: [%.3f, %.3f])",
                row_sums.min(),
                row_sums.max(),
            )

    return arr


def validate_weights(weights: ArrayLike, n_samples: int) -> NDArray[np.float64]:
    """Validate and return sample weights as float64 array.

    Args:
        weights: Sample weights (must be non-negative)
        n_samples: Expected number of samples

    Returns:
        Validated weights

    Raises:
        ValueError: If weights are invalid
    """
    arr = np.asarray(weights, dtype=np.float64)

    if arr.ndim != 1:
        raise ValueError(f"Weights must be 1D, got shape {arr.shape}")

    if len(arr) != n_samples:
        raise ValueError(f"Length mismatch: {n_samples} samples vs {len(arr)} weights")

    if not np.all(np.isfinite(arr)):
        if np.any(np.isnan(arr)):
            raise ValueError("Sample weights contain NaN values")
        if np.any(np.isinf(arr)):
            raise ValueError("Sample weights contain infinite values")
        raise ValueError("Sample weights must be finite")

    if np.any(arr < 0):
        raise ValueError("Sample weights must be non-negative")

    if np.sum(arr) == 0:
        raise ValueError("Sample weights sum to zero")

    return arr


def validate_threshold(
    threshold: float | ArrayLike,
    n_classes: int | None = None,
    *,
    allow_epsilon_outside: bool = False,
) -> NDArray[np.float64]:
    """Validate threshold value(s).

    Args:
        threshold: Threshold(s) to validate
        n_classes: For multiclass, expected number of thresholds
        allow_epsilon_outside: If True, allow values slightly outside [0,1] by
            floating-point epsilon.
            Used internally for thresholds that are nudged by nextafter().

    Returns:
        Validated threshold(s)

    Raises:
        ValueError: If thresholds are invalid
    """
    arr = np.atleast_1d(threshold).astype(np.float64)

    if not np.all(np.isfinite(arr)):
        raise ValueError("Thresholds must be finite")

    # Check bounds
    if allow_epsilon_outside:
        # Lenient check: allow tiny epsilon outside [0,1] for internal use
        eps = np.finfo(np.float64).eps * 100
        if np.any(arr < -eps) or np.any(arr > 1 + eps):
            raise ValueError(
                f"Thresholds far outside valid range: "
                f"[{arr.min():.6f}, {arr.max():.6f}]"
            )
    else:
        # Strict check for user-provided thresholds
        if np.any(arr < 0) or np.any(arr > 1):
            raise ValueError(
                f"Thresholds must be in [0, 1], got range "
                f"[{arr.min():.3f}, {arr.max():.3f}]"
            )

    # Check count
    if n_classes is not None and len(arr) != n_classes:
        raise ValueError(f"Expected {n_classes} thresholds, got {len(arr)}")

    return arr


# ============================================================================
# Utility Functions - Common patterns across the codebase
# ============================================================================


def normalize_binary_probabilities(p: np.ndarray) -> np.ndarray:
    """Extract 1D probability array from 1D or 2D input.

    Handles common binary probability formats:
    - (n,) array: returned as-is
    - (n, 2) array: returns p[:, 1] (positive class probability)
    - (n, 1) array: returns p.ravel()

    Args:
        p: Input probabilities, already converted to numpy array

    Returns:
        1D probability array
    """
    if p.ndim == 2 and p.shape[1] == 2:
        return p[:, 1]
    if p.ndim == 2 and p.shape[1] == 1:
        return p.ravel()
    return p


def apply_threshold(
    values: np.ndarray, threshold: float | np.ndarray, comparison: str = ">"
) -> np.ndarray:
    """Apply threshold with configurable comparison operator.

    Args:
        values: Values to threshold
        threshold: Threshold value(s)
        comparison: Comparison operator (">" or ">=")

    Returns:
        Binary predictions (0 or 1) as int32
    """
    if comparison == ">=":
        return (values >= threshold).astype(np.int32)
    return (values > threshold).astype(np.int32)


def get_sample_weights(sample_weight: ArrayLike | None, n_samples: int) -> np.ndarray:
    """Return sample weights, defaulting to uniform if None.

    Validation is delegated to `validate_weights`, whose ValueError (wrong
    length, non-finite or negative entries) propagates unchanged.

    Args:
        sample_weight: Sample weights, or None for uniform weights
        n_samples: Number of samples

    Returns:
        Sample weights as float64 array
    """
    if sample_weight is None:
        return np.ones(n_samples, dtype=np.float64)
    return validate_weights(sample_weight, n_samples)


def apply_thresholds_multiclass(
    probs: np.ndarray,
    thresholds: np.ndarray,
    comparison: str = ">",
) -> np.ndarray:
    """Apply per-class thresholds with argmax fallback for multiclass classification.

    Decision rule: predict the class with highest probability among those exceeding
    their threshold. Falls back to argmax when no class exceeds its threshold.

    Args:
        probs: Probability matrix of shape (n_samples, n_classes)
        thresholds: Per-class thresholds of shape (n_classes,)
        comparison: Comparison operator (">" or ">=")

    Returns:
        Predicted class labels of shape (n_samples,) as int32
    """
    if comparison == ">=":
        valid = probs >= thresholds[None, :]
    else:
        valid = probs > thresholds[None, :]

    masked = np.where(valid, probs, -np.inf)
    predictions = np.argmax(masked, axis=1).astype(np.int32)

    no_valid = ~np.any(valid, axis=1)
    if np.any(no_valid):
        predictions[no_valid] = np.argmax(probs[no_valid], axis=1)

    return predictions


# ============================================================================
# Prediction Factory Functions - DRY pattern for creating predict closures
# ============================================================================


def make_binary_predictor(threshold: float, comparison: str = ">"):
    """Factory for binary prediction functions.

    Creates a prediction function that applies the given threshold with
    the specified comparison operator.

    Args:
        threshold: Decision threshold
        comparison: Comparison operator (">" or ">=")

    Returns:
        Function that takes probabilities and returns binary predictions
    """

    def predict(probs):
        p = normalize_binary_probabilities(np.asarray(probs))
        return apply_threshold(p, threshold, comparison)

    return predict


def make_multiclass_predictor(thresholds: np.ndarray, comparison: str = ">"):
    """Factory for multiclass prediction functions.

    Creates a prediction function that applies per-class thresholds with
    argmax fallback.

    Args:
        thresholds: Per-class thresholds of shape (n_classes,)
        comparison: Comparison operator (">" or ">=")

    Returns:
        Function that takes 2D probabilities and returns class predictions
    """
    thresholds_arr = np.asarray(thresholds, dtype=np.float64)

    def predict(probs):
        p = np.asarray(probs)
        if p.ndim != 2:
            raise ValueError("Multiclass requires 2D probabilities")
        return apply_thresholds_multiclass(p, thresholds_arr, comparison)

    return predict


def make_margin_predictor(thresholds: np.ndarray):
    """Factory for margin-based multiclass prediction functions.

    Creates a prediction function using the margin rule: argmax(p_j - τ_j).
    This ensures exactly one class is predicted per sample (single-label).

    Args:
        thresholds: Per-class thresholds of shape (n_classes,)

    Returns:
        Function that takes 2D probabilities and returns class predictions
    """
    thresholds_arr = np.asarray(thresholds, dtype=np.float64)

    def predict(probs):
        p = np.asarray(probs, dtype=np.float64)
        if p.ndim != 2:
            raise ValueError("Multiclass requires 2D probabilities")
        margins = p - thresholds_arr[None, :]
        return np.argmax(margins, axis=1).astype(np.int32)

    return predict


# ============================================================================
# High-Level Validation - Combine multiple validations
# ============================================================================


def _validate_weights_for_samples(
    weights: ArrayLike | None,
    n_samples: int,
) -> NDArray[np.float64] | None:
    """Helper to validate weights against sample count."""
    if weights is None:
        return None
    return validate_weights(weights, n_samples)


def validate_binary_classification(
    labels: ArrayLike,
    scores: ArrayLike,
    weights: ArrayLike | None = None,
    *,
    require_proba: bool = True,
) -> tuple[NDArray[np.int8], NDArray[np.float64], NDArray[np.float64] | None]:
    """Validate binary classification inputs.

    Args:
        labels: Binary labels (0 or 1)
        scores: Predicted scores/probabilities
        weights: Sample weights
        require_proba: If True, enforce [0,1] range. If False, allow arbitrary scores.

    Returns:
        (labels as int8, scores as float64, weights as float64 or None)

    Raises:
        ValueError: If inputs are invalid or shapes don't match
    """
    # Validate each component
    labels = validate_binary_labels(labels)
    scores = validate_probabilities(scores, binary=True, require_proba=require_proba)

    # Check shapes match
    if len(labels) != len(scores):
        raise ValueError(
            f"Length mismatch: {len(labels)} labels vs {len(scores)} scores"
        )

    # Validate weights if provided
    weights = _validate_weights_for_samples(weights, len(labels))

    return labels, scores, weights


def validate_multiclass_classification(
    labels: ArrayLike,
    probabilities: ArrayLike,
    weights: ArrayLike | None = None,
    *,
    require_proba: bool = True,
) -> tuple[NDArray[np.int32], NDArray[np.float64], NDArray[np.float64] | None]:
    """Validate multiclass classification inputs.

    Args:
        labels: True class labels (integers)
        probabilities: Predicted probabilities (1D or 2D)
        weights: Sample weights
        require_proba: If True, enforce [0,1] range. If False, allow arbitrary scores.

    Returns:
        (labels as int32, probabilities as float64, weights as float64 or None)

    Raises:
        ValueError: If inputs are invalid or shapes don't match
    """
    # Validate probabilities
    probs = validate_probabilities(
        probabilities, binary=False, require_proba=require_proba
    )

    # Determine n_classes from the probability matrix; 1D means binary.
    n_classes = probs.shape[1] if probs.ndim == 2 else 2

    # Validate labels with n_classes constraint
    labels = validate_multiclass_labels(labels, n_classes)

    # Check shapes match
    n_samples = len(labels)
    if probs.ndim == 2 and probs.shape[0] != n_samples:
        raise ValueError(
            f"Shape mismatch: {n_samples} labels vs {probs.shape[0]} probability rows"
        )
    if probs.ndim == 1 and len(probs) != n_samples:
        raise ValueError(
            f"Length mismatch: {n_samples} labels vs {len(probs)} probabilities"
        )

    # Validate weights if provided
    weights = _validate_weights_for_samples(weights, n_samples)

    return labels, probs, weights


# ============================================================================
# Convenience Functions
# ============================================================================


def infer_problem_type(predictions: ArrayLike) -> str:
    """Infer whether this is binary or multiclass from predictions shape.

    Args:
        predictions: Predicted probabilities

    Returns:
        "binary" or "multiclass"

    Raises:
        ValueError: If shape is invalid
    """
    arr = np.asarray(predictions)

    match arr.ndim:
        case 1:
            return "binary"
        case 2:
            # All 2D arrays are treated as multiclass
            # (n, k) where k >= 2 is multiclass (OvR threshold optimization)
            return "multiclass"
        case _:
            raise ValueError(
                f"Cannot infer problem type from shape {arr.shape}. "
                f"Expected 1D or 2D array."
            )


def validate_classification(
    labels: ArrayLike,
    predictions: ArrayLike,
    weights: ArrayLike | None = None,
    *,
    require_proba: bool = True,
) -> tuple[NDArray, NDArray, NDArray | None, str]:
    """Validate any classification problem, automatically inferring the type.

    Args:
        labels: True class labels
        predictions: Predicted probabilities
        weights: Sample weights
        require_proba: If True, enforce [0,1] range. If False, allow arbitrary scores.

    Returns:
        (labels, predictions, weights, problem_type)
        where problem_type is "binary" or "multiclass"
    """
    problem_type = infer_problem_type(predictions)

    if problem_type == "binary":
        labels, predictions, weights = validate_binary_classification(
            labels, predictions, weights, require_proba=require_proba
        )
    else:
        labels, predictions, weights = validate_multiclass_classification(
            labels, predictions, weights, require_proba=require_proba
        )

    return labels, predictions, weights, problem_type


def validate_inputs(
    labels: ArrayLike,
    predictions: ArrayLike,
    weights: ArrayLike | None = None,
    *,
    require_binary: bool = False,
    allow_multiclass: bool = True,
    require_proba: bool = True,
) -> tuple[NDArray, NDArray, NDArray | None]:
    """Validate classification inputs with flexible type handling.

    Args:
        labels: True class labels
        predictions: Predicted scores/probabilities
        weights: Sample weights
        require_binary: If True, force binary classification validation
        allow_multiclass: If False, raise error for multiclass inputs
        require_proba: If True, enforce [0,1] range. If False, allow arbitrary scores.

    Returns:
        (labels, predictions, weights) validated and converted

    Raises:
        ValueError: If inputs are invalid or multiclass when not allowed
    """
    if require_binary:
        return validate_binary_classification(
            labels, predictions, weights, require_proba=require_proba
        )

    # Try to infer problem type
    pred_arr = np.asarray(predictions)

    if pred_arr.ndim == 1:
        # Binary case
        return validate_binary_classification(
            labels, predictions, weights, require_proba=require_proba
        )
    if pred_arr.ndim == 2 and allow_multiclass:
        # Multiclass case
        return validate_multiclass_classification(
            labels, predictions, weights, require_proba=require_proba
        )
    raise ValueError(
        f"Invalid prediction array shape: {pred_arr.shape}. "
        f"Multiclass allowed: {allow_multiclass}"
    )


# ============================================================================
# Choice Validators
# ============================================================================


def validate_choice(value: str, choices: set[str], name: str) -> str:
    """Validate that a string is in a set of valid choices.

    Args:
        value: Value to validate
        choices: Valid choices
        name: Name of the parameter (for error message)

    Returns:
        The validated value

    Raises:
        ValueError: If value is not in choices
    """
    if value not in choices:
        sorted_choices = sorted(choices)
        raise ValueError(f"Invalid {name} '{value}'. Must be one of: {sorted_choices}")
    return value


def _validate_metric_name(metric_name: str) -> None:
    """Validate that metric exists in the metric registry.

    Args:
        metric_name: Name of the metric to validate

    Raises:
        TypeError: If metric_name is not a string
        ValueError: If metric is not registered
    """
    if not isinstance(metric_name, str):
        raise TypeError(f"metric must be a string, got {type(metric_name)}")

    from .metrics_core import METRICS

    if metric_name not in METRICS:
        available = sorted(METRICS.keys())
        raise ValueError(
            f"Unknown metric '{metric_name}'. "
            f"Available metrics: {', '.join(available[:10])}"
            + (f" ... ({len(available)} total)" if len(available) > 10 else "")
        )


def _validate_averaging_method(average: str) -> None:
    """Validate averaging method for multiclass metrics."""
    validate_choice(average, {"macro", "micro", "weighted", "none"}, "averaging method")


def _validate_optimization_method(method: str) -> None:
    """Validate optimization method."""
    validate_choice(
        method,
        {"auto", "unique_scan", "sort_scan", "minimize", "gradient", "coord_ascent"},
        "optimization method",
    )


def _validate_comparison_operator(comparison: str) -> None:
    """Validate comparison operator for thresholding."""
    validate_choice(comparison, {">", ">="}, "comparison operator")


def _validate_threshold(threshold: float | ArrayLike) -> None:
    """Validate threshold value(s) - convenience wrapper.

    This is kept for backward compatibility but just calls validate_threshold().
    """
    validate_threshold(threshold)


def _validate_threshold_inputs(
    y_true: ArrayLike,
    pred_proba: ArrayLike,
    threshold: float,
    sample_weight: ArrayLike | None = None,
    comparison: str = ">",
    require_proba: bool = True,
) -> tuple[NDArray[np.int8], NDArray[np.float64], NDArray[np.float64] | None]:
    """Unified validation helper for threshold-based operations.

    This function consolidates common validation patterns used across
    confusion matrix computation and threshold optimization functions.

    Args:
        y_true: True binary labels
        pred_proba: Predicted probabilities or scores
        threshold: Decision threshold
        sample_weight: Sample weights
        comparison: Comparison operator
        require_proba: Whether to enforce [0,1] probability range

    Returns:
        (validated_labels, validated_proba, validated_weights)

    Raises:
        ValueError: If any input validation fails
    """
    # Validate labels and probabilities
    y_true_val, pred_proba_val, sample_weight_val = validate_binary_classification(
        y_true, pred_proba, sample_weight, require_proba=require_proba
    )

    # Validate threshold
    if require_proba:
        validate_threshold(threshold, allow_epsilon_outside=False)
    else:
        # For scores, just check it's a finite number
        if not np.isfinite(threshold):
            raise ValueError(f"Threshold must be finite, got {threshold}")

    # Validate comparison operator
    _validate_comparison_operator(comparison)

    return y_true_val, pred_proba_val, sample_weight_val
