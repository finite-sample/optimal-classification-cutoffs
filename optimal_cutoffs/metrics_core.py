"""Metric registry, confusion matrix utilities, and built-in metrics."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .validation import (
    _validate_comparison_operator,
    validate_binary_classification,
    validate_multiclass_classification,
    validate_threshold,
)

# ============================================================================
# Metric Registry
# ============================================================================

type NumericValue = np.ndarray | float
type MetricFunction = Callable[
    [NumericValue, NumericValue, NumericValue, NumericValue], NumericValue
]


@dataclass
class MetricInfo:
    """Complete information about a metric."""

    fn: MetricFunction
    is_piecewise: bool = True
    maximize: bool = True
    needs_proba: bool = False


# Metrics registry
METRICS: dict[str, MetricInfo] = {}


def register_metric(
    name: str | None = None,
    func: MetricFunction | None = None,
    is_piecewise: bool = True,
    maximize: bool = True,
    needs_proba: bool = False,
) -> MetricFunction | Callable[[MetricFunction], MetricFunction]:
    """Register a metric function.

    Parameters
    ----------
    name
        Key under which to store the metric. If not provided, uses function's __name__.
    func
        Metric callable accepting (tp, tn, fp, fn) as scalars or arrays.
        Handles both scalar and array inputs via NumPy broadcasting.
    is_piecewise
        Whether metric is piecewise-constant w.r.t. threshold changes.
    maximize
        Whether to maximize (True) or minimize (False) the metric.
    needs_proba
        Whether metric requires probability scores (e.g., log-loss, Brier score).

    Returns
    -------
    callable or decorator
        The registered function or a decorator if func is None.
    """
    if func is not None:
        metric_name = name or func.__name__
        METRICS[metric_name] = MetricInfo(
            fn=func,
            is_piecewise=is_piecewise,
            maximize=maximize,
            needs_proba=needs_proba,
        )
        # Clear cache when registry changes
        get_metric_function.cache_clear()
        return func

    def decorator(
        f: Callable[
            [
                np.ndarray | float,
                np.ndarray | float,
                np.ndarray | float,
                np.ndarray | float,
            ],
            np.ndarray | float,
        ],
    ) -> Callable[
        [
            np.ndarray | float,
            np.ndarray | float,
            np.ndarray | float,
            np.ndarray | float,
        ],
        np.ndarray | float,
    ]:
        metric_name = name or f.__name__
        METRICS[metric_name] = MetricInfo(
            fn=f,
            is_piecewise=is_piecewise,
            maximize=maximize,
            needs_proba=needs_proba,
        )
        get_metric_function.cache_clear()
        return f

    return decorator


def register_alias(alias_name: str, target_name: str) -> None:
    """Register an alias for an existing metric.

    Parameters
    ----------
    alias_name
        The alias name to register.
    target_name
        The name of the existing metric to point to.

    Raises
    ------
    ValueError
        If the target metric doesn't exist.
    """
    if target_name not in METRICS:
        available = sorted(METRICS.keys())
        preview = ", ".join(available[:10])
        raise ValueError(
            f"Target metric '{target_name}' not found. Available: [{preview}]"
        )
    METRICS[alias_name] = METRICS[target_name]
    get_metric_function.cache_clear()


def register_metrics(
    metrics: dict[
        str,
        Callable[
            [
                np.ndarray | float,
                np.ndarray | float,
                np.ndarray | float,
                np.ndarray | float,
            ],
            np.ndarray | float,
        ],
    ],
    is_piecewise: bool = True,
    maximize: bool = True,
    needs_proba: bool = False,
) -> None:
    """Register multiple metric functions at once.

    Parameters
    ----------
    metrics
        Mapping of metric names to functions that handle both scalars and arrays.
    is_piecewise
        Whether metrics are piecewise-constant.
    maximize
        Whether metrics should be maximized.
    needs_proba
        Whether metrics require probability scores.
    """
    for name, metric_fn in metrics.items():
        METRICS[name] = MetricInfo(
            fn=metric_fn,
            is_piecewise=is_piecewise,
            maximize=maximize,
            needs_proba=needs_proba,
        )
    get_metric_function.cache_clear()


@lru_cache(maxsize=128)
def get_metric_function(metric_name: str) -> Callable[..., Any]:
    """Get metric function with caching for hot paths.

    Parameters
    ----------
    metric_name
        Name of the metric.

    Returns
    -------
    callable
        The metric function that handles both scalar and array inputs.

    Raises
    ------
    ValueError
        If metric doesn't exist.
    """
    if metric_name not in METRICS:
        available = sorted(METRICS.keys())
        preview = ", ".join(available[:10])
        raise ValueError(
            f"Unknown metric '{metric_name}'. "
            f"Available (first 10): [{preview}] ... ({len(available)} total)"
        )

    info = METRICS[metric_name]
    return info.fn


def is_piecewise_metric(metric_name: str) -> bool:
    """Check if a metric is piecewise-constant."""
    return METRICS.get(metric_name, MetricInfo(fn=lambda *_: 0.0)).is_piecewise


def should_maximize_metric(metric_name: str) -> bool:
    """Check if a metric should be maximized."""
    return METRICS.get(metric_name, MetricInfo(fn=lambda *_: 0.0)).maximize


def needs_probability_scores(metric_name: str) -> bool:
    """Check if a metric needs probability scores."""
    return METRICS.get(metric_name, MetricInfo(fn=lambda *_: 0.0)).needs_proba


def has_vectorized_implementation(metric_name: str) -> bool:
    """Check if a metric has a vectorized implementation.

    Note: Always returns True since all metrics handle both scalar and array inputs.
    """
    return metric_name in METRICS


# ============================================================================
# Metric Computation Helpers
# ============================================================================


def _safe_div(
    numerator: np.ndarray | float, denominator: np.ndarray | float
) -> np.ndarray | float:
    """Safe division that returns 0 when denominator is 0, handles inf/nan cases.

    This function provides safe division for metric computation where:
    - Division by zero returns 0.0 (common convention for precision/recall)
    - Division by negative numbers works normally
    - Handles inf/nan cases appropriately
    """
    if isinstance(numerator, np.ndarray) or isinstance(denominator, np.ndarray):
        num = np.asarray(numerator, dtype=float)
        den = np.asarray(denominator, dtype=float)

        # Use numpy's divide with proper handling of zero denominators
        result = np.zeros_like(num, dtype=float)
        valid_mask = den != 0

        # Perform division only where denominator is non-zero
        result = np.divide(num, den, out=result, where=valid_mask)

        # Handle any remaining inf/nan values (e.g., from inf/inf)
        result = np.where(np.isfinite(result), result, 0.0)

        return result
    else:
        # Scalar case
        if denominator == 0:
            return 0.0
        scalar_result = numerator / denominator
        # Handle inf/nan cases in scalar arithmetic
        return scalar_result if np.isfinite(scalar_result) else 0.0


# ============================================================================
# Metric Implementations (Handle Both Scalars and Arrays)
# ============================================================================


def f1_score(
    tp: np.ndarray | float,
    tn: np.ndarray | float,
    fp: np.ndarray | float,
    fn: np.ndarray | float,
) -> np.ndarray | float:
    """F1 score: 2*TP / (2*TP + FP + FN).

    Automatically handles both scalar and array inputs via NumPy broadcasting.
    """
    return _safe_div(2 * tp, 2 * tp + fp + fn)


def accuracy_score(
    tp: np.ndarray | float,
    tn: np.ndarray | float,
    fp: np.ndarray | float,
    fn: np.ndarray | float,
) -> np.ndarray | float:
    """Accuracy: (TP + TN) / (TP + TN + FP + FN)."""
    return _safe_div(tp + tn, tp + tn + fp + fn)


def precision_score(
    tp: np.ndarray | float,
    tn: np.ndarray | float,
    fp: np.ndarray | float,
    fn: np.ndarray | float,
) -> np.ndarray | float:
    """Precision: TP / (TP + FP)."""
    return _safe_div(tp, tp + fp)


def recall_score(
    tp: np.ndarray | float,
    tn: np.ndarray | float,
    fp: np.ndarray | float,
    fn: np.ndarray | float,
) -> np.ndarray | float:
    """Recall: TP / (TP + FN)."""
    return _safe_div(tp, tp + fn)


def iou_score(
    tp: np.ndarray | float,
    tn: np.ndarray | float,
    fp: np.ndarray | float,
    fn: np.ndarray | float,
) -> np.ndarray | float:
    """IoU/Jaccard: TP / (TP + FP + FN)."""
    return _safe_div(tp, tp + fp + fn)


def specificity_score(
    tp: np.ndarray | float,
    tn: np.ndarray | float,
    fp: np.ndarray | float,
    fn: np.ndarray | float,
) -> np.ndarray | float:
    """Specificity: TN / (TN + FP)."""
    return _safe_div(tn, tn + fp)


# ============================================================================
# Confusion Matrix Computation
# ============================================================================


def confusion_matrix_from_predictions(
    true_labels: ArrayLike,
    pred_labels: ArrayLike,
    sample_weight: ArrayLike | None = None,
) -> tuple[float, float, float, float]:
    """Compute confusion matrix from binary predictions (no thresholding).

    This is the canonical single-pass implementation used throughout the codebase.
    Uses optimized bincount approach (4x faster than boolean masking).

    Parameters
    ----------
    true_labels
        True binary labels (0 or 1)
    pred_labels
        Predicted binary labels (0 or 1)
    sample_weight
        Sample weights. If None, uniform weights are used.

    Returns
    -------
    tuple[float, float, float, float]
        (tp, tn, fp, fn) - Always returns floats for consistency.

    Examples
    --------
    >>> true = [0, 1, 0, 1, 1]
    >>> pred = [0, 1, 1, 1, 0]
    >>> tp, tn, fp, fn = confusion_matrix_from_predictions(true, pred)
    >>> (tp, tn, fp, fn)
    (2.0, 1.0, 1.0, 1.0)
    """
    from .validation import get_sample_weights

    true_labels = np.asarray(true_labels, dtype=np.int8)
    pred_labels = np.asarray(pred_labels, dtype=np.int8)
    weights = get_sample_weights(sample_weight, len(true_labels))

    # Single-pass optimization: 2-bit encoding (true*2 + pred)
    # 0=TN, 1=FP, 2=FN, 3=TP
    combined = true_labels * 2 + pred_labels
    counts = np.bincount(combined, weights=weights, minlength=4)

    return float(counts[3]), float(counts[0]), float(counts[1]), float(counts[2])


def confusion_matrix_at_threshold(
    true_labels: ArrayLike,
    pred_proba: ArrayLike,
    threshold: float,
    sample_weight: ArrayLike | None = None,
    comparison: str = ">",
    *,
    require_proba: bool = True,
) -> tuple[float, float, float, float]:
    """Compute confusion matrix by applying threshold to probabilities.

    Parameters
    ----------
    true_labels
        True binary labels in {0, 1}.
    pred_proba
        Predicted probabilities in [0, 1] (if require_proba=True) or scores.
    threshold
        Decision threshold.
    sample_weight
        Sample weights.
    comparison
        Comparison operator for thresholding.
    require_proba
        If True, enforce [0,1] range. If False, allow arbitrary scores.

    Returns
    -------
    tuple[float, float, float, float]
        (tp, tn, fp, fn) - Always returns floats for consistency.
    """
    # Validate inputs
    true_labels, pred_proba, sample_weight = validate_binary_classification(
        true_labels, pred_proba, sample_weight, require_proba=require_proba
    )

    # Validate threshold
    if require_proba:
        validate_threshold(threshold, allow_epsilon_outside=False)

    _validate_comparison_operator(comparison)

    # Apply threshold
    from .validation import apply_threshold

    pred_labels = apply_threshold(pred_proba, threshold, comparison).astype(np.int8)

    # Use optimized computation
    return confusion_matrix_from_predictions(true_labels, pred_labels, sample_weight)


def compute_vectorized_confusion_matrices(
    y_sorted: NDArray[np.int8], weights_sorted: NDArray[np.float64]
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Compute confusion matrices for all possible thresholds using cumulative sums.

    Given labels and weights sorted by descending probabilities, returns
    (tp, tn, fp, fn) as vectors for every possible cut k.

    Convention:
    - Index 0: Predict nothing as positive (all negative predictions)
    - Index k (k > 0): Predict first k items as positive

    Parameters
    ----------
    y_sorted
        Binary labels sorted by descending probability.
    weights_sorted
        Sample weights sorted by descending probability.

    Returns
    -------
    tuple[NDArray[np.float64], ...]
        Arrays of (tp, tn, fp, fn) for each threshold. Length is n+1.
    """
    # Total positive and negative weights
    P = float(np.sum(weights_sorted * y_sorted))
    N = float(np.sum(weights_sorted * (1 - y_sorted)))

    # Cumulative sums
    tp_cumsum = np.cumsum(weights_sorted * y_sorted)
    fp_cumsum = np.cumsum(weights_sorted * (1 - y_sorted))

    # Include "predict nothing" at index 0
    tp = np.concatenate([[0.0], tp_cumsum])
    fp = np.concatenate([[0.0], fp_cumsum])

    # Complement counts
    fn = P - tp
    tn = N - fp

    return tp, tn, fp, fn


def apply_metric_to_confusion_counts(
    metric_fn: Callable[[NDArray, NDArray, NDArray, NDArray], NDArray],
    tp: NDArray[np.float64],
    tn: NDArray[np.float64],
    fp: NDArray[np.float64],
    fn: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply vectorized metric function to confusion matrix counts.

    Parameters
    ----------
    metric_fn
        Vectorized metric accepting (tp, tn, fp, fn) arrays.
    tp
        True positives array.
    tn
        True negatives array.
    fp
        False positives array.
    fn
        False negatives array.

    Returns
    -------
    NDArray[np.float64]
        Array of metric scores.

    Raises
    ------
    ValueError
        If metric function returns wrong shape.
    """
    scores = metric_fn(tp, tn, fp, fn)
    scores = np.asarray(scores)

    if scores.shape != tp.shape:
        raise ValueError(
            f"metric_fn must return array with shape {tp.shape}, got {scores.shape}"
        )

    return scores


# ============================================================================
# Multiclass Helper Functions
# ============================================================================


def compute_exclusive_predictions(
    pred_prob: np.ndarray,
    thresholds: np.ndarray,
    comparison: str = ">",
) -> np.ndarray:
    """Predict class with highest margin: argmax(p_j - tau_j).

    Falls back to argmax(p_j) when no class exceeds its threshold.

    Note: Margin-based decisions can select lower-probability classes with
    better margins. See documentation for details on decision rules.

    Parameters
    ----------
    pred_prob
        Predicted probabilities (n_samples, n_classes)
    thresholds
        Per-class thresholds (n_classes,)
    comparison
        Comparison operator (">" or ">=")

    Returns
    -------
    np.ndarray
        Predicted class labels (n_samples,)
    """
    margins = pred_prob - thresholds  # broadcast
    mask = margins > 0 if comparison == ">" else margins >= 0

    # Argmax of margins where valid; -inf elsewhere
    masked_margins = np.where(mask, margins, -np.inf)
    best_by_margin = np.argmax(masked_margins, axis=1)
    any_above = np.any(mask, axis=1)
    best_by_prob = np.argmax(pred_prob, axis=1)

    return np.where(any_above, best_by_margin, best_by_prob)


def ovr_confusion_counts(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_label: int,
    weights: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Compute TP, FP, FN for one class vs rest (skip TN for efficiency).

    Parameters
    ----------
    true_labels
        True class labels
    pred_labels
        Predicted class labels
    class_label
        Class to compute metrics for
    weights
        Sample weights

    Returns
    -------
    tuple[float, float, float]
        (tp, fp, fn) - TN not computed as it's often unused in OvR metrics
    """
    if weights is None:
        weights = np.ones_like(true_labels, dtype=float)

    true_positive_mask = true_labels == class_label
    pred_positive_mask = pred_labels == class_label

    tp = float(np.sum(weights[true_positive_mask & pred_positive_mask]))
    fp = float(np.sum(weights[~true_positive_mask & pred_positive_mask]))
    fn = float(np.sum(weights[true_positive_mask & ~pred_positive_mask]))

    return tp, fp, fn


# ============================================================================
# High-Level Metric Computation
# ============================================================================


def compute_metric_at_threshold(
    true_labels: ArrayLike,
    pred_proba: ArrayLike,
    threshold: float,
    metric: str = "f1",
    sample_weight: ArrayLike | None = None,
    comparison: str = ">",
) -> float:
    """Compute metric score at a given threshold.

    Parameters
    ----------
    true_labels
        True binary labels
    pred_proba
        Predicted probabilities
    threshold
        Decision threshold
    metric
        Metric name (must be registered)
    sample_weight
        Sample weights
    comparison
        Comparison operator

    Returns
    -------
    float
        Metric score at the threshold
    """
    tp, tn, fp, fn = confusion_matrix_at_threshold(
        true_labels, pred_proba, threshold, sample_weight, comparison
    )

    metric_func = get_metric_function(metric)
    return float(metric_func(tp, tn, fp, fn))


def multiclass_metric_single_label(
    true_labels: ArrayLike,
    pred_proba: ArrayLike,
    thresholds: ArrayLike,
    metric_name: str,
    comparison: str = ">",
    sample_weight: ArrayLike | None = None,
) -> float:
    """Compute exclusive single-label multiclass metrics.

    Uses margin-based decision rule: predict class with highest margin (p_j - tau_j).
    Computes sample-level accuracy or macro-averaged precision/recall/F1.

    Parameters
    ----------
    true_labels
        True class labels (n_samples,)
    pred_proba
        Predicted probabilities (n_samples, n_classes)
    thresholds
        Per-class thresholds (n_classes,)
    metric_name
        Metric to compute ("accuracy", "f1", "precision", "recall")
    comparison
        Comparison operator
    sample_weight
        Sample weights

    Returns
    -------
    float
        Computed metric value
    """
    true_labels = np.asarray(true_labels)
    pred_proba = np.asarray(pred_proba)
    thresholds = np.asarray(thresholds)

    # Get exclusive predictions
    pred_labels = compute_exclusive_predictions(pred_proba, thresholds, comparison)

    if metric_name == "accuracy":
        # Sample-level accuracy
        correct = true_labels == pred_labels
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            return float(np.average(correct, weights=sample_weight))
        else:
            return float(np.mean(correct))

    # Macro-averaged metrics
    labels = np.unique(true_labels.astype(int))
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

    metric_func = get_metric_function(metric_name)
    per_class = []

    for c in labels:
        true_binary = (true_labels == c).astype(int)
        pred_binary = (pred_labels == c).astype(int)

        tp, tn, fp, fn = confusion_matrix_from_predictions(true_binary, pred_binary, sw)
        # TN not meaningful in macro-averaged OvR, pass 0
        per_class.append(metric_func(tp, 0, fp, fn))

    return float(np.mean(per_class) if per_class else 0.0)


def multiclass_metric_ovr(
    confusion_matrices: list[tuple[float, float, float, float]],
    metric_name: str,
    average: str = "macro",
) -> float | np.ndarray:
    """Compute multiclass metrics from per-class confusion matrices (OvR).

    Parameters
    ----------
    confusion_matrices
        List of per-class (tp, tn, fp, fn) tuples
    metric_name
        Metric name (must be registered)
    average
        Averaging strategy

    Returns
    -------
    float or np.ndarray
        Aggregated metric (float) or per-class scores (array if average="none")

    Raises
    ------
    ValueError
        If metric doesn't support requested averaging or is unknown
    """
    metric_func = get_metric_function(metric_name)

    match average:
        case "macro":
            scores = [metric_func(*cm) for cm in confusion_matrices]
            return float(np.mean(scores))

        case "micro":
            # Sum only TP, FP, FN (TN inflated in OvR)
            total_tp = sum(cm[0] for cm in confusion_matrices)
            total_fp = sum(cm[2] for cm in confusion_matrices)
            total_fn = sum(cm[3] for cm in confusion_matrices)

            # Compute micro metrics directly
            if metric_name == "precision":
                return float(
                    total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
                )
            elif metric_name in ("recall", "sensitivity", "tpr"):
                return float(
                    total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
                )
            elif metric_name == "f1":
                precision = (
                    total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
                )
                recall = (
                    total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
                )
                return float(
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )
            elif metric_name == "accuracy":
                raise ValueError(
                    "Micro-averaged accuracy requires exclusive single-label predictions. "
                    "Use multiclass_metric_single_label() instead."
                )
            else:
                raise ValueError(
                    f"Micro-averaged '{metric_name}' is not defined in OvR. "
                    "Supported: 'precision', 'recall', 'f1'."
                )

        case "weighted":
            scores = []
            supports = []
            for cm in confusion_matrices:
                tp, tn, fp, fn = cm
                scores.append(metric_func(*cm))
                supports.append(tp + fn)  # True positives for this class

            total_support = sum(supports)
            if total_support == 0:
                return 0.0

            weighted_score = (
                sum(
                    score * support
                    for score, support in zip(scores, supports, strict=False)
                )
                / total_support
            )
            return float(weighted_score)

        case "none":
            scores = [metric_func(*cm) for cm in confusion_matrices]
            return np.array(scores)

        case _:
            raise ValueError(
                f"Unknown averaging method: {average}. "
                f"Must be one of: 'macro', 'micro', 'weighted', 'none'."
            )


def compute_multiclass_metrics_from_labels(
    true_labels: ArrayLike,
    pred_labels: ArrayLike,
    metric: str = "f1",
    average: str = "macro",
    sample_weight: ArrayLike | None = None,
    n_classes: int | None = None,
) -> float | np.ndarray:
    """Compute multiclass metrics from true and predicted labels.

    Parameters
    ----------
    true_labels
        True class labels
    pred_labels
        Predicted class labels
    metric
        Metric to compute
    average
        Averaging strategy
    sample_weight
        Sample weights
    n_classes
        Number of classes (inferred if None)

    Returns
    -------
    float or np.ndarray
        Computed metric score
    """
    from .validation import get_sample_weights

    true_labels = np.asarray(true_labels, dtype=int)
    pred_labels = np.asarray(pred_labels, dtype=int)

    if true_labels.shape != pred_labels.shape:
        raise ValueError("true_labels and pred_labels must have same shape")

    weights = get_sample_weights(sample_weight, true_labels.shape[0])

    if n_classes is None:
        n_classes = (
            int(max(true_labels.max(initial=-1), pred_labels.max(initial=-1))) + 1
        )

    # Special case: accuracy (computed directly)
    if metric == "accuracy":
        correct = (true_labels == pred_labels).astype(float)
        return float(np.average(correct, weights=weights))

    # Build OvR confusion matrices
    cms: list[tuple[float, float, float, float]] = []
    for k in range(n_classes):
        true_bin = (true_labels == k).astype(int)
        pred_bin = (pred_labels == k).astype(int)
        tp, tn, fp, fn = confusion_matrix_from_predictions(true_bin, pred_bin, weights)
        cms.append((tp, tn, fp, fn))

    return multiclass_metric_ovr(cms, metric_name=metric, average=average)


def multiclass_confusion_matrices_at_thresholds(
    true_labels: ArrayLike,
    pred_proba: ArrayLike,
    thresholds: ArrayLike,
    sample_weight: ArrayLike | None = None,
    comparison: str = ">",
    *,
    require_proba: bool = False,
) -> list[tuple[float, float, float, float]]:
    """Compute per-class confusion matrices for multiclass (OvR).

    Parameters
    ----------
    true_labels
        True class labels
    pred_proba
        Predicted probabilities (n_samples, n_classes) or scores
    thresholds
        Per-class thresholds
    sample_weight
        Sample weights
    comparison
        Comparison operator
    require_proba
        If True, enforce [0,1] range

    Returns
    -------
    list[tuple[float, float, float, float]]
        List of per-class (tp, tn, fp, fn) tuples
    """
    # Validate inputs
    true_labels, pred_proba, sample_weight = validate_multiclass_classification(
        true_labels, pred_proba, sample_weight, require_proba=require_proba
    )
    _validate_comparison_operator(comparison)

    if pred_proba.ndim == 1:
        # Binary case
        thr_arr = np.asarray(thresholds)
        thr_scalar = float(thr_arr.reshape(-1)[0])
        return [
            confusion_matrix_at_threshold(
                true_labels,
                pred_proba,
                thr_scalar,
                sample_weight,
                comparison,
                require_proba=require_proba,
            )
        ]

    # Multiclass case
    n_classes = pred_proba.shape[1]
    thresholds = np.asarray(thresholds, dtype=float)

    if thresholds.shape != (n_classes,):
        raise ValueError(
            f"thresholds must have shape ({n_classes},), got {thresholds.shape}"
        )

    if require_proba:
        validate_threshold(thresholds, n_classes)

    confusion_matrices = []

    for class_idx in range(n_classes):
        true_binary = (true_labels == class_idx).astype(int)
        pred_binary_proba = pred_proba[:, class_idx]
        threshold = thresholds[class_idx]

        cm = confusion_matrix_at_threshold(
            true_binary,
            pred_binary_proba,
            threshold,
            sample_weight,
            comparison,
            require_proba=require_proba,
        )
        confusion_matrices.append(cm)

    return confusion_matrices


# ============================================================================
# Metric Factory Functions
# ============================================================================


def make_linear_counts_metric(
    w_tp: float = 0.0,
    w_tn: float = 0.0,
    w_fp: float = 0.0,
    w_fn: float = 0.0,
    name: str | None = None,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Create a vectorized linear utility metric from confusion matrix.

    Returns: metric(tp, tn, fp, fn) = w_tp*tp + w_tn*tn + w_fp*fp + w_fn*fn

    Parameters
    ----------
    w_tp
        Weight for true positives
    w_tn
        Weight for true negatives
    w_fp
        Weight for false positives
    w_fn
        Weight for false negatives
    name
        If provided, automatically registers the metric

    Returns
    -------
    callable
        Vectorized metric function

    Examples
    --------
    >>> # Cost-sensitive: FN costs 5x more than FP
    >>> metric = make_linear_counts_metric(w_fp=-1.0, w_fn=-5.0, name="cost_5to1")
    >>> # Now can use: optimize_threshold(y, y_pred, metric="cost_5to1")
    """

    def _metric(
        tp: np.ndarray, tn: np.ndarray, fp: np.ndarray, fn: np.ndarray
    ) -> np.ndarray:
        """Vectorized linear combination of confusion matrix counts."""
        return (
            w_tp * np.asarray(tp, dtype=float)
            + w_tn * np.asarray(tn, dtype=float)
            + w_fp * np.asarray(fp, dtype=float)
            + w_fn * np.asarray(fn, dtype=float)
        )

    # Auto-register if name provided
    if name is not None:
        # Create metric function from vectorized implementation
        def linear_metric(
            tp: np.ndarray | float,
            tn: np.ndarray | float,
            fp: np.ndarray | float,
            fn: np.ndarray | float,
        ) -> np.ndarray | float:
            """Linear counts metric."""
            return _metric(
                np.asarray(tp, dtype=float),
                np.asarray(tn, dtype=float),
                np.asarray(fp, dtype=float),
                np.asarray(fn, dtype=float),
            )

        register_metric(
            name=name,
            func=linear_metric,
            is_piecewise=True,
            maximize=True,
            needs_proba=False,
        )

    return _metric


def make_cost_metric(
    fp_cost: float,
    fn_cost: float,
    tp_benefit: float = 0.0,
    tn_benefit: float = 0.0,
    name: str | None = None,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Create a vectorized cost-sensitive metric.

    Returns: tp_benefit*TP + tn_benefit*TN - fp_cost*FP - fn_cost*FN

    Parameters
    ----------
    fp_cost
        Cost of false positives (positive value)
    fn_cost
        Cost of false negatives (positive value)
    tp_benefit
        Benefit for true positives
    tn_benefit
        Benefit for true negatives
    name
        If provided, automatically registers the metric

    Returns
    -------
    callable
        Vectorized metric function

    Examples
    --------
    >>> # Classic cost-sensitive
    >>> metric = make_cost_metric(fp_cost=1.0, fn_cost=5.0, name="cost_sensitive")
    >>> # Now can use: optimize_threshold(y, y_pred, metric="cost_sensitive")
    """
    return make_linear_counts_metric(
        w_tp=tp_benefit,
        w_tn=tn_benefit,
        w_fp=-fp_cost,
        w_fn=-fn_cost,
        name=name,
    )


# ============================================================================
# Register Built-in Metrics
# ============================================================================

register_metric("f1", f1_score)
register_metric("accuracy", accuracy_score)
register_metric("precision", precision_score)
register_metric("recall", recall_score)
register_metric("iou", iou_score)
register_metric("specificity", specificity_score)

# Register aliases
register_alias("jaccard", "iou")
register_alias("tnr", "specificity")
register_alias("ppv", "precision")
register_alias("tpr", "recall")
register_alias("sensitivity", "recall")
