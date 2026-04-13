"""Multi-class classification threshold optimization.

This module implements threshold optimization for multi-class classification
where we have K mutually exclusive classes and must predict exactly one class.

Key approaches:
1. OvR Independent: Treat each class as independent binary (multi-label style)
2. Margin Rule: Use argmax(p_j - τ_j) for coupled single-label predictions
3. Micro averaging: Single threshold applied to all classes

The margin rule is Bayes-optimal when costs have OvR structure but requires
coordinate ascent for general metrics like F1.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .core import OptimizationResult, Task
from .validation import validate_multiclass_classification


def optimize_ovr_independent(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    metric: str = "f1",
    method: str = "auto",
    sample_weight: ArrayLike | None = None,
    comparison: str = ">",
    tolerance: float = 1e-10,
) -> OptimizationResult:
    """Optimize multiclass metrics using independent per-class thresholds (OvR).

    Treats each class as an independent binary problem (class vs rest).
    This does NOT enforce single-label predictions - can predict 0, 1, or
    multiple classes. Use this for macro-averaged metrics when you want
    exact optimization per class.

    Decision rule: ŷ_j = 1 if p_j ≥ τ_j (independent for each class)

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels in {0, 1, ..., K-1}
    y_score : array-like of shape (n_samples, n_classes)
        Predicted probabilities for each class
    metric : str, default="f1"
        Metric to optimize per class
    method : str, default="auto"
        Binary optimization method
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights
    comparison : str, default=">"
        Comparison operator
    tolerance : float, default=1e-10
        Numerical tolerance

    Returns
    -------
    OptimizationResult
        Result with per-class thresholds optimized independently

    Examples
    --------
    >>> y_true = [0, 1, 2, 0, 1]
    >>> y_score = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], ...]
    >>> result = optimize_ovr_independent(y_true, y_score, metric="f1")
    >>> predictions = result.predict(y_score)  # Can predict multiple classes
    """
    from .binary import optimize_metric_binary

    # Validate inputs
    y_true, y_score, sample_weight = validate_multiclass_classification(
        y_true, y_score, sample_weight, require_proba=True
    )

    n_samples, n_classes = y_score.shape

    # Optimize each class vs rest independently
    optimal_thresholds = np.zeros(n_classes, dtype=np.float64)
    optimal_scores = np.zeros(n_classes, dtype=np.float64)

    for k in range(n_classes):
        # Create binary problem: class k vs rest
        y_true_k = (y_true == k).astype(int)
        y_score_k = y_score[:, k]

        # Optimize threshold for class k
        result_k = optimize_metric_binary(
            y_true_k,
            y_score_k,
            metric=metric,
            method=method,
            sample_weight=sample_weight,
            comparison=comparison,
            tolerance=tolerance,
        )

        optimal_thresholds[k] = result_k.thresholds[0]
        optimal_scores[k] = result_k.scores[0]

    # Macro average score
    macro_score = np.mean(optimal_scores)

    from .validation import apply_threshold

    def predict_multiclass_independent(probs: ArrayLike) -> np.ndarray:
        """Independent per-class predictions (can predict multiple classes)."""
        p = np.asarray(probs, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] != n_classes:
            raise ValueError(f"Expected probabilities shape (n_samples, {n_classes})")

        return apply_threshold(p, optimal_thresholds[None, :], comparison)

    return OptimizationResult(
        thresholds=optimal_thresholds,
        scores=np.array([macro_score], dtype=np.float64),
        predict=predict_multiclass_independent,
        task=Task.MULTICLASS,
        metric=f"macro_{metric}_ovr_independent",
        n_classes=n_classes,
    )


def optimize_ovr_margin(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    metric: str = "f1",
    max_iter: int = 30,
    sample_weight: ArrayLike | None = None,
    comparison: str = ">",
    tolerance: float = 1e-10,
) -> OptimizationResult:
    """Optimize multiclass metrics using margin rule with coordinate ascent.

    Uses margin-based prediction: ŷ = argmax_j (p_j - τ_j)
    This ensures exactly one class is predicted per sample (single-label).

    Thresholds are coupled because changing τ_j affects which samples are
    assigned to class j, which affects confusion matrices for all classes.
    Uses coordinate ascent to find local optimum.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels in {0, 1, ..., K-1}
    y_score : array-like of shape (n_samples, n_classes)
        Predicted probabilities for each class
    metric : str, default="f1"
        Metric to optimize (currently supports "f1" only)
    max_iter : int, default=30
        Maximum coordinate ascent iterations
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights
    comparison : str, default=">"
        Comparison operator (only ">" supported for margin rule)
    tolerance : float, default=1e-12
        Convergence tolerance

    Returns
    -------
    OptimizationResult
        Result with per-class thresholds optimized via coordinate ascent

    Examples
    --------
    >>> result = optimize_ovr_margin(y_true, y_score, metric="f1")
    >>> predictions = result.predict(y_score)  # Exactly one class per sample

    Notes
    -----
    The margin rule is Bayes-optimal when costs have OvR structure:
    C(i,j) = -r_j if i=j, else c_j

    In this case, optimal thresholds are: τ_j = c_j/(c_j + r_j) (closed form!)
    """
    from .optimize import coordinate_ascent_kernel

    # Validate inputs
    y_true, y_score, sample_weight = validate_multiclass_classification(
        y_true, y_score, sample_weight, require_proba=True
    )

    n_samples, n_classes = y_score.shape

    if metric != "f1":
        raise NotImplementedError("supports 'f1' metric only")

    if comparison != ">":
        raise NotImplementedError("'>' is required")

    # Prepare data for coordinate ascent kernel
    from .validation import get_sample_weights

    y_true_int32 = np.asarray(y_true, dtype=np.int32)
    y_score_float64 = np.asarray(y_score, dtype=np.float64, order="C")
    n_samples = len(y_true_int32)
    weights = get_sample_weights(sample_weight, n_samples)

    # Run coordinate ascent
    thresholds, best_score, history = coordinate_ascent_kernel(
        y_true_int32,
        y_score_float64,
        weights,
        max_iter=max_iter,
        tol=tolerance,
    )

    from .validation import make_margin_predictor

    return OptimizationResult(
        thresholds=thresholds.astype(np.float64),
        scores=np.array([best_score]),
        predict=make_margin_predictor(thresholds),
        task=Task.MULTICLASS,
        metric=f"macro_{metric}_margin_rule",
        n_classes=n_classes,
    )


def optimize_micro_multiclass(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    metric: str = "f1",
    method: str = "auto",
    sample_weight: ArrayLike | None = None,
    comparison: str = ">",
    tolerance: float = 1e-10,
) -> OptimizationResult:
    """Optimize micro-averaged multiclass metrics using single threshold.

    For micro averaging, we use a single threshold applied to all classes,
    then predict the class with highest valid probability. This reduces to
    a single binary optimization problem on flattened data.

    Decision rule: ŷ = argmax{j: p_j ≥ τ} p_j (or argmax p_j if none valid)

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels in {0, 1, ..., K-1}
    y_score : array-like of shape (n_samples, n_classes)
        Predicted probabilities for each class
    metric : str, default="f1"
        Metric to optimize
    method : str, default="auto"
        Binary optimization method
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights
    comparison : str, default=">"
        Comparison operator
    tolerance : float, default=1e-10
        Numerical tolerance

    Returns
    -------
    OptimizationResult
        Result with single threshold applied to all classes

    Examples
    --------
    >>> result = optimize_micro_multiclass(y_true, y_score, metric="f1")
    >>> result.thresholds  # Same threshold for all classes
    [0.3, 0.3, 0.3]
    """
    from .binary import optimize_metric_binary

    # Validate inputs
    y_true, y_score, sample_weight = validate_multiclass_classification(
        y_true, y_score, sample_weight, require_proba=True
    )

    n_samples, n_classes = y_score.shape

    # Flatten to single binary problem for micro averaging
    # Each (sample, class) pair becomes a binary prediction
    classes = np.arange(n_classes)
    true_binary_flat = (
        np.repeat(y_true, n_classes) == np.tile(classes, n_samples)
    ).astype(int)
    y_score_flat = y_score.ravel()

    # Replicate sample weights if provided
    sample_weight_flat = (
        None if sample_weight is None else np.repeat(sample_weight, n_classes)
    )

    # Optimize single threshold on flattened problem
    result = optimize_metric_binary(
        true_binary_flat,
        y_score_flat,
        metric=metric,
        method=method,
        sample_weight=sample_weight_flat,
        comparison=comparison,
        tolerance=tolerance,
    )

    optimal_threshold = result.thresholds[0]

    from .validation import make_multiclass_predictor

    # Return same threshold for all classes
    thresholds = np.full(n_classes, optimal_threshold, dtype=np.float64)

    return OptimizationResult(
        thresholds=thresholds,
        scores=result.scores,
        predict=make_multiclass_predictor(thresholds, comparison),
        task=Task.MULTICLASS,
        metric=f"micro_{metric}",
        n_classes=n_classes,
    )


def optimize_multiclass(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    metric: str = "f1",
    average: str = "macro",
    method: str = "auto",
    sample_weight: ArrayLike | None = None,
    comparison: str = ">",
    tolerance: float = 1e-10,
) -> OptimizationResult:
    """General multiclass threshold optimization with automatic method selection.

    Routes to appropriate algorithm based on averaging strategy and method:

    - Macro + auto/coord_ascent: Margin rule with coordinate ascent (single-label)
    - Macro + independent: Independent OvR optimization (can predict multiple)
    - Micro: Single threshold optimization (single-label)

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels in {0, 1, ..., K-1}
    y_score : array-like of shape (n_samples, n_classes)
        Predicted probabilities for each class
    metric : str, default="f1"
        Metric to optimize
    average : {"macro", "micro"}, default="macro"
        Averaging strategy
    method : {"auto", "coord_ascent", "independent"}, default="auto"
        Optimization method:
        - "auto": For macro, uses coord_ascent (margin rule)
        - "coord_ascent": Margin rule with coordinate ascent
        - "independent": Independent per-class optimization (OvR)
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights
    comparison : str, default=">"
        Comparison operator
    tolerance : float, default=1e-10
        Numerical tolerance

    Returns
    -------
    OptimizationResult
        Result with optimal thresholds and prediction function

    Examples
    --------
    >>> # Margin rule (single-label, coordinate ascent)
    >>> result = optimize_multiclass(y_true, y_score, method="coord_ascent")
    >>>
    >>> # Independent optimization (can predict multiple classes)
    >>> result = optimize_multiclass(y_true, y_score, method="independent")
    >>>
    >>> # Micro averaging (single threshold)
    >>> result = optimize_multiclass(y_true, y_score, average="micro")
    """
    match average:
        case "micro":
            return optimize_micro_multiclass(
                y_true,
                y_score,
                metric=metric,
                method=method,
                sample_weight=sample_weight,
                comparison=comparison,
                tolerance=tolerance,
            )
        case "macro":
            match method:
                case "auto":
                    # Auto method: choose best method based on metric and comparison compatibility
                    if metric == "f1" and comparison == ">":
                        # F1 with ">" is supported by coordinate ascent - use it for better coupling
                        return optimize_ovr_margin(
                            y_true,
                            y_score,
                            metric=metric,
                            max_iter=30,
                            sample_weight=sample_weight,
                            comparison=comparison,
                            tolerance=tolerance,
                        )
                    else:
                        # Other metrics/comparisons not supported by coord_ascent - use independent
                        return optimize_ovr_independent(
                            y_true,
                            y_score,
                            metric=metric,
                            method="auto",
                            sample_weight=sample_weight,
                            comparison=comparison,
                            tolerance=tolerance,
                        )
                case "coord_ascent":
                    return optimize_ovr_margin(
                        y_true,
                        y_score,
                        metric=metric,
                        max_iter=30,
                        sample_weight=sample_weight,
                        comparison=comparison,
                        tolerance=tolerance,
                    )
                case "independent" | "minimize" | "unique_scan" | "gradient":
                    # Route legacy and scipy methods to independent optimization
                    # minimize, unique_scan, gradient are legacy binary methods - use independent for multiclass
                    return optimize_ovr_independent(
                        y_true,
                        y_score,
                        metric=metric,
                        method="auto",
                        sample_weight=sample_weight,
                        comparison=comparison,
                        tolerance=tolerance,
                    )
                case _:
                    raise ValueError(f"Unknown method for macro averaging: {method}")
        case _:
            raise ValueError(f"Unknown average: {average}. Use 'macro' or 'micro'")


__all__ = [
    "optimize_ovr_independent",
    "optimize_ovr_margin",
    "optimize_micro_multiclass",
    "optimize_multiclass",
]
