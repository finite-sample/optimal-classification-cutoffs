"""Multi-label classification threshold optimization.

This module implements threshold optimization for multi-label classification
where we have K independent binary labels, each with its own threshold τ_j.

Key insight: Multi-label problems are K independent binary problems!
- Macro averaging: Optimize each label independently → O(K·n log n)
- Micro averaging: Thresholds are coupled through global TP/FP/FN → Coordinate ascent

All functions assume calibrated probabilities: E[y|p] = p
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .core import OptimizationResult, Task


def optimize_macro_multilabel(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    metric: str = "f1",
    method: str = "auto",
    sample_weight: ArrayLike | None = None,
    comparison: str = ">",
    tolerance: float = 1e-10,
) -> OptimizationResult:
    """Optimize macro-averaged metrics for multi-label classification.

    For macro averaging, each label is optimized independently:
    Macro-F1 = (1/K) Σ_j F1_j(τ_j)

    Since each F1_j depends only on τ_j, we can optimize each threshold
    independently using binary optimization. This is exact and efficient.

    Parameters
    ----------
    y_true : array-like of shape (n_samples, n_labels)
        True multi-label binary matrix
    y_score : array-like of shape (n_samples, n_labels)
        Predicted probabilities for each label
    metric : str, default="f1"
        Metric to optimize per label ("f1", "precision", "recall")
    method : str, default="auto"
        Binary optimization method for each label
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights
    comparison : str, default=">"
        Comparison operator
    tolerance : float, default=1e-10
        Numerical tolerance

    Returns
    -------
    OptimizationResult
        Result with per-label thresholds and macro-averaged score

    Examples
    --------
    >>> # 3 independent labels
    >>> y_true = [[1, 0, 1], [0, 1, 0], [1, 1, 1]]
    >>> y_score = [[0.8, 0.2, 0.9], [0.1, 0.7, 0.3], [0.9, 0.8, 0.7]]
    >>> result = optimize_macro_multilabel(y_true, y_score, metric="f1")
    >>> len(result.thresholds)  # One per label
    3
    """
    from .binary import optimize_metric_binary

    # Validate inputs for multilabel
    y_true = np.asarray(y_true, dtype=np.int8)
    y_score = np.asarray(y_score, dtype=np.float64)

    if y_true.ndim != 2:
        raise ValueError(f"Multilabel y_true must be 2D, got shape {y_true.shape}")
    if y_score.ndim != 2:
        raise ValueError(f"Multilabel y_score must be 2D, got shape {y_score.shape}")
    if y_true.shape != y_score.shape:
        raise ValueError(
            f"Shape mismatch: labels {y_true.shape} vs probs {y_score.shape}"
        )

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if len(sample_weight) != y_true.shape[0]:
            raise ValueError("Sample weights must match number of samples")

    n_samples, n_labels = y_true.shape

    # Optimize each label independently
    optimal_thresholds = np.zeros(n_labels, dtype=np.float64)
    optimal_scores = np.zeros(n_labels, dtype=np.float64)

    for j in range(n_labels):
        # Extract binary problem for label j
        y_true_j = y_true[:, j] if y_true.ndim == 2 else (y_true == j).astype(int)
        y_score_j = y_score[:, j]

        # Optimize threshold for this label
        result_j = optimize_metric_binary(
            y_true_j,
            y_score_j,
            metric=metric,
            method=method,
            sample_weight=sample_weight,
            comparison=comparison,
            tolerance=tolerance,
        )

        optimal_thresholds[j] = result_j.thresholds[0]
        optimal_scores[j] = result_j.scores[0]

    # Macro average score
    macro_score = np.mean(optimal_scores)

    def predict_multilabel(probs: ArrayLike) -> np.ndarray:
        """Predict using per-label thresholds (independent decisions)."""
        from .validation import apply_threshold

        p = np.asarray(probs, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] != n_labels:
            raise ValueError(f"Expected probabilities shape (n_samples, {n_labels})")

        return apply_threshold(p, optimal_thresholds[None, :], comparison)

    return OptimizationResult(
        thresholds=optimal_thresholds,
        scores=np.array([macro_score]),
        predict=predict_multilabel,
        task=Task.MULTILABEL,
        metric=f"macro_{metric}",
        n_classes=n_labels,
    )


def optimize_micro_multilabel(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    metric: str = "f1",
    max_iter: int = 30,
    sample_weight: ArrayLike | None = None,
    comparison: str = ">",
    tolerance: float = 1e-10,
) -> OptimizationResult:
    """Optimize micro-averaged metrics for multi-label classification.

    For micro averaging, thresholds are coupled through global TP/FP/FN:
    Micro-F1 = 2·TP_total / (2·TP_total + FP_total + FN_total)

    where TP_total = Σ_j TP_j(τ_j). Changing any τ_j affects the global metric,
    so we use coordinate ascent to optimize the coupled problem.

    Parameters
    ----------
    y_true : array-like of shape (n_samples, n_labels)
        True multi-label binary matrix
    y_score : array-like of shape (n_samples, n_labels)
        Predicted probabilities for each label
    metric : str, default="f1"
        Metric to optimize ("f1", "precision", "recall")
    max_iter : int, default=30
        Maximum coordinate ascent iterations
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights
    comparison : str, default=">"
        Comparison operator
    tolerance : float, default=1e-12
        Convergence tolerance

    Returns
    -------
    OptimizationResult
        Result with per-label thresholds optimized for micro averaging

    Examples
    --------
    >>> result = optimize_micro_multilabel(y_true, y_score, metric="f1")
    >>> # Thresholds are coupled - changing one affects global metric
    """
    from .metrics_core import get_metric_function

    # Validate inputs for multilabel
    labels_arr: np.ndarray = np.asarray(y_true, dtype=np.int8)
    proba_arr: np.ndarray = np.asarray(y_score, dtype=np.float64)

    if labels_arr.ndim != 2:
        raise ValueError(f"Multilabel y_true must be 2D, got shape {labels_arr.shape}")
    if proba_arr.ndim != 2:
        raise ValueError(f"Multilabel y_score must be 2D, got shape {proba_arr.shape}")
    if labels_arr.shape != proba_arr.shape:
        raise ValueError(
            f"Shape mismatch: labels {labels_arr.shape} vs probs {proba_arr.shape}"
        )

    n_samples, n_labels = labels_arr.shape

    from .validation import get_sample_weights

    weights_arr = get_sample_weights(sample_weight, n_samples)

    # Initialize thresholds
    thresholds = np.zeros(n_labels, dtype=np.float64)

    metric_fn = get_metric_function(metric)

    def compute_global_metric(tau: np.ndarray) -> float:
        """Compute micro-averaged metric for given thresholds."""
        total_tp = total_fp = total_fn = 0.0

        for j in range(n_labels):
            # Binary predictions for label j
            if comparison == ">=":
                pred_j = (proba_arr[:, j] >= tau[j]).astype(int)
            else:
                pred_j = (proba_arr[:, j] > tau[j]).astype(int)

            true_j = labels_arr[:, j]

            # Confusion matrix for label j
            tp_j = np.sum(weights_arr * (true_j == 1) * (pred_j == 1))
            fp_j = np.sum(weights_arr * (true_j == 0) * (pred_j == 1))
            fn_j = np.sum(weights_arr * (true_j == 1) * (pred_j == 0))

            total_tp += tp_j
            total_fp += fp_j
            total_fn += fn_j

        # Micro metric (TN not meaningful for micro averaging)
        return float(metric_fn(total_tp, 0.0, total_fp, total_fn))

    best_score = compute_global_metric(thresholds)

    # Coordinate ascent
    for _iteration in range(max_iter):
        improved = False

        for j in range(n_labels):
            # Fix all other thresholds, optimize τ_j
            candidates = np.unique(proba_arr[:, j])
            best_tau_j = thresholds[j]
            best_score_j = best_score

            for tau_j in candidates:
                thresholds[j] = tau_j
                score = compute_global_metric(thresholds)

                if score > best_score_j + tolerance:
                    best_score_j = score
                    best_tau_j = tau_j
                    improved = True

            thresholds[j] = best_tau_j
            best_score = best_score_j

        if not improved:
            break

    def predict_multilabel_micro(probs: ArrayLike) -> np.ndarray:
        """Predict using micro-optimized thresholds."""
        from .validation import apply_threshold

        p = np.asarray(probs, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] != n_labels:
            raise ValueError(f"Expected probabilities shape (n_samples, {n_labels})")

        return apply_threshold(p, thresholds[None, :], comparison)

    return OptimizationResult(
        thresholds=thresholds,
        scores=np.array([best_score]),
        predict=predict_multilabel_micro,
        task=Task.MULTILABEL,
        metric=f"micro_{metric}",
        n_classes=n_labels,
    )


def optimize_multilabel(
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
    """General multi-label threshold optimization with automatic method selection.

    Routes to appropriate algorithm based on averaging strategy:
    - Macro: Independent optimization per label (exact, O(K·n log n))
    - Micro: Coordinate ascent for coupled thresholds (local optimum)

    Parameters
    ----------
    y_true : array-like of shape (n_samples, n_labels)
        True multi-label binary matrix
    y_score : array-like of shape (n_samples, n_labels)
        Predicted probabilities for each label
    metric : str, default="f1"
        Metric to optimize
    average : {"macro", "micro"}, default="macro"
        Averaging strategy
    method : str, default="auto"
        Optimization method (passed to binary optimizer for macro)
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights
    comparison : str, default=">"
        Comparison operator
    tolerance : float, default=1e-10
        Numerical tolerance

    Returns
    -------
    OptimizationResult
        Result with optimal thresholds and metric score

    Examples
    --------
    >>> # Independent per-label optimization
    >>> result = optimize_multilabel(y_true, y_score, average="macro")
    >>>
    >>> # Coupled optimization for global metric
    >>> result = optimize_multilabel(y_true, y_score, average="micro")
    """
    match average:
        case "macro":
            return optimize_macro_multilabel(
                y_true,
                y_score,
                metric=metric,
                method=method,
                sample_weight=sample_weight,
                comparison=comparison,
                tolerance=tolerance,
            )
        case "micro":
            return optimize_micro_multilabel(
                y_true,
                y_score,
                metric=metric,
                max_iter=30,
                sample_weight=sample_weight,
                comparison=comparison,
                tolerance=tolerance,
            )
        case _:
            raise ValueError(f"Unknown average: {average}. Use 'macro' or 'micro'")


__all__ = [
    "optimize_macro_multilabel",
    "optimize_micro_multilabel",
    "optimize_multilabel",
]
