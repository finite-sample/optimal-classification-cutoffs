"""Expected metric optimization under calibration assumption.

This module implements expected threshold optimization using Dinkelbach's algorithm.
Under the calibration assumption, expected metrics can be optimized exactly by
finding optimal thresholds on calibrated probabilities.

Key simplifications:
- Single Dinkelbach implementation for all metrics
- Direct computation instead of coefficient abstraction
- Clear separation: core algorithm vs metrics vs multiclass
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .core import OptimizationResult, Task

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# ============================================================================
# Helper Functions
# ============================================================================


def _vectorize_threshold_function(
    scalar_fn: Callable[[float], float], candidates: np.ndarray
) -> np.ndarray:
    """Vectorize a scalar threshold function over candidate array.

    Args:
        scalar_fn: Function that takes a scalar threshold and returns a scalar value
        candidates: Array of threshold candidates to evaluate

    Returns:
        Array of function values for each candidate
    """
    candidates = np.asarray(candidates)
    if candidates.ndim == 0:
        return np.asarray(scalar_fn(float(candidates)))
    return np.array([scalar_fn(float(t)) for t in candidates])


# ============================================================================
# Core Algorithm - Single Dinkelbach implementation
# ============================================================================


def dinkelbach_optimize(
    probabilities: np.ndarray[Any, Any],
    numerator_fn: Callable[[float], float],
    denominator_fn: Callable[[float], float],
    max_iter: int = 100,
    tol: float = 1e-12,
) -> tuple[float, float]:
    """Core Dinkelbach algorithm for ratio optimization.

    Solves: max_t numerator(t) / denominator(t)

    Args:
        probabilities: Calibrated probabilities. Shape: (n,).
        numerator_fn: Computes numerator at given threshold
        denominator_fn: Computes denominator at given threshold
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Optimal threshold
        score : float
        Optimal ratio value
    """
    # Sort probabilities once
    p = np.sort(probabilities)

    # Initial lambda and defaults for edge case where max_iter == 0
    lam = 0.5
    new_lam = lam
    converged = False
    best_t = 0.5  # Default threshold
    den = 1.0  # Default denominator to avoid unbound variable

    if max_iter == 0:
        # Handle edge case: return median threshold with initial lambda
        return float(np.median(p)), lam

    for iteration in range(max_iter):
        # Vectorized grid search over unique probabilities
        candidates = np.unique(p)

        # Vectorize the objective function evaluation
        numerator_values = _vectorize_threshold_function(numerator_fn, candidates)
        denominator_values = _vectorize_threshold_function(denominator_fn, candidates)
        obj_values = numerator_values - lam * denominator_values

        # Find the best candidate
        best_idx = np.argmax(obj_values)
        best_t = candidates[best_idx]

        # Update lambda
        num = numerator_fn(best_t)
        den = denominator_fn(best_t)

        if den == 0:
            logger.warning(
                "Dinkelbach algorithm terminated early due to zero denominator "
                "(numerical instability) at iteration %d",
                iteration + 1,
            )
            break

        new_lam = num / den

        if abs(new_lam - lam) < tol:
            converged = True
            break

        lam = new_lam

    if not converged and den != 0:
        final_tolerance = abs(new_lam - lam)
        logger.warning(
            "Dinkelbach algorithm did not converge within %d iterations. "
            "Final tolerance: %.2e, target: %.2e",
            max_iter,
            final_tolerance,
            tol,
        )

    return float(best_t), float(lam)


# ============================================================================
# Metric-specific expected optimization
# ============================================================================


def dinkelbach_expected_fbeta_binary(
    y_score: np.ndarray[Any, Any],
    beta: float = 1.0,
    sample_weight: np.ndarray[Any, Any] | None = None,
    comparison: str = ">",
) -> OptimizationResult:
    """Expected F-beta optimization under calibration, in O(n log n).

    Args:
        y_score: Calibrated probabilities for positive class. Shape: (n,).
        beta: F-beta parameter
        sample_weight: Sample weights. Shape: (n,). Optional.
        comparison: Comparison operator (kept for backward compatibility)

    Returns:
        Optimization result with threshold, score, and predict function

    Raises:
        ValueError: If probabilities fall outside [0, 1] or weights are negative.
    """
    # Step 1: Setup and validation
    from .validation import get_sample_weights

    p = np.asarray(y_score, dtype=np.float64)
    n = len(p)
    w = get_sample_weights(sample_weight, n)

    # Validate inputs
    if not np.all((p >= 0) & (p <= 1)):
        raise ValueError("Probabilities must be in [0, 1]")
    if np.any(w < 0):
        raise ValueError("Weights must be non-negative")

    # Step 2: Sort probabilities and weights together
    sort_idx = np.argsort(p)
    p_sorted = p[sort_idx]
    w_sorted = w[sort_idx]

    # Step 3: Pre-compute cumulative sums for O(1) metric evaluation
    wp_sorted = w_sorted * p_sorted
    w1mp_sorted = w_sorted * (1 - p_sorted)

    # Cumulative sums from right (for p > threshold)
    cumsum_wp_right = np.zeros(n + 1)
    cumsum_w1mp_right = np.zeros(n + 1)

    cumsum_wp_right[:-1] = np.cumsum(wp_sorted[::-1])[::-1]
    cumsum_w1mp_right[:-1] = np.cumsum(w1mp_sorted[::-1])[::-1]

    # Cumulative sums from left (for p <= threshold)
    cumsum_wp_left = np.zeros(n + 1)
    cumsum_wp_left[1:] = np.cumsum(wp_sorted)

    # Step 4: Define efficient metric functions using cumulative sums
    def compute_fbeta_at_index(idx: int) -> tuple[float, float]:
        """Compute F-beta numerator and denominator at threshold index."""
        # For threshold at p_sorted[idx], we predict positive for p > p_sorted[idx]

        # Expected TP: sum of p*w for p > threshold
        expected_tp = cumsum_wp_right[idx + 1]

        # Expected FP: sum of (1-p)*w for p > threshold
        expected_fp = cumsum_w1mp_right[idx + 1]

        # Expected FN: sum of p*w for p <= threshold
        expected_fn = cumsum_wp_left[idx + 1]

        beta2 = beta**2
        numerator = (1 + beta2) * expected_tp
        denominator = (1 + beta2) * expected_tp + expected_fp + beta2 * expected_fn

        return numerator, denominator

    # Step 5: Dinkelbach iterations with efficient evaluation
    lambda_val = 0.5
    new_lambda = lambda_val
    max_iter = 100
    tol = 1e-12
    converged = False
    den = 1.0
    best_threshold = 0.5
    best_score = 0.0

    # Get unique threshold candidates (including edge cases)
    unique_p, unique_idx = np.unique(p_sorted, return_index=True)
    # Add threshold at endpoints for completeness
    threshold_indices = [*list(unique_idx), n]
    if len(unique_p) == 0 or unique_p[0] > 0:
        threshold_indices = [-1, *threshold_indices]

    for iteration in range(max_iter):
        best_threshold = 0.5
        best_value = -np.inf
        best_score = 0.0

        # Evaluate objective at each unique threshold
        for idx in threshold_indices:
            if idx == -1:
                # Threshold = 0 (everything predicted positive)
                threshold = 0.0
                num = cumsum_wp_right[0]
                den = cumsum_wp_right[0] + cumsum_w1mp_right[0]
            elif idx == n:
                # Threshold = 1 (nothing predicted positive)
                threshold = 1.0
                num, den = 0.0, cumsum_wp_left[n]
                if den == 0:
                    den = 1.0  # Avoid division by zero
            else:
                threshold = p_sorted[idx]
                num, den = compute_fbeta_at_index(idx)

            # Objective: maximize num - lambda * den
            obj_value = num - lambda_val * den

            if obj_value > best_value:
                best_value = obj_value
                best_threshold = threshold
                best_score = num / den if den > 0 else 0.0

        # Update lambda
        if best_threshold == 0.0:
            idx = -1
        elif best_threshold == 1.0:
            idx = n
        else:
            idx = int(np.searchsorted(p_sorted, best_threshold, side="right")) - 1

        if idx == -1:
            num = cumsum_wp_right[0]
            den = cumsum_wp_right[0] + cumsum_w1mp_right[0]
        elif idx == n:
            num, den = 0.0, cumsum_wp_left[n]
            if den == 0:
                den = 1.0
        else:
            num, den = compute_fbeta_at_index(idx)

        if den == 0:
            logger.warning(
                "Dinkelbach expected F-beta optimization terminated early due to zero "
                "denominator "
                "(numerical instability) at iteration %d",
                iteration + 1,
            )
            break

        new_lambda = num / den if den > 0 else 0.0

        if abs(new_lambda - lambda_val) < tol:
            converged = True
            break

        lambda_val = new_lambda

    if not converged and den != 0:
        final_tolerance = abs(new_lambda - lambda_val)
        logger.warning(
            "Dinkelbach expected F-beta optimization did not converge within %d "
            "iterations. "
            "Final tolerance: %.2e, target: %.2e",
            max_iter,
            final_tolerance,
            tol,
        )

    best_threshold_float = float(best_threshold)
    best_score_float = float(best_score)

    from .validation import make_binary_predictor

    return OptimizationResult(
        thresholds=np.array([best_threshold_float]),
        scores=np.array([best_score_float]),
        predict=make_binary_predictor(best_threshold_float, comparison),
        task=Task.BINARY,
        metric=f"expected_f{beta}",
        n_classes=2,
    )


def expected_precision(
    probabilities: np.ndarray[Any, Any], weights: np.ndarray[Any, Any] | None = None
) -> tuple[float, float]:
    """Expected precision optimization.

    Precision = TP / (TP + FP)
    """
    p = np.asarray(probabilities, dtype=np.float64)

    if weights is None:
        weights = np.ones_like(p)

    wp = weights * p
    w1mp = weights * (1 - p)

    def numerator(t):
        mask = p > t
        return np.sum(wp[mask])  # Expected TP

    def denominator(t):
        mask = p > t
        return np.sum(wp[mask]) + np.sum(w1mp[mask])  # Expected TP + FP

    return dinkelbach_optimize(p, numerator, denominator)


def expected_jaccard(
    probabilities: np.ndarray[Any, Any], weights: np.ndarray[Any, Any] | None = None
) -> tuple[float, float]:
    """Expected Jaccard/IoU optimization.

    Jaccard = TP / (TP + FP + FN)
    """
    p = np.asarray(probabilities, dtype=np.float64)

    if weights is None:
        weights = np.ones_like(p)

    wp = weights * p

    def numerator(t):
        mask = p > t
        return np.sum(wp[mask])  # Expected TP

    def denominator(t):
        # TP + FP + FN = all predicted positive + all actual positive - TP
        mask = p > t
        tp = np.sum(wp[mask])
        predicted_pos = np.sum(weights[mask])
        actual_pos = np.sum(wp)  # Total expected positives
        return predicted_pos + actual_pos - tp

    return dinkelbach_optimize(p, numerator, denominator)


# ============================================================================
# Multiclass/Multilabel wrapper
# ============================================================================


def dinkelbach_expected_fbeta_multilabel(
    y_score: np.ndarray[Any, Any],
    beta: float = 1.0,
    sample_weight: np.ndarray[Any, Any] | None = None,
    average: Literal["macro", "micro", "weighted"] = "macro",
    y_true: np.ndarray[Any, Any] | None = None,
    comparison: str = ">",
) -> OptimizationResult:
    """Expected F-beta optimization for multilabel/multiclass.

    Args:
        y_score: Class probabilities. Shape: (n_samples, n_classes).
        beta: F-beta parameter
        sample_weight: Sample weights. Shape: (n_samples,). Optional.
        average: Averaging strategy:
            - "macro": Per-class thresholds, unweighted mean
            - "micro": Single global threshold
            - "weighted": Per-class thresholds, weighted by true class frequencies
        y_true: True class labels. Required when average="weighted" to compute class
            frequencies.
            Should contain integer class indices from 0 to n_classes-1. Shape:
            (n_samples,). Optional.
        comparison: Comparison operator (kept for backward compatibility)

    Returns:
        Results with 'thresholds' and 'score' keys

    Raises:
        ValueError: If probabilities are not 2D, or weighted averaging is asked
                    for without true labels to derive class frequencies from.
    """
    P = np.asarray(y_score, dtype=np.float64)

    if P.ndim != 2:
        raise ValueError(f"Expected 2D probabilities, got shape {P.shape}")

    _n_samples, n_classes = P.shape

    if average == "micro":
        # Flatten all probabilities into single binary problem
        p_flat = P.ravel()

        if sample_weight is not None:
            # Repeat weights for each class
            w_flat = np.repeat(sample_weight, n_classes)
        else:
            w_flat = None

        result = dinkelbach_expected_fbeta_binary(p_flat, beta, w_flat, comparison)
        threshold = result.threshold
        score = result.score

        from .validation import make_multiclass_predictor

        thresholds_arr = np.full(n_classes, threshold)
        return OptimizationResult(
            thresholds=thresholds_arr,
            scores=np.full(n_classes, score),
            predict=make_multiclass_predictor(thresholds_arr, comparison),
            task=Task.MULTICLASS,
            metric=f"expected_f{beta}",
            n_classes=n_classes,
        )

    # macro or weighted
    # Optimize per-class thresholds
    thresholds = np.zeros(n_classes)
    scores = np.zeros(n_classes)

    for k in range(n_classes):
        result = dinkelbach_expected_fbeta_binary(
            P[:, k], beta, sample_weight, comparison
        )
        thresholds[k] = result.threshold
        scores[k] = result.score

    # Note: Previous avg_score calculation removed as it was unused
    # Validate weighted averaging requirements
    if average == "weighted" and y_true is None:
        raise ValueError(
            "Weighted averaging requires true_labels to compute class frequencies"
        )

    from .validation import make_multiclass_predictor

    return OptimizationResult(
        thresholds=thresholds,
        scores=scores,
        predict=make_multiclass_predictor(thresholds, comparison),
        task=Task.MULTICLASS,
        metric=f"expected_f{beta}",
        n_classes=n_classes,
    )


def expected_optimize_multiclass(
    probabilities: np.ndarray[Any, Any],
    metric: str = "f1",
    average: Literal["macro", "micro", "weighted"] = "macro",
    weights: np.ndarray[Any, Any] | None = None,
    **metric_params,
) -> OptimizationResult:
    """Expected optimization for multiclass/multilabel.

    Args:
        probabilities: Class probabilities. Shape: (n_samples, n_classes).
        metric: Metric to optimize ("f1", "precision", "jaccard")
        average: Averaging strategy
        weights: Sample weights. Shape: (n_samples,). Optional.
        **metric_params: Additional parameters (e.g., beta for F-beta)

    Returns:
        Optimization result with thresholds, scores, and predict function

    Raises:
        ValueError: If probabilities are not 2D, or the metric is unsupported.
    """
    P = np.asarray(probabilities, dtype=np.float64)

    if P.ndim != 2:
        raise ValueError(f"Expected 2D probabilities, got shape {P.shape}")

    _n_samples, n_classes = P.shape

    # Select metric function
    # The three branches return different shapes (an OptimizationResult from the
    # Dinkelbach solver, a (threshold, score) pair from the closed forms), so the
    # union is declared up front rather than inferred from whichever comes first.
    metric_fn: Callable[..., OptimizationResult | tuple[float, float]]
    if metric.lower() in {"f1", "fbeta"}:
        beta = metric_params.get("beta", 1.0)

        def expected_fbeta(p, w):
            return dinkelbach_expected_fbeta_binary(p, beta, w)

        metric_fn = expected_fbeta
    elif metric.lower() == "precision":
        metric_fn = expected_precision
    elif metric.lower() in {"jaccard", "iou"}:
        metric_fn = expected_jaccard
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Handle different averaging strategies
    if average == "micro":
        # Flatten all probabilities into single binary problem
        p_flat = P.ravel()

        # Repeat weights for each class
        w_flat = np.repeat(weights, n_classes) if weights is not None else None

        result_micro = metric_fn(p_flat, w_flat)
        if isinstance(result_micro, OptimizationResult):
            # For F1/Fbeta metrics
            threshold = result_micro.threshold
            score = result_micro.score
        else:
            # For precision/jaccard metrics that return tuples
            threshold, score = result_micro

        # Create prediction function for micro averaging
        def predict_micro(probs):
            p = np.asarray(probs)
            if p.ndim == 2:
                # Apply same threshold to all classes, predict argmax of those above
                # threshold
                above_threshold = p > threshold
                if np.any(above_threshold, axis=1).all():
                    masked = np.where(above_threshold, p, -np.inf)
                    return np.argmax(masked, axis=1)
                # Fallback to argmax for samples with no class above threshold
                return np.argmax(p, axis=1)
            # Binary case
            return (p > threshold).astype(int)

        return OptimizationResult(
            thresholds=np.full(n_classes, threshold),
            scores=np.full(n_classes, score),
            predict=predict_micro,
            task=Task.MULTICLASS,
            metric=f"expected_{metric}",
            n_classes=n_classes,
        )

    # macro or weighted
    # Optimize per-class thresholds
    thresholds = np.zeros(n_classes)
    scores = np.zeros(n_classes)

    for k in range(n_classes):
        result_k = metric_fn(P[:, k], weights)
        if isinstance(result_k, OptimizationResult):
            # For F1/Fbeta metrics
            thresholds[k] = result_k.threshold
            scores[k] = result_k.score
        else:
            # For precision/jaccard metrics that return tuples
            thresholds[k], scores[k] = result_k

    # Weight by class frequency for weighted average
    if average == "weighted":
        # Weight by class frequency
        if weights is not None:
            class_weights = np.sum(P * weights[:, None], axis=0)
        else:
            class_weights = np.sum(P, axis=0)

        class_weights /= class_weights.sum()

    # Create prediction function for macro/weighted averaging
    def predict_macro(probs):
        p = np.asarray(probs)
        if p.ndim == 2:
            # Apply per-class thresholds, predict argmax of those above threshold
            above_threshold = p > thresholds[None, :]
            has_valid = np.any(above_threshold, axis=1)
            predictions = np.zeros(p.shape[0], dtype=int)

            # For samples with at least one class above threshold
            if np.any(has_valid):
                masked = np.where(above_threshold, p, -np.inf)
                predictions[has_valid] = np.argmax(masked[has_valid], axis=1)

            # For samples with no class above threshold, use argmax
            if np.any(~has_valid):
                predictions[~has_valid] = np.argmax(p[~has_valid], axis=1)

            return predictions
        # Binary case - use first threshold
        return (p > thresholds[0]).astype(int)

    return OptimizationResult(
        thresholds=thresholds,
        scores=scores,
        predict=predict_macro,
        task=Task.MULTICLASS,
        metric=f"expected_{metric}",
        n_classes=n_classes,
    )
