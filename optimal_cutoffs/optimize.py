"""Unified threshold optimization for binary and multiclass classification.

This module consolidates all threshold optimization functionality into a single,
streamlined interface. It includes high-performance Numba kernels, multiple
optimization algorithms, and support for both binary and multiclass problems.

Key features:
- Fast Numba kernels with Python fallbacks
- Binary and multiclass threshold optimization
- Multiple algorithms: sort-scan, scipy, gradient, coordinate ascent
- Sample weight support (including in coordinate ascent)
- Direct functional API without over-engineered abstractions
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import optimize

from .core import OptimizationResult, Task
from .numba_utils import jit, numba_with_fallback
from .validation import get_sample_weights, validate_binary_classification

logger = logging.getLogger(__name__)

# ============================================================================
# Data Validation
# ============================================================================


# Removed validate_binary_data - use validate_binary_classification from validation.py
# instead


# ============================================================================
# Fast Kernels (Unified with auto-fallback)
# ============================================================================


@numba_with_fallback(nopython=True, fastmath=True, cache=True)
def fast_f1_score(tp: float, tn: float, fp: float, fn: float) -> float:
    """Compute F1 score from confusion matrix."""
    denom = 2.0 * tp + fp + fn
    return 2.0 * tp / denom if denom > 0.0 else 0.0


@numba_with_fallback(nopython=True, cache=True)
def compute_confusion_matrix_weighted(
    labels: np.ndarray, predictions: np.ndarray, weights: np.ndarray | None
) -> tuple[float, float, float, float]:
    """Compute weighted confusion matrix elements (serial, race-free)."""
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    n = labels.shape[0]

    if weights is None:
        for i in range(n):
            if labels[i] == 1:
                if predictions[i]:
                    tp += 1.0
                else:
                    fn += 1.0
            else:
                if predictions[i]:
                    fp += 1.0
                else:
                    tn += 1.0
    else:
        for i in range(n):
            w = weights[i]
            if labels[i] == 1:
                if predictions[i]:
                    tp += w
                else:
                    fn += w
            else:
                if predictions[i]:
                    fp += w
                else:
                    tn += w

    return tp, tn, fp, fn


# ============================================================================
# Fast Kernels (Numba JIT-compiled)
# ============================================================================


@jit(nopython=True, fastmath=True, cache=True)
def sort_scan_kernel(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    inclusive: bool,
) -> tuple[float, float]:
    """Numba sort-and-scan for F1. Honors inclusive operator at boundaries.

    Note: weights must be a valid array (use np.ones for uniform weights).
    """
    n = labels.shape[0]
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    sorted_weights = weights[order]

    tp = 0.0
    fn = 0.0
    fp = 0.0
    tn = 0.0

    for i in range(n):
        if sorted_labels[i] == 1:
            fn += sorted_weights[i]
        else:
            tn += sorted_weights[i]

    eps = 1e-10
    best_threshold = sorted_scores[0] + (eps if inclusive else 0.0)
    best_score = fast_f1_score(tp, tn, fp, fn)

    for i in range(n):
        w = sorted_weights[i]
        if sorted_labels[i] == 1:
            tp += w
            fn -= w
        else:
            fp += w
            tn -= w

        score = fast_f1_score(tp, tn, fp, fn)
        if score > best_score:
            best_score = score
            if i < n - 1:
                best_threshold = 0.5 * (sorted_scores[i] + sorted_scores[i + 1])
            else:
                best_threshold = sorted_scores[i] - (eps if inclusive else 0.0)

    return best_threshold, best_score


@jit(nopython=True, fastmath=True, cache=True)
def compute_macro_f1(tp: np.ndarray, fp: np.ndarray, support: np.ndarray) -> float:
    """Compute macro F1 from per-class TP/FP and support (FN = support - TP)."""
    f1_sum = 0.0
    k = tp.shape[0]
    for c in range(k):
        fn = support[c] - tp[c]
        denom = 2.0 * tp[c] + fp[c] + fn
        if denom > 0.0:
            f1_sum += 2.0 * tp[c] / denom
    return f1_sum / float(k)


@jit(nopython=True, fastmath=True, cache=True)
def coordinate_ascent_kernel(
    y_true: np.ndarray,
    probs: np.ndarray,
    weights: np.ndarray,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Numba coordinate ascent for multiclass macro-F1 with sample weights.

    Predict via argmax over (p - tau). We iteratively adjust one class's
    threshold at a time by scanning the implied breakpoints for that class.

    Note: weights must be a valid array (use np.ones for uniform weights).
    """
    n, k = probs.shape
    thresholds = np.zeros(k, dtype=np.float64)
    history = np.zeros(max_iter, dtype=np.float64)

    # Per-class weighted supports (sum of weights for true label == c)
    support = np.zeros(k, dtype=np.float64)
    for i in range(n):
        support[y_true[i]] += weights[i]

    # Initialize by assigning every sample to its current best class
    tp = np.zeros(k, dtype=np.float64)
    fp = np.zeros(k, dtype=np.float64)
    for i in range(n):
        w = weights[i]
        pred = 0
        best = probs[i, 0] - thresholds[0]
        for j in range(1, k):
            val = probs[i, j] - thresholds[j]
            if val > best:
                best = val
                pred = j
        if y_true[i] == pred:
            tp[pred] += w
        else:
            fp[pred] += w

    best_score = compute_macro_f1(tp, fp, support)
    no_improve_rounds = 0

    for it in range(max_iter):
        improved_any = False

        for c in range(k):
            # For every i, compute breakpoint b_i = p_ic - max_{j!=c}(p_ij - tau_j)
            breakpoints = np.empty(n, dtype=np.float64)
            alternatives = np.empty(n, dtype=np.int32)

            for i in range(n):
                max_other = -1e308
                max_other_idx = -1
                for j in range(k):
                    if j != c:
                        v = probs[i, j] - thresholds[j]
                        if v > max_other:
                            max_other = v
                            max_other_idx = j
                breakpoints[i] = probs[i, c] - max_other
                alternatives[i] = max_other_idx

            order = np.argsort(-breakpoints)

            # Baseline: everyone currently assigned to alternatives
            tp_cand = np.zeros(k, dtype=np.float64)
            fp_cand = np.zeros(k, dtype=np.float64)
            for i in range(n):
                w = weights[i]
                pred = alternatives[i]
                if y_true[i] == pred:
                    tp_cand[pred] += w
                else:
                    fp_cand[pred] += w

            baseline = compute_macro_f1(tp_cand, fp_cand, support)
            current_best = baseline
            best_idx = -1

            # Simulate crossing each breakpoint in order
            for rank in range(n):
                idx = order[rank]
                w = weights[idx]
                old_pred = alternatives[idx]

                if y_true[idx] == old_pred:
                    tp_cand[old_pred] -= w
                else:
                    fp_cand[old_pred] -= w

                if y_true[idx] == c:
                    tp_cand[c] += w
                else:
                    fp_cand[c] += w

                score = compute_macro_f1(tp_cand, fp_cand, support)
                if score > current_best:
                    current_best = score
                    best_idx = rank

            # If we found an improvement for this coordinate, commit it
            if best_idx >= 0 and current_best > baseline + tol:
                sorted_breaks = breakpoints[order]
                if best_idx + 1 < n:
                    new_threshold = 0.5 * (
                        sorted_breaks[best_idx] + sorted_breaks[best_idx + 1]
                    )
                else:
                    new_threshold = sorted_breaks[best_idx] - 1e-6

                thresholds[c] = new_threshold
                # Rebuild committed tp/fp to match current thresholds
                tp[:] = 0.0
                fp[:] = 0.0
                for i in range(n):
                    w = weights[i]
                    pred = 0
                    best = probs[i, 0] - thresholds[0]
                    for j in range(1, k):
                        val = probs[i, j] - thresholds[j]
                        if val > best:
                            best = val
                            pred = j
                    if y_true[i] == pred:
                        tp[pred] += w
                    else:
                        fp[pred] += w

                new_global = compute_macro_f1(tp, fp, support)
                if new_global > best_score + tol:
                    best_score = new_global
                improved_any = True

        history[it] = best_score

        if not improved_any:
            no_improve_rounds += 1
            if no_improve_rounds >= 2:
                return thresholds, best_score, history[: it + 1]
        else:
            no_improve_rounds = 0

    return thresholds, best_score, history


# ============================================================================
# Binary Optimization Algorithms
# ============================================================================


def optimize_sort_scan(
    labels: np.ndarray,
    scores: np.ndarray,
    metric: str,
    weights: np.ndarray | None = None,
    operator: str = ">=",
) -> OptimizationResult:
    """Sort-and-scan optimization for piecewise-constant metrics."""
    from .validation import get_sample_weights

    logger.debug("Using sort_scan optimization for %s metric", metric)
    labels, scores, _ = validate_binary_classification(labels, scores, weights)
    weights = get_sample_weights(weights, len(labels))

    if metric.lower() in ("f1", "f1_score"):
        threshold, score = sort_scan_kernel(
            labels, scores, weights, inclusive=(operator == ">=")
        )
    else:
        # _generic_sort_scan can handle None weights, but pass the array for consistency
        threshold, score = _generic_sort_scan(labels, scores, metric, weights, operator)

    from .validation import make_binary_predictor

    return OptimizationResult(
        thresholds=np.array([threshold], dtype=float),
        scores=np.array([score], dtype=float),
        predict=make_binary_predictor(threshold, operator),
        task=Task.BINARY,
        metric=metric,
        n_classes=2,
    )


def _generic_sort_scan(
    labels: np.ndarray,
    scores: np.ndarray,
    metric: str,
    weights: np.ndarray | None,
    operator: str,
) -> tuple[float, float]:
    """Generic sort-and-scan implementation for any metric."""
    if len(labels) == 0:
        return 0.5, 0.0

    from .metrics_core import METRICS

    metric_fn = METRICS[metric].fn

    order = np.argsort(scores)  # ascending
    sorted_scores = scores[order]

    eps = 1e-10  # default tolerance for boundary conditions
    boundary_thresholds = np.array(
        [sorted_scores[0] - eps, sorted_scores[-1] + eps], dtype=float
    )
    all_thresholds = np.unique(
        np.concatenate([np.unique(sorted_scores), boundary_thresholds])
    )

    best_threshold = float(all_thresholds[0])
    best_score = -np.inf

    for thr in all_thresholds:
        preds = (scores >= thr) if operator == ">=" else (scores > thr)
        tp, tn, fp, fn = compute_confusion_matrix_weighted(labels, preds, weights)
        s = float(metric_fn(tp, tn, fp, fn))
        if s > best_score:
            best_score = s
            best_threshold = float(thr)

    return best_threshold, float(best_score)


def optimize_scipy(
    labels: np.ndarray,
    scores: np.ndarray,
    metric: str,
    weights: np.ndarray | None = None,
    operator: str = ">=",
    method: str = "bounded",
    tol: float = 1e-6,
) -> OptimizationResult:
    """Scipy-based optimization for smooth metrics."""
    logger.debug("Using scipy optimization (%s) for %s metric", method, metric)
    labels, scores, weights = validate_binary_classification(labels, scores, weights)

    from .metrics_core import METRICS

    # All metric functions now available through registry
    metric_fn = METRICS[metric].fn

    def objective(threshold: float) -> float:
        preds = (scores >= threshold) if operator == ">=" else (scores > threshold)
        tp, tn, fp, fn = compute_confusion_matrix_weighted(labels, preds, weights)
        score = float(metric_fn(tp, tn, fp, fn))
        return -score

    eps = 1e-10  # default tolerance for boundary conditions
    score_min, score_max = float(np.min(scores)), float(np.max(scores))
    bounds = (score_min - eps, score_max + eps)

    try:
        result = optimize.minimize_scalar(
            objective, bounds=bounds, method=method, options={"xatol": tol}
        )
        # scipy's OptimizeResult is typed as a bare object in its stubs.
        optimal_threshold = float(result.x)  # pyright: ignore[reportAttributeAccessIssue]
        optimal_score = -float(result.fun)  # pyright: ignore[reportAttributeAccessIssue]
    except Exception:
        logger.warning("Scipy optimization failed, falling back to sort_scan")
        return optimize_sort_scan(labels, scores, metric, weights, operator)

    from .validation import make_binary_predictor

    return OptimizationResult(
        thresholds=np.array([optimal_threshold], dtype=float),
        scores=np.array([optimal_score], dtype=float),
        predict=make_binary_predictor(optimal_threshold, operator),
        task=Task.BINARY,
        metric=metric,
        n_classes=2,
    )


def optimize_gradient(
    labels: np.ndarray,
    scores: np.ndarray,
    metric: str,
    weights: np.ndarray | None = None,
    operator: str = ">=",
    learning_rate: float = 0.01,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> OptimizationResult:
    """Simple gradient ascent optimization (use for smooth metrics)."""
    logger.debug(
        "Using gradient optimization for %s metric (max_iter=%d)", metric, max_iter
    )
    labels, scores, weights = validate_binary_classification(labels, scores, weights)

    from .metrics_core import METRICS, is_piecewise_metric

    # All metric functions now available through registry
    metric_fn = METRICS[metric].fn

    if is_piecewise_metric(metric):
        logger.warning(
            "Gradient optimization is ineffective for piecewise-constant metrics. "
            "Use sort_scan instead."
        )

    threshold = float(np.median(scores))

    def evaluate_metric(t: float) -> float:
        preds = (scores >= t) if operator == ">=" else (scores > t)
        tp, tn, fp, fn = compute_confusion_matrix_weighted(labels, preds, weights)
        return float(metric_fn(tp, tn, fp, fn))

    # Natural bounds from the score distribution
    lo = float(np.min(scores)) - 1e-10
    hi = float(np.max(scores)) + 1e-10

    for _ in range(max_iter):
        h = 1e-8
        grad = (evaluate_metric(threshold + h) - evaluate_metric(threshold - h)) / (
            2.0 * h
        )
        if abs(grad) < tol:
            break
        threshold += learning_rate * grad
        threshold = float(np.clip(threshold, lo, hi))

    final_score = evaluate_metric(threshold)

    from .validation import make_binary_predictor

    return OptimizationResult(
        thresholds=np.array([threshold], dtype=float),
        scores=np.array([final_score], dtype=float),
        predict=make_binary_predictor(threshold, operator),
        task=Task.BINARY,
        metric=metric,
        n_classes=2,
    )


# ============================================================================
# Multiclass Optimization
# ============================================================================


def find_optimal_threshold_multiclass(
    true_labs: np.ndarray,
    pred_prob: np.ndarray,
    metric: str = "f1",
    method: str = "auto",
    average: str = "macro",
    sample_weight: np.ndarray | None = None,
    comparison: str = ">",
    tolerance: float = 1e-10,
) -> OptimizationResult:
    """Find optimal per-class thresholds for multiclass classification."""
    from .validation import validate_multiclass_classification

    true_labs, pred_prob, _ = validate_multiclass_classification(true_labs, pred_prob)

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if sample_weight.shape[0] != true_labs.shape[0]:
            raise ValueError("sample_weight must have same length as true_labs")

    n_samples, n_classes = pred_prob.shape

    if method == "coord_ascent":
        # Coordinate ascent supports weights now. Metric fixed to macro-F1.
        if metric != "f1":
            raise NotImplementedError(
                "Coordinate ascent currently supports 'f1' metric only."
            )
        if comparison != ">":
            # Argmax over shifted scores doesn't meaningfully support '>=' semantics
            raise NotImplementedError(
                "Coordinate ascent uses argmax(P - tau); '>' is required."
            )

        # Convert types for Numba
        true_labs_int32 = np.asarray(true_labs, dtype=np.int32)
        pred_prob_float64 = np.asarray(pred_prob, dtype=np.float64, order="C")
        n_samples = len(true_labs_int32)
        weights = get_sample_weights(sample_weight, n_samples)

        thresholds, best_score, _ = coordinate_ascent_kernel(
            true_labs_int32, pred_prob_float64, weights, max_iter=30, tol=1e-12
        )

        from .validation import make_margin_predictor

        scores = np.full(n_classes, best_score, dtype=float)
        return OptimizationResult(
            thresholds=thresholds.astype(float),
            scores=scores,
            predict=make_margin_predictor(thresholds),
            task=Task.MULTICLASS,
            metric=metric,
            n_classes=n_classes,
        )

    # Map method to binary optimization function
    match method:
        case "auto":
            from .metrics_core import is_piecewise_metric

            optimize_fn = (
                optimize_sort_scan if is_piecewise_metric(metric) else optimize_scipy
            )
        case "sort_scan":
            optimize_fn = optimize_sort_scan
        case "scipy":
            optimize_fn = optimize_scipy
        case "gradient":
            optimize_fn = optimize_gradient
        case _:
            optimize_fn = optimize_sort_scan

    operator = ">=" if comparison == ">=" else ">"

    if average == "micro":
        # Build flat labels by vectorization
        classes = np.arange(n_classes)
        true_binary_flat = (
            np.repeat(true_labs, n_classes) == np.tile(classes, n_samples)
        ).astype(np.int8)
        pred_prob_flat = pred_prob.ravel()

        # Flattened weights if provided
        sample_weight_flat = (
            None if sample_weight is None else np.repeat(sample_weight, n_classes)
        )

        # Call optimization function directly based on selected method
        if optimize_fn is optimize_scipy:
            result = optimize_scipy(  # preen: allow-dropped-arg
                true_binary_flat,
                pred_prob_flat,
                metric,
                sample_weight_flat,
                operator,
                tol=tolerance,
            )
        elif optimize_fn is optimize_gradient:
            result = optimize_gradient(
                true_binary_flat,
                pred_prob_flat,
                metric,
                sample_weight_flat,
                operator,
                tol=tolerance,
            )
        else:
            result = optimize_fn(
                true_binary_flat, pred_prob_flat, metric, sample_weight_flat, operator
            )
        optimal_threshold = result.thresholds[0]

        from .validation import make_multiclass_predictor

        thresholds = np.full(n_classes, optimal_threshold, dtype=float)
        scores = np.full(n_classes, result.scores[0], dtype=float)

        return OptimizationResult(
            thresholds=thresholds,
            scores=scores,
            predict=make_multiclass_predictor(thresholds, operator),
            task=Task.MULTICLASS,
            metric=metric,
            n_classes=n_classes,
        )

    # Macro/weighted/none: independent per-class thresholds (OvR)
    optimal_thresholds = np.zeros(n_classes, dtype=float)
    optimal_scores = np.zeros(n_classes, dtype=float)

    true_binary_all = np.zeros((n_samples, n_classes), dtype=np.int8)
    for c in range(n_classes):
        true_binary_all[:, c] = (true_labs == c).astype(np.int8)

    for c in range(n_classes):
        # Call optimization function directly based on selected method
        if optimize_fn is optimize_scipy:
            result = optimize_scipy(  # preen: allow-dropped-arg
                true_binary_all[:, c],
                pred_prob[:, c],
                metric,
                sample_weight,
                operator,
                tol=tolerance,
            )
        elif optimize_fn is optimize_gradient:
            result = optimize_gradient(
                true_binary_all[:, c],
                pred_prob[:, c],
                metric,
                sample_weight,
                operator,
                tol=tolerance,
            )
        else:
            result = optimize_fn(
                true_binary_all[:, c],
                pred_prob[:, c],
                metric,
                sample_weight,
                operator,
            )
        optimal_thresholds[c] = result.thresholds[0]
        optimal_scores[c] = result.scores[0]

    from .validation import make_multiclass_predictor

    return OptimizationResult(
        thresholds=optimal_thresholds,
        scores=optimal_scores,
        predict=make_multiclass_predictor(optimal_thresholds, operator),
        task=Task.MULTICLASS,
        metric=metric,
        n_classes=n_classes,
    )


# ============================================================================
# Main API Functions
# ============================================================================


def find_optimal_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    metric: str = "f1",
    weights: np.ndarray | None = None,
    strategy: str = "auto",
    operator: str = ">=",
    require_probability: bool = True,
    tolerance: float = 1e-10,
) -> OptimizationResult:
    """Simple functional interface for binary threshold optimization."""
    if require_probability:
        s = np.asarray(scores)
        if np.any((s < 0) | (s > 1)):
            raise ValueError("Scores must be in [0, 1] when require_probability=True")

    if strategy == "auto":
        from .metrics_core import is_piecewise_metric

        if is_piecewise_metric(metric):
            return optimize_sort_scan(labels, scores, metric, weights, operator)
        return optimize_scipy(labels, scores, metric, weights, operator, tol=tolerance)
    if strategy == "sort_scan":
        return optimize_sort_scan(labels, scores, metric, weights, operator)
    if strategy == "scipy":
        return optimize_scipy(labels, scores, metric, weights, operator, tol=tolerance)
    if strategy == "gradient":
        return optimize_gradient(
            labels, scores, metric, weights, operator, tol=tolerance
        )
    return optimize_sort_scan(labels, scores, metric, weights, operator)
