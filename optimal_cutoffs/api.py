"""Main API for threshold optimization.

Two canonical functions:
- optimize_thresholds(): For threshold-based optimization
- optimize_decisions(): For cost-matrix based decisions (no thresholds)
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import (
    Average,
    OptimizationResult,
    Task,
    infer_task_with_explanation,
    select_average_with_explanation,
    select_method_with_explanation,
)


def optimize_thresholds(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    metric: str = "f1",
    task: Task = Task.AUTO,
    average: Average = Average.AUTO,
    method: str = "auto",
    mode: str = "empirical",
    sample_weight: ArrayLike | None = None,
    utility: Mapping[str, float | int] | None = None,
    fp_costs: ArrayLike | None = None,
    fn_costs: ArrayLike | None = None,
    comparison: str = ">",
    tolerance: float = 1e-10,
    **kwargs,
) -> OptimizationResult:
    """Find optimal thresholds for classification problems.

    This is THE canonical entry point for threshold optimization.
    Auto-detects problem type and selects appropriate algorithms.

    Parameters
    ----------
    y_true
        True labels
    y_score
        Predicted scores/probabilities
        - Binary: 1D array of scores
        - Multiclass: 2D array (n_samples, n_classes)
        - Multilabel: 2D array (n_samples, n_labels)
    metric
        Metric to optimize ("f1", "precision", "recall", "accuracy", etc.)
    task
        Problem type. AUTO infers from data shape and probability sums.
    average
        Averaging strategy for multiclass/multilabel. AUTO selects sensible default.
    method
        Optimization algorithm. AUTO selects best method per task+metric.
    mode
        "empirical" (standard) or "expected" (requires calibrated probabilities)
    sample_weight
        Sample weights
    utility
        Utility specification for binary Bayes optimization with keys "tp", "tn", "fp", "fn".
        Required when mode="bayes" for binary classification.
    fp_costs
        Per-class false positive costs for multiclass Bayes optimization.
        Required when mode="bayes" for multiclass classification.
    fn_costs
        Per-class false negative costs for multiclass Bayes optimization.
        Required when mode="bayes" for multiclass classification.
    comparison
        Comparison operator for threshold. Must be ">" or ">=".
    tolerance
        Numerical tolerance for optimization.
    **kwargs
        Additional keyword arguments passed to optimization algorithms.

    Returns
    -------
    OptimizationResult
        Result with .thresholds, .predict(), and explanation of auto-selections

    Raises
    ------
    TypeError
        If 'bayes' is passed as a keyword argument (deprecated).
    ValueError
        If mode='bayes' requires utility parameter but none provided.
        If comparison operator is not '>' or '>='.
        If mode='expected' with unsupported metric.
        If method is deprecated ('dinkelbach', 'smart_brute').
        If unknown metric name is provided.
        If y_true required for empirical mode but not provided.

    Examples
    --------
    >>> # Binary classification - simple case
    >>> result = optimize_thresholds(y_true, y_scores, metric="f1")
    >>> print(f"Optimal threshold: {result.threshold}")

    >>> # Multiclass classification
    >>> result = optimize_thresholds(y_true, y_probs, metric="f1")
    >>> print(f"Per-class thresholds: {result.thresholds}")
    >>> print(f"Task inferred as: {result.task.value}")

    >>> # Explicit control when needed
    >>> result = optimize_thresholds(
    ...     y_true, y_probs,
    ...     metric="precision",
    ...     task=Task.MULTICLASS,
    ...     average=Average.MACRO
    ... )
    """
    # Early validation for mode-specific requirements
    # Bayes mode requires either 'utility' (binary) or 'fp_costs'/'fn_costs' (multiclass)
    if mode == "bayes":
        has_utility = utility is not None
        has_costs = fp_costs is not None and fn_costs is not None
        if not has_utility and not has_costs:
            raise ValueError(
                "mode='bayes' requires 'utility' (binary) or 'fp_costs'/'fn_costs' (multiclass)"
            )

    # Check for deprecated parameters
    if "bayes" in kwargs:
        raise TypeError(
            "optimize_thresholds() got an unexpected keyword argument 'bayes'"
        )

    # Validate comparison operator
    if comparison not in [">", ">="]:
        raise ValueError(
            f"Invalid comparison operator: {comparison}. Must be '>' or '>='"
        )

    # Validate expected mode requirements
    if mode == "expected" and metric not in ["f1", "fbeta"]:
        raise ValueError("mode='expected' currently supports F-beta only")

    # Handle deprecated method mappings
    method_mappings = {
        "unique_scan": "sort_scan",  # Map deprecated unique_scan to sort_scan
    }
    if method in method_mappings:
        method = method_mappings[method]

    # Validate deprecated/invalid methods
    deprecated_methods = [
        "dinkelbach",
        "smart_brute",
    ]  # Reject these deprecated methods
    if method in deprecated_methods:
        raise ValueError(f"Invalid optimization method: '{method}' is deprecated")

    # Validate metric exists
    from .metrics_core import METRICS

    if metric not in METRICS:
        raise ValueError(
            f"Unknown metric: '{metric}'. Available metrics: {list(METRICS.keys())}"
        )

    # Convert string parameters to proper enums
    if isinstance(task, str):
        task = Task(task.lower())
    if isinstance(average, str):
        average = Average(average.lower())

    # Check that empirical mode has true labels
    if mode == "empirical" and y_true is None:
        raise ValueError("y_true required for empirical optimization")

    # For Bayes and Expected modes, y_true is not needed
    # Handle None inputs gracefully for these cases
    if (mode == "bayes" or mode == "expected") and y_true is None:
        # Create dummy y_true for downstream processing
        y_score = np.asarray(y_score)
        y_true = np.zeros(len(y_score), dtype=int)  # Dummy labels
    else:
        # Convert inputs normally
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)

    # Track all notes and warnings for explainability
    all_notes = []
    all_warnings = []

    # 1. Task inference
    match task:
        case Task.AUTO:
            inferred_task, task_notes, task_warnings = infer_task_with_explanation(
                y_true, y_score
            )
            all_notes.extend(task_notes)
            all_warnings.extend(task_warnings)
        case _:
            inferred_task = task
            all_notes.append(f"Using explicit task: {task.value}")

    # 2. Average strategy selection
    match average:
        case Average.AUTO:
            inferred_average, avg_notes = select_average_with_explanation(
                inferred_task, metric
            )
            all_notes.extend(avg_notes)
        case _:
            inferred_average = average
            all_notes.append(f"Using explicit averaging: {average.value}")

    # 3. Method selection
    if method == "auto":
        n_samples = len(y_true)
        inferred_method, method_notes = select_method_with_explanation(
            inferred_task, metric, n_samples
        )
        all_notes.extend(method_notes)
    else:
        inferred_method = method
        all_notes.append(f"Using explicit method: {method}")

    # 4. Handle auto method selection edge cases
    final_method = inferred_method
    if (
        method == "auto"
        and inferred_method == "coord_ascent"
        and inferred_task == Task.MULTICLASS
        and comparison == ">="
    ):
        # Auto-selected coord_ascent but comparison=">=" provided, fall back to independent
        final_method = "independent"
        all_notes.append(
            "Switched from coord_ascent to independent due to comparison='>=' requirement"
        )

    # Route to appropriate implementation
    result = _route_to_implementation(
        y_true,
        y_score,
        task=inferred_task,
        metric=metric,
        average=inferred_average,
        method=final_method,
        mode=mode,
        sample_weight=sample_weight,
        utility=utility,
        fp_costs=fp_costs,
        fn_costs=fn_costs,
        comparison=comparison,
        tolerance=tolerance,
        **kwargs,
    )

    # 5. Add explainability metadata
    result.task = inferred_task
    result.method = final_method
    result.average = inferred_average
    result.notes = all_notes
    result.warnings = all_warnings
    result.metric = metric

    return result


def optimize_decisions(
    y_score: ArrayLike,
    cost_matrix: ArrayLike,
    **kwargs,
) -> OptimizationResult:
    """Find optimal decisions using cost matrix (no thresholds).

    For problems where thresholds aren't the right abstraction.
    Uses Bayes-optimal decision rule: argmin_action E[cost | probabilities].

    Parameters
    ----------
    y_score
        Predicted probabilities (n_samples, n_classes)
    cost_matrix
        Cost matrix (n_classes, n_actions) or (n_classes, n_classes)
        cost_matrix[i, j] = cost of predicting action j when true class is i
    **kwargs
        Additional keyword arguments passed to the Bayes optimal decision function.

    Returns
    -------
    OptimizationResult
        Result with .predict() function (no .thresholds)

    Examples
    --------
    >>> # Cost matrix: rows=true class, cols=predicted class
    >>> costs = [[0, 1, 10], [5, 0, 1], [50, 10, 0]]  # FN costs 5x more than FP
    >>> result = optimize_decisions(y_probs, costs)
    >>> y_pred = result.predict(y_probs_test)
    """
    from .bayes_core import bayes_optimal_decisions

    return bayes_optimal_decisions(
        np.asarray(y_score), cost_matrix=np.asarray(cost_matrix), **kwargs
    )


def _route_to_implementation(
    y_true: NDArray,
    y_score: NDArray,
    *,
    task: Task,
    metric: str,
    average: Average,
    method: str,
    mode: str,
    sample_weight: NDArray | None = None,
    utility: Mapping[str, float | int] | None = None,
    fp_costs: ArrayLike | None = None,
    fn_costs: ArrayLike | None = None,
    comparison: str = ">",
    tolerance: float = 1e-10,
    **kwargs,
) -> OptimizationResult:
    """Route to appropriate implementation based on task and method."""

    match task:
        case Task.BINARY:
            return _optimize_binary(
                y_true,
                y_score,
                metric=metric,
                method=method,
                mode=mode,
                sample_weight=sample_weight,
                utility=utility,
                comparison=comparison,
                tolerance=tolerance,
                **kwargs,
            )

        case Task.MULTICLASS:
            return _optimize_multiclass(
                y_true,
                y_score,
                metric=metric,
                average=average,
                method=method,
                mode=mode,
                sample_weight=sample_weight,
                fp_costs=fp_costs,
                fn_costs=fn_costs,
                comparison=comparison,
                tolerance=tolerance,
                **kwargs,
            )

        case Task.MULTILABEL:
            return _optimize_multilabel(
                y_true,
                y_score,
                metric=metric,
                average=average,
                method=method,
                mode=mode,
                sample_weight=sample_weight,
                comparison=comparison,
                tolerance=tolerance,
                **kwargs,
            )

        case _:
            raise ValueError(f"Unknown task: {task}")


def _optimize_binary(
    y_true: NDArray,
    y_score: NDArray,
    *,
    metric: str,
    method: str,
    mode: str,
    sample_weight: NDArray | None = None,
    utility: Mapping[str, float | int] | None = None,
    comparison: str = ">",
    tolerance: float = 1e-10,
    **kwargs,
) -> OptimizationResult:
    """Route binary optimization to appropriate algorithm."""

    if mode == "expected":
        from .expected import dinkelbach_expected_fbeta_binary

        return dinkelbach_expected_fbeta_binary(
            y_score, comparison=comparison, **kwargs
        )

    if mode == "bayes":
        from .bayes import threshold as bayes_threshold
        from .core import OptimizationResult, Task

        # Extract costs/benefits from utility dictionary. All four documented keys
        # ("tp", "tn", "fp", "fn") enter the closed form
        #   tau* = (u_tn - u_fp) / [(u_tp - u_fn) + (u_tn - u_fp)]
        util = utility or {}
        cost_fp = -util.get(
            "fp", 0
        )  # Convert from utility (negative cost) to positive cost
        cost_fn = -util.get(
            "fn", 0
        )  # Convert from utility (negative cost) to positive cost
        benefit_tp = util.get("tp", 0)
        benefit_tn = util.get("tn", 0)

        # Compute Bayes optimal threshold
        optimal_thresh = bayes_threshold(
            cost_fp=cost_fp,
            cost_fn=cost_fn,
            benefit_tp=benefit_tp,
            benefit_tn=benefit_tn,
        )

        # Return OptimizationResult format
        def predict_fn(scores):
            return (scores >= optimal_thresh).astype(int)

        return OptimizationResult(
            thresholds=np.array([optimal_thresh]),
            scores=np.array([np.nan]),  # Bayes optimization doesn't produce a score
            predict=predict_fn,
            task=Task.BINARY,
        )

    # Empirical mode
    # Check if utility-based optimization is requested
    if utility is not None:
        from .binary import optimize_utility_binary

        return optimize_utility_binary(
            y_true, y_score, utility=dict(utility), sample_weight=sample_weight
        )

    match method:
        case "sort_scan":
            from .piecewise import optimal_threshold_sortscan

            # Convert comparison parameter to inclusive for sort_scan method
            inclusive = comparison == ">="

            return optimal_threshold_sortscan(
                y_true,
                y_score,
                metric=metric,
                sample_weight=sample_weight,
                inclusive=inclusive,
                tolerance=tolerance,
                **kwargs,
            )

        case "minimize":
            from .binary import optimize_metric_binary

            return optimize_metric_binary(
                y_true,
                y_score,
                metric=metric,
                method="minimize",
                sample_weight=sample_weight,
                comparison=comparison,
                tolerance=tolerance,
                **kwargs,
            )

        case "gradient":
            from .binary import optimize_metric_binary

            return optimize_metric_binary(
                y_true,
                y_score,
                metric=metric,
                method="gradient",
                sample_weight=sample_weight,
                comparison=comparison,
                tolerance=tolerance,
                **kwargs,
            )

        case _:
            raise ValueError(
                f"Invalid optimization method: '{method}' is not supported for binary classification"
            )


def _optimize_multiclass(
    y_true: NDArray,
    y_score: NDArray,
    *,
    metric: str,
    average: Average,
    method: str,
    mode: str,
    sample_weight: NDArray | None = None,
    fp_costs: ArrayLike | None = None,
    fn_costs: ArrayLike | None = None,
    comparison: str = ">",
    tolerance: float = 1e-10,
    **kwargs,
) -> OptimizationResult:
    """Route multiclass optimization to appropriate algorithm."""

    if mode == "bayes":
        from .bayes_core import bayes_thresholds_from_costs

        if fp_costs is None or fn_costs is None:
            raise ValueError("Bayes mode requires 'fp_costs' and 'fn_costs' arrays")

        return bayes_thresholds_from_costs(
            np.asarray(fp_costs),
            np.asarray(fn_costs),
        )

    if mode == "expected":
        from .expected import expected_optimize_multiclass

        # Convert Average enum to literal string
        avg_literal = average.value if average != Average.AUTO else "macro"
        return expected_optimize_multiclass(
            y_score,
            metric=metric,
            average=avg_literal,  # type: ignore[arg-type]
            sample_weight=sample_weight,
            **kwargs,
        )

    # Empirical mode
    match method:
        case "coord_ascent":
            # coord_ascent only supports ">" comparison
            if comparison == ">=":
                raise NotImplementedError("'>' is required for coord_ascent method")
            else:
                from .multiclass import optimize_ovr_margin

                return optimize_ovr_margin(
                    y_true,
                    y_score,
                    metric=metric,
                    sample_weight=sample_weight,
                    comparison=comparison,
                    tolerance=tolerance,
                    **kwargs,
                )

        case "independent":
            from .multiclass import optimize_ovr_independent

            return optimize_ovr_independent(
                y_true,
                y_score,
                metric=metric,
                sample_weight=sample_weight,
                comparison=comparison,
                tolerance=tolerance,
                **kwargs,
            )

        case "micro":
            from .multiclass import optimize_micro_multiclass

            return optimize_micro_multiclass(
                y_true,
                y_score,
                metric=metric,
                sample_weight=sample_weight,
                comparison=comparison,
                tolerance=tolerance,
                **kwargs,
            )

        case "sort_scan":
            # For multiclass, sort_scan should route to appropriate OvR method
            from .multiclass import optimize_ovr_independent

            return optimize_ovr_independent(
                y_true,
                y_score,
                metric=metric,
                sample_weight=sample_weight,
                comparison=comparison,
                tolerance=tolerance,
                **kwargs,
            )

        case "minimize":
            # minimize method for multiclass routes to independent optimization
            from .multiclass import optimize_ovr_independent

            return optimize_ovr_independent(
                y_true,
                y_score,
                metric=metric,
                sample_weight=sample_weight,
                comparison=comparison,
                tolerance=tolerance,
                **kwargs,
            )

        case _:
            raise ValueError(
                f"Invalid optimization method: '{method}' is not supported for multiclass classification"
            )


def _optimize_multilabel(
    y_true: NDArray,
    y_score: NDArray,
    *,
    metric: str,
    average: Average,
    method: str,
    mode: str,
    sample_weight: NDArray | None = None,
    comparison: str = ">",
    tolerance: float = 1e-10,
    **kwargs,
) -> OptimizationResult:
    """Route multilabel optimization to appropriate algorithm."""

    if mode == "expected":
        from .expected import dinkelbach_expected_fbeta_multilabel

        # Convert Average enum to literal string
        avg_literal = average.value if average != Average.AUTO else "macro"
        return dinkelbach_expected_fbeta_multilabel(
            y_score,
            average=avg_literal,  # type: ignore[arg-type]
            comparison=comparison,
            **kwargs,
        )

    # Empirical mode
    match (method, average):
        case ("independent", Average.MACRO):
            from .multilabel import optimize_macro_multilabel

            return optimize_macro_multilabel(
                y_true,
                y_score,
                metric=metric,
                sample_weight=sample_weight,
                comparison=comparison,
                tolerance=tolerance,
                **kwargs,
            )

        case ("coordinate_ascent", Average.MICRO):
            from .multilabel import optimize_micro_multilabel

            return optimize_micro_multilabel(
                y_true,
                y_score,
                metric=metric,
                sample_weight=sample_weight,
                comparison=comparison,
                tolerance=tolerance,
                **kwargs,
            )

        case _:
            # Fallback to general multilabel router
            from .multilabel import optimize_multilabel

            return optimize_multilabel(
                y_true,
                y_score,
                metric=metric,
                average=average.value,
                sample_weight=sample_weight,
                comparison=comparison,
                tolerance=tolerance,
                **kwargs,
            )
