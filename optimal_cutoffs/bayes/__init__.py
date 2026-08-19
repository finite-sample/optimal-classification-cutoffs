"""Bayes-optimal decision making.

Clean interface for cost-based optimization without the complexity
of the original BayesOptimal class hierarchy.
"""

import numpy as np
from numpy.typing import NDArray

from ..bayes_core import BayesOptimal, UtilitySpec
from ..core import OptimizationResult


def threshold(
    cost_fp: float,
    cost_fn: float,
    benefit_tp: float = 0.0,
    benefit_tn: float = 0.0,
) -> float:
    """Compute binary Bayes-optimal threshold from costs and benefits.

    Args:
        cost_fp: Cost of false positive (predicting positive when actually negative)
        cost_fn: Cost of false negative (predicting negative when actually positive)
        benefit_tp: Benefit of true positive (predicting positive correctly)
        benefit_tn: Benefit of true negative (predicting negative correctly)

    Returns:
        Optimal threshold τ* = (benefit_tn + cost_fp) / [(benefit_tp + cost_fn) +
        (benefit_tn + cost_fp)]

    Examples:
        >>> # FN costs 5x more than FP
        >>> t = threshold(cost_fp=1.0, cost_fn=5.0)
        >>> # Will be < 0.5 (more conservative, avoids costly false negatives)
    """
    from ..bayes_core import bayes_optimal_threshold

    return bayes_optimal_threshold(
        fp_cost=cost_fp,
        fn_cost=cost_fn,
        tp_benefit=benefit_tp,
        tn_benefit=benefit_tn,
    ).thresholds[0]


def thresholds_from_costs(
    fp_costs: NDArray | list[float], fn_costs: NDArray | list[float], **kwargs
) -> np.ndarray:
    """Compute per-class Bayes-optimal thresholds from OvR costs.

    Args:
        fp_costs: False positive costs per class
        fn_costs: False negative costs per class
        **kwargs: Forwarded to
            :func:`~optimal_cutoffs.bayes_core.bayes_thresholds_from_costs`.

    Returns:
        Per-class optimal thresholds

    Examples:
        >>> # Different costs per class
        >>> fp_costs = [1.0, 2.0, 0.5]  # Class 1 FP costs 2x more
        >>> fn_costs = [5.0, 1.0, 10.0] # Class 2 FN costs 10x more
        >>> thresholds = thresholds_from_costs(fp_costs, fn_costs)
    """
    from ..bayes_core import bayes_thresholds_from_costs

    return bayes_thresholds_from_costs(fp_costs, fn_costs, **kwargs).thresholds


def policy(cost_matrix: NDArray) -> OptimizationResult:
    """Create Bayes-optimal decision policy from cost matrix.

    This is for general decision making where thresholds aren't
    the right abstraction.

    Args:
        cost_matrix: Cost matrix (n_classes, n_actions)
            cost_matrix[i, j] = cost of taking action j when true class is i

    Returns:
        Policy with .predict() method (no .thresholds)

    Examples:
        >>> costs = [[0, 1, 10], [5, 0, 1], [50, 10, 0]]
        >>> policy = policy(costs)
        >>> decisions = policy.predict(probabilities)
    """
    from ..bayes_core import bayes_optimal_decisions

    return bayes_optimal_decisions(
        probabilities=None,  # Will be provided later via .predict()
        cost_matrix=cost_matrix,
    )


# Note: BayesOptimal and UtilitySpec imported at top for power users

__all__ = [
    "BayesOptimal",
    "UtilitySpec",
    "policy",
    "threshold",
    "thresholds_from_costs",
]
