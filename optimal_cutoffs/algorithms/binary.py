"""Binary classification algorithms.

Specific algorithms moved from the top-level API.
"""

from numpy.typing import ArrayLike

from ..core import OptimizationResult


def exact_f1(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    beta: float = 1.0,
    sample_weight: ArrayLike | None = None,
    **kwargs,
) -> OptimizationResult:
    """Exact F-beta optimization using O(n log n) sort-and-scan.

    Moved from optimize_f1_binary().
    """
    from ..binary import optimize_f1_binary

    return optimize_f1_binary(
        y_true, y_score, beta=beta, sample_weight=sample_weight, **kwargs
    )


def scipy_optimize(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    metric: str = "f1",
    sample_weight: ArrayLike | None = None,
    **kwargs,
) -> OptimizationResult:
    """General metric optimization using scipy.optimize.

    Moved from optimize_metric_binary() with method="minimize".
    """
    from ..binary import optimize_metric_binary

    return optimize_metric_binary(
        y_true,
        y_score,
        metric=metric,
        method="minimize",
        sample_weight=sample_weight,
        **kwargs,
    )


def gradient_ascent(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    metric: str = "f1",
    sample_weight: ArrayLike | None = None,
    **kwargs,
) -> OptimizationResult:
    """Gradient ascent optimization.

    Moved from optimize_metric_binary() with method="gradient".
    """
    from ..binary import optimize_metric_binary

    return optimize_metric_binary(
        y_true,
        y_score,
        metric=metric,
        method="gradient",
        sample_weight=sample_weight,
        **kwargs,
    )


