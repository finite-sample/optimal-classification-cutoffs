"""Multiclass classification algorithms.

Specific algorithms moved from the top-level API.
"""

from numpy.typing import ArrayLike

from ..core import OptimizationResult


def ovr_independent(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    metric: str = "f1",
    method: str = "auto",
    sample_weight: ArrayLike | None = None,
    **kwargs,
) -> OptimizationResult:
    """Independent per-class optimization (One-vs-Rest).

    Can predict 0, 1, or multiple classes per sample.
    Moved from optimize_ovr_independent().
    """
    from ..multiclass import optimize_ovr_independent

    return optimize_ovr_independent(
        y_true,
        y_score,
        metric=metric,
        method=method,
        sample_weight=sample_weight,
        **kwargs,
    )


def ovr_margin(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    metric: str = "f1",
    max_iter: int = 30,
    sample_weight: ArrayLike | None = None,
    **kwargs,
) -> OptimizationResult:
    """Margin-based prediction with coordinate ascent.

    Ensures exactly one class predicted per sample (single-label).
    Uses argmax(p_j - τ_j) decision rule.
    Moved from optimize_ovr_margin().
    """
    from ..multiclass import optimize_ovr_margin

    return optimize_ovr_margin(
        y_true,
        y_score,
        metric=metric,
        max_iter=max_iter,
        sample_weight=sample_weight,
        **kwargs,
    )


def coordinate_ascent(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    metric: str = "f1",
    max_iter: int = 30,
    sample_weight: ArrayLike | None = None,
    **kwargs,
) -> OptimizationResult:
    """Coordinate ascent for coupled multiclass optimization.

    Alias for ovr_margin() - same algorithm.
    """
    return ovr_margin(
        y_true,
        y_score,
        metric=metric,
        max_iter=max_iter,
        sample_weight=sample_weight,
        **kwargs,
    )


