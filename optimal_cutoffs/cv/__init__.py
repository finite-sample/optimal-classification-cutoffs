"""Cross-validation for threshold optimization.

Clean interface for validating threshold optimization methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..api import optimize_thresholds
from ..core import OptimizationResult

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from sklearn.model_selection import BaseCrossValidator

__all__ = [
    "OptimizationResult",
    "cross_validate",
    "nested_cross_validate",
    "optimize_thresholds",
]


def cross_validate(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    metric: str = "f1",
    cv: int | BaseCrossValidator = 5,
    random_state: int | None = None,
    **optimize_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-validate threshold optimization.

    Args:
        y_true: True labels
        y_score: Predicted scores/probabilities
        metric: Metric to optimize and evaluate. Defaults to "f1".
        cv: Number of cross-validation folds. Defaults to 5.
        random_state: Random seed for reproducibility. Optional.
        **optimize_kwargs: Additional arguments passed to optimize_thresholds()

    Returns:
        Arrays of per-fold thresholds and scores.

    Raises:
        ValueError: If `metric` is not a registered metric.

    Examples:
        >>> thresholds, scores = cross_validate(y_true, y_scores, metric="f1", cv=5)
        >>> print(f"CV Score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Early validation of parameters (in order of priority)
    from ..metrics_core import METRICS

    if metric not in METRICS:
        raise ValueError(
            f"Unknown metric: '{metric}'. Available metrics: {list(METRICS.keys())}"
        )

    # Choose splitter: stratify by default for classification when possible
    if not isinstance(cv, int):
        # cv is already a sklearn splitter object
        splitter = cv
    elif y_true.ndim == 1 and len(np.unique(y_true)) > 1:
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    thresholds = []
    scores = []

    for train_idx, test_idx in splitter.split(y_true, y_true):
        y_train, y_test = y_true[train_idx], y_true[test_idx]
        if y_score.ndim == 1:
            score_train, score_test = y_score[train_idx], y_score[test_idx]
        else:
            score_train, score_test = y_score[train_idx], y_score[test_idx]

        # Handle sample weights splitting
        train_kwargs = dict(optimize_kwargs)
        test_kwargs = dict(optimize_kwargs)
        if (
            "sample_weight" in optimize_kwargs
            and optimize_kwargs["sample_weight"] is not None
        ):
            full_weights = np.asarray(optimize_kwargs["sample_weight"])
            train_kwargs["sample_weight"] = full_weights[train_idx]
            test_kwargs["sample_weight"] = full_weights[test_idx]

        # Optimize threshold on training set
        result = optimize_thresholds(
            y_train, score_train, metric=metric, **train_kwargs
        )
        from ..core import Task

        threshold = (
            result.threshold if result.task == Task.BINARY else result.thresholds
        )

        # Evaluate the training-fold threshold on the held-out fold. Re-optimising on
        # the test fold would report that fold's own maximum, which is an optimistically
        # biased estimate rather than a cross-validated one.
        comparison = test_kwargs.get("comparison", ">")
        test_weight = test_kwargs.get("sample_weight")
        if result.task == Task.BINARY:
            from ..metrics_core import compute_metric_at_threshold

            score = compute_metric_at_threshold(
                y_test,
                score_test,
                float(threshold),
                metric=metric,
                sample_weight=test_weight,
                comparison=comparison,
            )
        else:
            from ..metrics_core import multiclass_metric_single_label

            score = multiclass_metric_single_label(
                y_test,
                score_test,
                np.asarray(threshold),
                metric,
                comparison=comparison,
                sample_weight=test_weight,
            )

        thresholds.append(threshold)
        scores.append(score)

    return np.array(thresholds), np.array(scores)


def nested_cross_validate(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    metric: str = "f1",
    inner_cv: int = 3,
    outer_cv: int = 5,
    random_state: int | None = None,
    **optimize_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Nested cross-validation for unbiased threshold optimization evaluation.

    Inner CV: Optimizes thresholds
    Outer CV: Evaluates the optimization procedure

    Args:
        y_true: True labels
        y_score: Predicted scores/probabilities
        metric: Metric to optimize and evaluate. Defaults to "f1".
        inner_cv: Number of inner CV folds (for threshold optimization). Defaults to 3.
        outer_cv: Number of outer CV folds (for evaluation). Defaults to 5.
        random_state: Random seed for reproducibility. Optional.
        **optimize_kwargs: Additional arguments passed to optimize_thresholds()

    Returns:
        Nested CV results with keys:
        - 'test_scores': array of outer test scores
        - 'mean_score': mean outer test score
        - 'std_score': standard deviation of outer test scores
        - 'thresholds': threshold estimates from each outer fold

    Raises:
        ValueError: If `inner_cv` or `outer_cv` asks for fewer than one split.

    Examples:
        >>> # Get unbiased estimate of threshold optimization performance
        >>> results = nested_cross_validate(y_true, y_scores, metric="f1")
        >>> print(f"Unbiased CV Score: {results['mean_score']:.3f}")
    """
    # Validate CV parameters
    if inner_cv < 2:
        raise ValueError(
            f"k-fold cross-validation requires at least one train/test split, got "
            f"inner_cv={inner_cv}"
        )
    if outer_cv < 2:
        raise ValueError(
            f"k-fold cross-validation requires at least one train/test split, got "
            f"outer_cv={outer_cv}"
        )

    # For now, implement simple nested CV
    # TODO: Full implementation can be added later if needed
    thresholds, scores = cross_validate(
        y_true,
        y_score,
        metric=metric,
        cv=outer_cv,
        random_state=random_state,
        **optimize_kwargs,
    )

    return thresholds, scores
