"""Expected metric optimization (Dinkelbach method).

Advanced algorithms for optimizing expected values of metrics
under probability distributions.
"""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ..core import OptimizationResult


def dinkelbach_fbeta_multilabel(
    y_score: ArrayLike,
    *,
    beta: float = 1.0,
    average: Literal["macro", "micro", "weighted"] = "macro",
    sample_weight: ArrayLike | None = None,
    **kwargs,
) -> OptimizationResult:
    """Expected F-beta optimization for multilabel classification.

    Moved from dinkelbach_expected_fbeta_multilabel().
    Requires calibrated probabilities.
    """
    from ..expected import dinkelbach_expected_fbeta_multilabel

    # Convert to ndarray for type safety
    y_score_arr = np.asarray(y_score)
    sample_weight_arr = np.asarray(sample_weight) if sample_weight is not None else None

    return dinkelbach_expected_fbeta_multilabel(
        y_score_arr,
        beta=beta,
        average=average,
        sample_weight=sample_weight_arr,
        **kwargs,
    )
