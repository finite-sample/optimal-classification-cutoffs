"""Core types and result objects for threshold optimization.

Clean design focused on explainable auto-selection and consistent interfaces.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Task(Enum):
    """Classification task types."""

    AUTO = "auto"
    BINARY = "binary"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"


class Average(Enum):
    """Metric averaging strategies."""

    AUTO = "auto"
    MACRO = "macro"
    MICRO = "micro"
    WEIGHTED = "weighted"
    NONE = "none"


@dataclass
class OptimizationResult:
    """Result from threshold optimization with explainable auto-selection."""

    # ============== CORE RESULT ==============
    thresholds: NDArray[np.float64]  # Always array: [0.5] or [0.3, 0.7, 0.4]
    scores: NDArray[np.float64]  # Corresponding metric values
    predict: Callable[[NDArray], NDArray] = field(repr=False)

    # ============== EXPLAINABLE AUTO-SELECTION ==============
    task: Task = Task.AUTO  # What was actually solved
    method: str = "auto"  # Algorithm actually used
    average: Average = Average.AUTO  # Averaging actually applied
    notes: list[str] = field(
        default_factory=list
    )  # ["Inferred multiclass from prob sums"]
    warnings: list[str] = field(
        default_factory=list
    )  # ["Expected mode needs calibrated probs"]

    # ============== METADATA ==============
    metric: str = "f1"
    n_classes: int = 2
    diagnostics: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def threshold(self) -> float:
        """Binary threshold. Raises for multiclass."""
        match self.task:
            case Task.BINARY:
                return float(self.thresholds[0])
            case _:
                raise ValueError(f"Use .thresholds for {self.task.value} problems")

    @property
    def score(self) -> float:
        """Summary score (mean for multiclass)."""
        return float(np.mean(self.scores))

    def __repr__(self) -> str:
        """Clean representation showing what matters."""
        match self.task:
            case Task.BINARY:
                return f"OptimizationResult(threshold={self.threshold:.3f}, {self.metric}={self.score:.3f})"
            case _:
                return f"OptimizationResult(task={self.task.value}, {self.metric}={self.score:.3f})"


def infer_task_with_explanation(
    y_true: ArrayLike, y_score: ArrayLike
) -> tuple[Task, list[str], list[str]]:
    """Infer task type with detailed explanation.

    Parameters
    ----------
    y_true
        True labels (not currently used for inference, reserved for future)
    y_score
        Predicted scores or probabilities used for task inference

    Returns
    -------
    task : Task
        Inferred task type
    notes : list[str]
        Explanation of inference logic
    warnings : list[str]
        Any assumptions or caveats

    Raises
    ------
    ValueError
        If y_score shape cannot be interpreted for task inference.
    """
    y_score = np.asarray(y_score)
    notes: list[str] = []
    warnings: list[str] = []

    if y_score.ndim == 1:
        notes.append("Detected 1D scores → binary classification")
        return Task.BINARY, notes, warnings

    if y_score.ndim == 2:
        n_samples, n_outputs = y_score.shape

        if n_outputs == 1:
            notes.append("Detected 2D scores with 1 column → binary classification")
            return Task.BINARY, notes, warnings

        if n_outputs == 2:
            # Check if probabilities sum to 1
            prob_sums = np.sum(y_score, axis=1)
            if np.allclose(prob_sums, 1.0, rtol=0.05):
                notes.append("Probabilities sum to 1 → multiclass classification")
                return Task.MULTICLASS, notes, warnings
            else:
                notes.append("Probabilities don't sum to 1 → multilabel classification")
                warnings.append("Assuming independent binary labels")
                return Task.MULTILABEL, notes, warnings

        # n_outputs > 2
        prob_sums = np.sum(y_score, axis=1)
        if np.allclose(prob_sums, 1.0, rtol=0.05):
            notes.append(
                f"Probabilities sum to 1 with {n_outputs} classes → multiclass"
            )
            return Task.MULTICLASS, notes, warnings
        else:
            notes.append(
                f"Probabilities don't sum to 1 with {n_outputs} outputs → multilabel"
            )
            warnings.append("Assuming independent binary labels")
            return Task.MULTILABEL, notes, warnings

    raise ValueError(f"Cannot infer task from shape {y_score.shape}")


def select_method_with_explanation(
    task: Task, metric: str, n_samples: int
) -> tuple[str, list[str]]:
    """Select optimization method with explanation.

    Parameters
    ----------
    task
        Classification task type
    metric
        Metric to optimize
    n_samples
        Number of samples (may influence method selection)

    Returns
    -------
    method : str
        Selected method name
    notes : list[str]
        Explanation of method selection

    Raises
    ------
    ValueError
        If task type is unknown or not supported.
    """
    notes = []

    match task:
        case Task.BINARY:
            if metric in ["f1", "precision", "recall"]:
                notes.append(f"Using O(n log n) exact optimization for {metric}")
                return "sort_scan", notes
            else:
                notes.append(f"Using scipy optimization for {metric}")
                return "minimize", notes

        case Task.MULTICLASS:
            if metric == "f1":
                notes.append("Using coordinate ascent for coupled multiclass F1")
                return "coord_ascent", notes
            else:
                notes.append(f"Using independent per-class optimization for {metric}")
                return "independent", notes

        case Task.MULTILABEL:
            notes.append(f"Using independent per-label optimization for {metric}")
            return "independent", notes

        case _:
            raise ValueError(f"Unknown task: {task}")


def select_average_with_explanation(
    task: Task, metric: str
) -> tuple[Average, list[str]]:
    """Select averaging strategy with explanation.

    Parameters
    ----------
    task
        Classification task type
    metric
        Metric to optimize

    Returns
    -------
    average : Average
        Selected averaging strategy
    notes : list[str]
        Explanation of averaging selection

    Raises
    ------
    ValueError
        If task type is unknown or not supported.
    """
    notes = []

    match task:
        case Task.BINARY:
            notes.append("Binary classification → no averaging needed")
            return Average.NONE, notes

        case Task.MULTICLASS | Task.MULTILABEL:
            if metric in ["f1", "precision", "recall"]:
                notes.append(
                    f"Using macro averaging for {metric} (balanced across classes)"
                )
                return Average.MACRO, notes
            else:
                notes.append(f"Using macro averaging for {metric}")
                return Average.MACRO, notes

        case _:
            raise ValueError(f"Unknown task: {task}")
