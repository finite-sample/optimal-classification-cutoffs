"""Bayes-optimal decisions and thresholds for classification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property
from typing import TYPE_CHECKING, Self

import numpy as np

from .core import OptimizationResult, Task

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ============================================================================
# Utility Specification
# ============================================================================


@dataclass(frozen=True, slots=True)
class UtilitySpec:
    """Complete utility specification for decision theory approaches."""

    tp_utility: float = 1.0
    tn_utility: float = 1.0
    fp_utility: float = -1.0
    fn_utility: float = -1.0

    # Note: compute_utility method temporarily removed to simplify migration

    @classmethod
    def from_costs(cls, fp_cost: float, fn_cost: float) -> Self:
        """Create from misclassification costs (converted to negative utilities)."""
        if not np.isfinite(fp_cost) or not np.isfinite(fn_cost):
            raise ValueError("Costs must be finite")
        return cls(
            tp_utility=0.0,
            tn_utility=0.0,
            fp_utility=-abs(fp_cost),
            fn_utility=-abs(fn_cost),
        )

    @classmethod
    def from_dict(cls, utility_dict: dict[str, float]) -> Self:
        """Create from dictionary with keys 'tp', 'tn', 'fp', 'fn'.

        Missing keys default to 0.0.
        """
        valid_keys = {"tp", "tn", "fp", "fn"}

        # Check for unknown keys
        unknown_keys = set(utility_dict.keys()) - valid_keys
        if unknown_keys:
            raise ValueError(
                f"Unknown utility keys: {unknown_keys}. Valid keys: {valid_keys}"
            )

        # Use defaults for missing keys
        defaults = {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}
        full_dict = {**defaults, **utility_dict}

        # Validate all values are finite
        for key, value in full_dict.items():
            if not np.isfinite(value):
                raise ValueError(
                    f"Utility value for '{key}' must be finite, got {value}"
                )

        return cls(
            tp_utility=full_dict["tp"],
            tn_utility=full_dict["tn"],
            fp_utility=full_dict["fp"],
            fn_utility=full_dict["fn"],
        )


# ============================================================================
# Core Abstractions
# ============================================================================


class DecisionRule(Enum):
    """How to make decisions from utilities."""

    THRESHOLD = auto()  # Binary threshold on probability
    ARGMAX = auto()  # Argmax of expected utilities
    MARGIN = auto()  # Argmax of margin (p - threshold)


@dataclass(frozen=True)
class BayesOptimal:
    """Unified Bayes-optimal decision maker."""

    utility: UtilitySpec | NDArray[np.float64]

    def __post_init__(self):
        """Validate utility specification."""
        if isinstance(self.utility, np.ndarray):
            if self.utility.ndim != 2:
                raise ValueError(f"Utility matrix must be 2D, got {self.utility.ndim}D")
            if not np.all(np.isfinite(self.utility)):
                raise ValueError("Utility matrix must contain finite values")

    @cached_property
    def is_binary(self) -> bool:
        """Check if this is a binary problem."""
        if isinstance(self.utility, UtilitySpec):
            return True
        if self.utility is None:
            raise ValueError("mode='bayes' requires utility parameter")
        return self.utility.shape == (2, 2)

    @cached_property
    def decision_rule(self) -> DecisionRule:
        """Determine optimal decision rule."""
        if self.is_binary:
            return DecisionRule.THRESHOLD
        if isinstance(self.utility, np.ndarray):
            # Square matrix -> standard classification
            if self.utility.shape[0] == self.utility.shape[1]:
                return DecisionRule.ARGMAX
            # More decisions than classes -> includes abstain
            return DecisionRule.ARGMAX
        return DecisionRule.MARGIN

    def compute_threshold(self) -> float:
        """Compute optimal threshold for binary case (only valid when D > 0).

        Uses the correct formula:
        τ* = (u_tn - u_fp) / [(u_tp - u_fn) + (u_tn - u_fp)]

        Returns:
            Optimal probability threshold (not clipped to [0,1])

        Raises:
            ValueError: If D <= 0, callers must use margin-based decision instead
        """
        if not self.is_binary:
            raise ValueError("Thresholds only defined for binary problems")

        _A, B, D = self._binary_params()

        # Only valid for D > 0; for D <= 0 callers must use the margin-based decision.
        if D <= 1e-12:
            raise ValueError(
                "compute_threshold is only valid when (tp-fn)+(tn-fp) > 0. "
                "Use margin-based decision for D <= 0."
            )
        return B / D  # Do NOT clip; preserve semantics

    def _binary_params(self) -> tuple[float, float, float]:
        """Extract binary utility parameters A, B, D."""
        if isinstance(self.utility, UtilitySpec):
            tp, tn, fp, fn = (
                self.utility.tp_utility,
                self.utility.tn_utility,
                self.utility.fp_utility,
                self.utility.fn_utility,
            )
        else:
            if self.utility.shape != (2, 2):
                raise ValueError("Binary decisions require a 2x2 utility matrix")
            tn, fn = float(self.utility[0, 0]), float(self.utility[0, 1])
            fp, tp = float(self.utility[1, 0]), float(self.utility[1, 1])

        A = tp - fn  # Benefit of TP over FN
        B = tn - fp  # Benefit of TN over FP
        D = A + B  # Total utility difference
        return A, B, D

    def _extract_binary_p(self, probs: np.ndarray) -> np.ndarray:
        """Extract P(y=1) from binary probabilities."""
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim == 1:
            return probs  # already P(y=1)
        if probs.ndim == 2 and probs.shape[1] == 2:
            return probs[:, 1]  # assume column 1 is P(y=1)
        raise ValueError("Binary probabilities must be shape (n,) or (n,2)")

    def _decide_binary(self, probs: np.ndarray) -> np.ndarray:
        """Make binary decisions using margin approach (handles all D cases)."""
        p = self._extract_binary_p(probs)
        _, B, D = self._binary_params()

        if abs(D) < 1e-12:  # D ≈ 0: decision is probability-independent
            # Predict positive if B <= 0, negative if B > 0
            return (
                np.ones_like(p, dtype=np.int32)
                if B <= 0
                else np.zeros_like(p, dtype=np.int32)
            )

        # The margin formula D*p - B >= 0 is universally correct
        # It naturally handles D > 0, D < 0, and gives consistent >= tie-breaking
        margin = D * p - B
        return (margin >= 0.0).astype(np.int32)

    def compute_thresholds(self, n_classes: int) -> NDArray[np.float64]:
        """Compute per-class thresholds for OvR multiclass.

        Args:
            n_classes: Number of classes

        Returns:
            Per-class thresholds

        Raises:
            NotImplementedError: If the utility is a matrix; per-class thresholds
                            need a UtilitySpec.
        """
        if isinstance(self.utility, UtilitySpec):
            # Use same utility for all classes
            threshold = self.compute_threshold()
            return np.full(n_classes, threshold)
        # Need per-class utilities
        raise NotImplementedError(
            "Per-class utilities from matrix not yet implemented. "
            "Use UtilitySpec for OvR thresholds."
        )

    def decide(self, probabilities: NDArray[np.float64]) -> NDArray[np.int32]:
        """Make Bayes-optimal decisions.

        Args:
            probabilities: Probability array. For binary: shape (n,) or (n,2).
                For multiclass: shape (n, n_classes).

        Returns:
            Optimal decisions

        Raises:
            ValueError: If the decision rule needs a utility matrix that was not
                            supplied, or the probabilities are not the expected shape.
        """
        probs = np.asarray(probabilities, dtype=np.float64)

        if self.decision_rule == DecisionRule.THRESHOLD:
            # Use margin-based binary decision
            return self._decide_binary(probs)

        if self.decision_rule == DecisionRule.ARGMAX:
            if not isinstance(self.utility, np.ndarray):
                raise ValueError("ARGMAX rule requires utility matrix")

            # Expected utilities: E[U|x] = Σ_y U(d,y) P(y|x)
            if probs.ndim != 2:
                raise ValueError("ARGMAX rule requires 2D probability matrix")
            expected = probs @ self.utility.T
            return np.argmax(expected, axis=1).astype(np.int32)

        # MARGIN - implement properly or remove
        if self.is_binary:
            return self._decide_binary(probs)
        # For multiclass margin: would need per-class thresholds
        if probs.ndim != 2:
            raise ValueError("Multiclass margin requires 2D probability matrix")
        n_classes = probs.shape[1]
        thresholds = self.compute_thresholds(n_classes)
        # Argmax of margin: p - threshold
        margins = probs - thresholds[None, :]
        return np.argmax(margins, axis=1).astype(np.int32)

    def expected_utility(self, probabilities: NDArray[np.float64]) -> float:
        """Compute expected utility under optimal decisions.

        Args:
            probabilities: Probability array

        Returns:
            Expected utility per sample

        Raises:
            ValueError: If the ARGMAX rule is used without a utility matrix.
        """
        probs = np.asarray(probabilities, dtype=np.float64)

        if self.decision_rule == DecisionRule.ARGMAX:
            if not isinstance(self.utility, np.ndarray):
                raise ValueError("ARGMAX rule requires utility matrix")
            expected = probs @ self.utility.T  # (n, k) @ (k, d) -> (n, d)
            chosen = np.max(expected, axis=1)
            return float(np.mean(chosen))

        # Binary (UtilitySpec or 2x2 matrix)
        p = self._extract_binary_p(probs)
        if isinstance(self.utility, UtilitySpec):
            tp, tn, fp, fn = (
                self.utility.tp_utility,
                self.utility.tn_utility,
                self.utility.fp_utility,
                self.utility.fn_utility,
            )
        else:
            tn, fn = float(self.utility[0, 0]), float(self.utility[0, 1])
            fp, tp = float(self.utility[1, 0]), float(self.utility[1, 1])

        eu_pos = p * tp + (1 - p) * fp
        eu_neg = p * fn + (1 - p) * tn
        return float(np.mean(np.maximum(eu_pos, eu_neg)))


# ============================================================================
# Factory Functions
# ============================================================================


def bayes_optimal_threshold(
    fp_cost: float,
    fn_cost: float,
    tp_benefit: float = 0.0,
    tn_benefit: float = 0.0,
) -> OptimizationResult:
    """Compute optimal threshold from costs and benefits.

    Args:
        fp_cost: Cost of false positive (positive value)
        fn_cost: Cost of false negative (positive value)
        tp_benefit: Benefit of true positive. Defaults to 0.0.
        tn_benefit: Benefit of true negative. Defaults to 0.0.

    Returns:
        Optimization result with threshold and predict function
    """
    # Convert costs to utilities (negate costs)
    utility = UtilitySpec(
        tp_utility=tp_benefit,
        tn_utility=tn_benefit,
        fp_utility=-abs(fp_cost),
        fn_utility=-abs(fn_cost),
    )

    optimizer = BayesOptimal(utility)
    threshold = optimizer.compute_threshold()

    # Compute expected utility as "score"
    # For now, use a placeholder since we need probabilities to compute expected utility
    expected_utility = 0.0

    from .validation import make_binary_predictor

    return OptimizationResult(
        thresholds=np.array([threshold]),
        scores=np.array([expected_utility]),
        predict=make_binary_predictor(threshold, ">="),
        task=Task.BINARY,
        metric="expected_utility",
        n_classes=2,
    )


def bayes_optimal_decisions(
    probabilities: NDArray[np.float64] | None = None,
    utility_matrix: NDArray[np.float64] | None = None,
    cost_matrix: NDArray[np.float64] | None = None,
) -> OptimizationResult:
    """Compute optimal decisions from utility or cost matrix.

    For general cost matrices, per-class thresholds are NOT optimal!
    This function implements the exact Bayes rule: argmin_j Σ_i p_i * C(i,j)

    This is the theoretically correct approach for arbitrary cost structures
    where costs depend on both true class i and predicted class j.

    Args:
        probabilities: Class probabilities (must be calibrated). Shape: (n_samples,
            n_classes).
        utility_matrix: Utility matrix U[i,j] = utility(true class=i, decision=j).
            Higher values = better outcomes. Shape: (n_classes, n_decisions). Optional.
        cost_matrix: Cost matrix C[i,j] = cost of taking decision j when the true class
            is i.
            Lower values = better outcomes. Shape: (n_classes, n_decisions). Optional.

    Returns:
        Optimization result with decision strategy and predict function

    Raises:
        ValueError: If neither or both of utility_matrix and cost_matrix are
                    given, or their shapes disagree with the probabilities.

    Examples:
        >>> # Asymmetric costs: misclassifying car as dog is expensive
        >>> cost_matrix = np.array([
        ...     [0,  10,  50],   # True dog: [predict dog, cat, car]
        ...     [10,  0,  40],   # True cat
        ...     [100, 90,  0],   # True car
        ... ])
        >>> result = bayes_optimal_decisions(y_prob, cost_matrix=cost_matrix)
        >>> y_pred = result.predict(y_prob)  # Uses direct Bayes rule

    Notes:
        **When to use this vs thresholds:**

        - Use this function when costs have GENERAL structure (e.g., car→dog costs
          more than car→cat)
        - Use threshold-based methods when costs have OvR structure (each class
          has independent FP/FN costs)

        **Complexity:** O(K²) per sample vs O(K) for thresholds

        **Why thresholds don't work:** The Bayes rule depends on the full
        probability vector p, not individual components p_j. Margin rules like
        argmax(p_j - τ_j) cannot capture cost correlations between classes.
    """
    # Validate input: exactly one of utility_matrix or cost_matrix
    if utility_matrix is None and cost_matrix is None:
        raise ValueError("Must provide either utility_matrix or cost_matrix")
    if utility_matrix is not None and cost_matrix is not None:
        raise ValueError("Provide either utility_matrix or cost_matrix, not both")

    # Convert utility/cost matrix to array
    if utility_matrix is not None:
        utility = np.asarray(utility_matrix, dtype=np.float64)
    else:
        utility = -np.asarray(
            cost_matrix, dtype=np.float64
        )  # Convert costs to utilities

    # Validate utility matrix shape
    if utility.ndim != 2:
        raise ValueError("utility_matrix must be 2D array")

    n_classes = utility.shape[0]
    n_decisions = utility.shape[1]

    # Handle case when probabilities are provided vs not
    if probabilities is not None:
        probs = np.asarray(probabilities, dtype=np.float64)

        if probs.ndim != 2:
            raise ValueError("probabilities must be 2D array")
        if probs.shape[1] != n_classes:
            raise ValueError(
                f"probabilities has {probs.shape[1]} classes but "
                f"utility_matrix has {n_classes}"
            )

        # Compute expected utilities/costs: E[U|x, j] = Σ_i P(i|x) U(i, j)
        expected = probs @ utility  # (n_samples, n_classes) @ (n_classes, n_decisions)
        # -> (n_samples, n_decisions)

        # Always maximize utility (whether provided directly or converted from costs)
        optimal_values = np.max(expected, axis=1)
        mean_value = float(np.mean(optimal_values))
    else:
        # No probabilities provided - create policy-only result
        mean_value = 0.0

    # Create prediction function (closure captures utility matrix)
    def predict_bayes_decisions(probabilities_new):
        p = np.asarray(probabilities_new, dtype=np.float64)
        if p.ndim != 2:
            raise ValueError("probabilities must be 2D array")
        if p.shape[1] != n_classes:
            raise ValueError(f"Expected {n_classes} classes, got {p.shape[1]}")

        # Compute expected utilities and return optimal decisions (always maximize
        # utility)
        expected_new = p @ utility
        return np.argmax(expected_new, axis=1).astype(np.int32)

    # For utility-based decisions, we don't have traditional "thresholds",
    # but we can use zeros as placeholders since the predict function handles everything
    thresholds = np.zeros(n_decisions, dtype=np.float64)
    scores = np.full(n_decisions, mean_value, dtype=np.float64)

    metric_name = "expected_cost" if cost_matrix is not None else "expected_utility"

    return OptimizationResult(
        thresholds=thresholds,
        scores=scores,
        predict=predict_bayes_decisions,
        task=Task.MULTICLASS,
        metric=metric_name,
        n_classes=n_decisions,
    )


def bayes_thresholds_from_costs(
    fp_costs: NDArray[np.float64] | list[float],
    fn_costs: NDArray[np.float64] | list[float],
    *,
    use_weighted_margin: bool = True,
) -> OptimizationResult:
    """Compute per-class Bayes thresholds from OvR costs (vectorized).

    For OvR cost structure where each class j has:
    - False positive cost: c_j
    - False negative cost: r_j

    The optimal thresholds are: τ_j = c_j / (c_j + r_j)

    IMPORTANT: When cost totals (c_j + r_j) differ across classes, the correct
    Bayes decision rule is the WEIGHTED margin:
    ŷ = argmax_j [(c_j + r_j) * (p_j - τ_j)]

    The unweighted margin p_j - τ_j is only optimal when all cost totals are equal.

    Args:
        fp_costs: False positive costs per class (positive values)
        fn_costs: False negative costs per class (positive values)
        use_weighted_margin: If True, prediction function uses correct weighted margin
            rule.
            If False, uses unweighted margin (only optimal when cost totals are equal).
            Defaults to True.

    Returns:
        Optimization result with per-class thresholds and Bayes-optimal predict function

    Raises:
        ValueError: If the cost arrays disagree in shape, are non-finite, sum to
                    zero, or disagree with the probabilities' class count.

    Notes:
        This implements the exact Bayes rule for OvR cost structure:
        C(i,j) = c_j if i ≠ j, else 0 (cost for predicting j when true class is i ≠ j)

    Examples:
        >>> # Different cost structures
        >>> fp_costs = [1, 5, 2]  # Class 1 has high FP cost
        >>> fn_costs = [1, 1, 8]  # Class 2 has high FN cost
        >>> result = bayes_thresholds_from_costs(fp_costs, fn_costs)
        >>> result.thresholds  # [0.5, 0.833, 0.2]
        >>>
        >>> # Cost totals: [2, 6, 10] - differ significantly!
        >>> # Must use weighted margin for Bayes optimality
        >>> pred = result.predict(y_prob)  # Uses weighted margin automatically
    """
    fp = np.asarray(fp_costs, dtype=np.float64)
    fn = np.asarray(fn_costs, dtype=np.float64)
    if fp.shape != fn.shape:
        raise ValueError("fp_costs and fn_costs must have same shape")

    # Validate inputs are finite
    if not np.all(np.isfinite(fp)) or not np.all(np.isfinite(fn)):
        raise ValueError("All costs must be finite")

    # Take absolute values to handle negative costs (utilities)
    fp_abs = np.abs(fp)
    fn_abs = np.abs(fn)

    den = fp_abs + fn_abs
    if np.any(den <= 0):
        raise ValueError("All |fp_cost| + |fn_cost| must be > 0")

    thresholds = fp_abs / den
    n_classes = len(thresholds)

    # Compute expected cost as "score" (negative values for costs)
    expected_costs = -(fp_abs + fn_abs)  # Negative for minimization

    # Create prediction function based on margin rule choice
    cost_totals = fp_abs + fn_abs  # (c_j + r_j) for each class

    def predict_multiclass_bayes(probs):
        p = np.asarray(probs)
        if p.ndim != 2:
            raise ValueError("Multiclass requires 2D probabilities")
        if p.shape[1] != n_classes:
            raise ValueError(f"Expected {n_classes} classes, got {p.shape[1]}")

        if use_weighted_margin:
            # Correct Bayes rule: argmax_j [(c_j + r_j) * (p_j - τ_j)]
            # Equivalently: argmax_j [(c_j + r_j) * p_j - c_j]
            weighted_scores = cost_totals[None, :] * p - fp_abs[None, :]
            predictions = np.argmax(weighted_scores, axis=1).astype(np.int32)
        else:
            # Unweighted margin (only optimal when all cost totals are equal)
            margins = p - thresholds[None, :]
            # Fall back to argmax when all margins negative
            valid = margins > 0
            masked_margins = np.where(valid, margins, -np.inf)
            predictions = np.argmax(masked_margins, axis=1).astype(np.int32)

            # For samples where no class has positive margin, use argmax
            no_valid = ~np.any(valid, axis=1)
            if np.any(no_valid):
                predictions[no_valid] = np.argmax(p[no_valid], axis=1)

        return predictions

    return OptimizationResult(
        thresholds=thresholds,
        scores=expected_costs,
        predict=predict_multiclass_bayes,
        task=Task.MULTICLASS,
        metric="expected_cost",
        n_classes=n_classes,
    )
