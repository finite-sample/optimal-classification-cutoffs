"""Reusable Hypothesis strategies for comprehensive threshold optimization testing.

This module provides controlled data generation strategies that enable systematic
testing of mathematical properties and edge cases in the optimization algorithms.
"""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


def probs_1d(
    min_size: int = 3,
    max_size: int = 300,
    allow_extremes: bool = True,
    tie_blocks: bool = True,
):
    """Generate 1D probabilities in [0,1] with controllable ties and extremes.

    Parameters
    ----------
    min_size : int
        Minimum array size
    max_size : int
        Maximum array size
    allow_extremes : bool
        Whether to force some values to be exactly 0.0 and 1.0
    tie_blocks : bool
        Whether to create blocks of tied values

    Returns
    -------
    hypothesis.strategies.SearchStrategy
        Strategy that generates probability arrays with controlled structure
    """
    base = arrays(
        dtype=float,
        shape=st.integers(min_value=min_size, max_value=max_size),
        elements=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )

    def _post_process(arr):
        n = arr.shape[0]
        rng = np.random.default_rng(123)

        if tie_blocks and n >= 4:
            # Create 2-3 random tie blocks
            n_blocks = min(3, n // 2)
            idx = rng.choice(n, size=n_blocks, replace=False)
            val = float(rng.uniform(0, 1))
            arr[idx] = val

        if allow_extremes and n >= 2:
            # Force some extreme values
            arr[0] = 0.0
            arr[-1] = 1.0

        return arr

    return base.map(_post_process)


def labels_binary_like(p_strategy=None):
    """Generate binary labels roughly aligned with probabilities.

    Ensures both classes are present to avoid degenerate cases in most tests.

    Parameters
    ----------
    p_strategy : hypothesis.strategies.SearchStrategy, optional
        Strategy for generating probabilities. If None, uses probs_1d()

    Returns
    -------
    hypothesis.strategies.SearchStrategy
        Strategy that generates (labels, probabilities) pairs
    """
    if p_strategy is None:
        p_strategy = probs_1d()

    @st.composite
    def _make_labels_and_probs(draw):
        probs = draw(p_strategy)

        # Generate labels roughly aligned with probabilities
        rng = np.random.default_rng(321)
        labels = (
            rng.uniform(0, 1, size=probs.shape[0]) < np.clip(probs, 0.05, 0.95)
        ).astype(int)

        # Ensure both classes present
        if labels.sum() == 0:
            labels[0] = 1
        if labels.sum() == labels.size:
            labels[0] = 0

        return labels, probs

    return _make_labels_and_probs()


def rational_weights(
    min_size: int = 3, max_size: int = 200, denominators: tuple = (2, 3, 4, 5)
):
    """Generate sample weights that are rational with small denominators.

    This enables exact expansion testing where each sample can be duplicated
    an integer number of times.

    Parameters
    ----------
    min_size : int
        Minimum array size
    max_size : int
        Maximum array size
    denominators : tuple
        Allowed denominators for rational weights

    Returns
    -------
    hypothesis.strategies.SearchStrategy
        Strategy that generates rational weight arrays
    """
    n = st.integers(min_value=min_size, max_value=max_size)
    den = st.sampled_from(denominators)
    return st.tuples(n, den).map(lambda t: _rational_weights(*t))


def _rational_weights(n: int, den: int) -> np.ndarray:
    """Generate rational weights with given size and denominator."""
    rng = np.random.default_rng(777)
    numerators = rng.integers(1, 5, size=n)  # numerators 1..4
    weights = numerators / den
    return weights.astype(float)


def multiclass_probs(n_classes: int, min_size: int = 10, max_size: int = 100):
    """Generate multiclass probability matrices that sum to 1.

    Parameters
    ----------
    n_classes : int
        Number of classes
    min_size : int
        Minimum number of samples
    max_size : int
        Maximum number of samples

    Returns
    -------
    hypothesis.strategies.SearchStrategy
        Strategy that generates normalized probability matrices
    """

    @st.composite
    def _make_multiclass_probs(draw):
        size = draw(st.integers(min_size, max_size))

        # Generate raw probabilities
        probs = draw(
            arrays(
                dtype=float,
                shape=(size, n_classes),
                elements=st.floats(0.01, 0.99, allow_nan=False, allow_infinity=False),
            )
        )

        # Normalize to sum to 1
        return probs / probs.sum(axis=1, keepdims=True)

    return _make_multiclass_probs()


def multiclass_labels_and_probs(
    n_classes: int, min_size: int = 10, max_size: int = 100
):
    """Generate multiclass labels and corresponding probability matrices.

    Parameters
    ----------
    n_classes : int
        Number of classes
    min_size : int
        Minimum number of samples
    max_size : int
        Maximum number of samples

    Returns
    -------
    hypothesis.strategies.SearchStrategy
        Strategy that generates (labels, probabilities) pairs
    """

    @st.composite
    def _make_multiclass_data(draw):
        size = draw(st.integers(min_size, max_size))

        # Generate labels with all classes represented
        rng = np.random.default_rng(456)
        labels = rng.choice(n_classes, size=size)

        # Ensure all classes have at least one sample
        for c in range(n_classes):
            if not np.any(labels == c):
                labels[rng.integers(0, size)] = c

        # Generate probabilities
        probs = draw(multiclass_probs(n_classes, min_size=size, max_size=size))

        return labels, probs

    return _make_multiclass_data()


def beta_bernoulli_calibrated(min_size: int = 50, max_size: int = 300):
    """Generate calibrated binary data using Beta-Bernoulli process.

    This creates data where P(y=1|p) = p, making it ideal for testing
    expected F-beta optimization methods like Dinkelbach.

    Parameters
    ----------
    min_size : int
        Minimum number of samples
    max_size : int
        Maximum number of samples

    Returns
    -------
    hypothesis.strategies.SearchStrategy
        Strategy that generates calibrated (labels, probabilities, alpha, beta) tuples
    """

    @st.composite
    def _make_calibrated_data(draw):
        size = draw(st.integers(min_size, max_size))
        alpha = draw(st.floats(0.5, 5.0))
        beta = draw(st.floats(0.5, 5.0))

        # Generate probabilities from Beta distribution
        rng = np.random.default_rng(888)
        probs = rng.beta(alpha, beta, size=size)

        # Generate labels from Bernoulli with these probabilities (perfectly calibrated)
        labels = rng.binomial(1, probs).astype(np.int8)

        return labels, probs, alpha, beta

    return _make_calibrated_data()


def tied_probabilities(
    base_prob: float = 0.5,
    tie_fraction: float = 0.3,
    min_size: int = 10,
    max_size: int = 50,
):
    """Generate probability arrays with many tied values.

    This is useful for testing tie-handling behavior in different comparison operators.

    Parameters
    ----------
    base_prob : float
        The probability value that will be tied
    tie_fraction : float
        Fraction of samples that should have the tied value
    min_size : int
        Minimum array size
    max_size : int
        Maximum array size

    Returns
    -------
    hypothesis.strategies.SearchStrategy
        Strategy that generates probability arrays with controlled ties
    """

    @st.composite
    def _make_tied_probs(draw):
        size = draw(st.integers(min_size, max_size))

        # Generate base probabilities
        probs = draw(
            arrays(
                dtype=float,
                shape=size,
                elements=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
            )
        )

        # Force ties at base_prob
        n_ties = max(2, int(size * tie_fraction))
        rng = np.random.default_rng(999)
        tie_indices = rng.choice(size, n_ties, replace=False)
        probs[tie_indices] = base_prob

        return probs

    return _make_tied_probs()


def extreme_probabilities(min_size: int = 5, max_size: int = 30):
    """Generate probability arrays with extreme values (0, 1, very close to boundaries).

    Parameters
    ----------
    min_size : int
        Minimum array size
    max_size : int
        Maximum array size

    Returns
    -------
    hypothesis.strategies.SearchStrategy
        Strategy that generates arrays with extreme probability values
    """

    @st.composite
    def _make_extreme_probs(draw):
        size = draw(st.integers(min_size, max_size))

        probs = np.zeros(size, dtype=float)
        rng = np.random.default_rng(111)

        for i in range(size):
            choice = rng.choice(["zero", "one", "near_zero", "near_one", "normal"])
            if choice == "zero":
                probs[i] = 0.0
            elif choice == "one":
                probs[i] = 1.0
            elif choice == "near_zero":
                probs[i] = rng.uniform(1e-10, 1e-5)
            elif choice == "near_one":
                probs[i] = rng.uniform(1.0 - 1e-5, 1.0 - 1e-10)
            else:  # normal
                probs[i] = rng.uniform(0.1, 0.9)

        return probs

    return _make_extreme_probs()


def permutation_invariant_multiclass(
    n_classes: int, min_size: int = 20, max_size: int = 60
):
    """Generate multiclass data with label permutations for invariance testing.

    Returns the original data and a permuted version for testing that
    algorithms are invariant to label permutations.

    Parameters
    ----------
    n_classes : int
        Number of classes
    min_size : int
        Minimum number of samples
    max_size : int
        Maximum number of samples

    Returns
    -------
    hypothesis.strategies.SearchStrategy
        Strategy that generates ((labels, probs), (labels_perm, probs_perm), perm)
        tuples
    """

    @st.composite
    def _make_permutation_data(draw):
        labels, probs = draw(multiclass_labels_and_probs(n_classes, min_size, max_size))

        # Generate permutation
        rng = np.random.default_rng(222)
        perm = rng.permutation(n_classes)

        # Apply permutation to labels and probabilities
        labels_perm = perm[labels]
        probs_perm = probs[:, perm]

        return (labels, probs), (labels_perm, probs_perm), perm

    return _make_permutation_data()
