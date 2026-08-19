"""Common test data generation utilities.

This module provides standardized data generation functions for consistent
testing across all test modules.
"""

import numpy as np


def generate_binary_data(
    n_samples: int = 100,
    imbalance_ratio: float = 0.5,
    noise: float = 0.1,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate binary classification test data.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    imbalance_ratio : float
        Fraction of positive samples (0 < ratio < 1)
    noise : float
        Amount of noise to add to probability generation
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    tuple of (labels, probabilities)
        Binary labels and corresponding probabilities
    """
    rng = np.random.default_rng(random_state)

    # Generate base probabilities with some structure
    base_probs = rng.beta(2, 2, n_samples)  # Bell-shaped distribution

    # Generate labels based on probabilities with noise
    labels = (
        base_probs + noise * rng.normal(0, 1, n_samples) > (1 - imbalance_ratio)
    ).astype(int)

    # Ensure both classes are present
    if labels.sum() == 0:
        labels[0] = 1
    elif labels.sum() == n_samples:
        labels[0] = 0

    return labels, base_probs


def generate_multiclass_data(
    n_samples: int = 100, n_classes: int = 3, random_state: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Generate multiclass classification test data.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    n_classes : int
        Number of classes
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    tuple of (labels, probabilities)
        Multiclass labels and probability matrix
    """
    rng = np.random.default_rng(random_state)

    # Generate labels ensuring all classes are present
    labels = rng.choice(n_classes, size=n_samples)
    for c in range(n_classes):
        if not np.any(labels == c):
            labels[rng.integers(0, n_samples)] = c

    # Generate probability matrix
    probs = rng.dirichlet(np.ones(n_classes), size=n_samples)

    return labels, probs


def generate_calibrated_probabilities(
    n_samples: int = 200, random_state: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Generate perfectly calibrated binary data.

    Uses Beta-Bernoulli process where P(y=1|p) = p exactly.
    Ideal for testing expected F-beta optimization methods.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    tuple of (labels, probabilities)
        Calibrated binary labels and probabilities
    """
    rng = np.random.default_rng(random_state)

    # Generate probabilities from Beta distribution
    probs = rng.beta(2, 2, n_samples)

    # Generate labels from Bernoulli with these probabilities
    labels = rng.binomial(1, probs)

    return labels, probs


def generate_tied_probabilities(
    n_samples: int = 50,
    base_prob: float = 0.5,
    tie_fraction: float = 0.3,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate data with many tied probability values.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    base_prob : float
        The probability value that will be tied
    tie_fraction : float
        Fraction of samples that should have the tied value
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    tuple of (labels, probabilities)
        Labels and probabilities with controlled ties
    """
    rng = np.random.default_rng(random_state)

    # Generate base probabilities
    probs = rng.uniform(0, 1, n_samples)

    # Force ties at base_prob
    n_ties = max(2, int(n_samples * tie_fraction))
    tie_indices = rng.choice(n_samples, n_ties, replace=False)
    probs[tie_indices] = base_prob

    # Generate labels roughly correlated with probabilities
    labels = (probs + 0.1 * rng.normal(0, 1, n_samples) > 0.5).astype(int)

    return labels, probs


def generate_extreme_probabilities(
    n_samples: int = 20, random_state: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Generate data with extreme probability values.

    Includes values at 0, 1, and very close to boundaries.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    tuple of (labels, probabilities)
        Labels and extreme probability values
    """
    rng = np.random.default_rng(random_state)

    probs = np.zeros(n_samples, dtype=float)

    for i in range(n_samples):
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

    # Generate labels with some correlation to probabilities
    labels = (probs > 0.5).astype(int)

    return labels, probs


def generate_sample_weights(
    n_samples: int, weight_type: str = "uniform", random_state: int | None = None
) -> np.ndarray:
    """Generate sample weights for testing.

    Parameters
    ----------
    n_samples : int
        Number of samples
    weight_type : str
        Type of weights: 'uniform', 'random', 'integer', 'extreme'
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Sample weights
    """
    rng = np.random.default_rng(random_state)

    if weight_type == "uniform":
        return np.ones(n_samples, dtype=float)
    if weight_type == "random":
        return rng.uniform(0.1, 3.0, n_samples)
    if weight_type == "integer":
        return rng.integers(1, 5, n_samples).astype(float)
    if weight_type == "extreme":
        weights = rng.uniform(0.01, 10.0, n_samples)
        # Add some very small and very large weights
        weights[0] = 1e-6
        weights[-1] = 1e6
        return weights
    raise ValueError(f"Unknown weight_type: {weight_type}")


def generate_imbalanced_data(
    n_samples: int = 1000,
    imbalance_ratio: float = 0.01,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate highly imbalanced binary data.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    imbalance_ratio : float
        Fraction of positive samples (very small for imbalanced data)
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    tuple of (labels, probabilities)
        Highly imbalanced binary data
    """
    rng = np.random.default_rng(random_state)

    n_positive = max(1, int(n_samples * imbalance_ratio))
    n_negative = n_samples - n_positive

    # Create labels
    labels = np.concatenate([np.zeros(n_negative), np.ones(n_positive)])

    # Generate probabilities slightly favoring the true class
    neg_probs = rng.uniform(0.0, 0.4, n_negative)
    pos_probs = rng.uniform(0.6, 1.0, n_positive)
    probs = np.concatenate([neg_probs, pos_probs])

    # Shuffle to randomize order
    shuffle_idx = rng.permutation(n_samples)
    labels = labels[shuffle_idx]
    probs = probs[shuffle_idx]

    return labels.astype(int), probs
