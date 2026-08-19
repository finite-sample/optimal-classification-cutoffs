"""Test fixtures with realistic datasets for optimal cutoff testing.

This module provides standardized, realistic datasets for testing threshold optimization
algorithms. All datasets use meaningful sample sizes (100+ samples) and realistic
distributions that mirror real-world classification problems.
"""

from typing import NamedTuple

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class BinaryDataset(NamedTuple):
    """Binary classification dataset."""

    y_true: np.ndarray
    y_prob: np.ndarray
    description: str


class MulticlassDataset(NamedTuple):
    """Multiclass classification dataset."""

    y_true: np.ndarray
    y_prob: np.ndarray
    n_classes: int
    description: str


def make_realistic_binary_dataset(
    n_samples: int = 500,
    class_balance: float = 0.3,
    noise_level: float = 0.1,
    random_state: int = 42,
) -> BinaryDataset:
    """Create a realistic binary classification dataset.

    Args:
        n_samples: Number of samples
        class_balance: Fraction of positive class (0 < balance < 1)
        noise_level: Amount of noise to add to probabilities (0 to 0.5)
        random_state: Random seed for reproducibility

    Returns:
        BinaryDataset with realistic probabilities and labels
    """
    rng = np.random.default_rng(random_state)

    # Generate features and labels using sklearn
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,
        weights=[1 - class_balance, class_balance],
        random_state=random_state,
    )

    # Train a logistic regression to get realistic probabilities
    model = LogisticRegression(random_state=random_state)
    model.fit(X, y)
    y_prob = model.predict_proba(X)[:, 1]

    # Add controlled noise to probabilities
    if noise_level > 0:
        noise = rng.normal(0, noise_level, size=y_prob.shape)
        y_prob = np.clip(y_prob + noise, 0.01, 0.99)

    description = (
        f"Binary dataset: {n_samples} samples, {class_balance:.1%} positive, "
        f"noise={noise_level}"
    )

    return BinaryDataset(y, y_prob, description)


def make_imbalanced_binary_dataset(
    n_samples: int = 1000, positive_fraction: float = 0.05, random_state: int = 42
) -> BinaryDataset:
    """Create a highly imbalanced binary dataset.

    Args:
        n_samples: Number of samples
        positive_fraction: Fraction of positive class (should be small)
        random_state: Random seed

    Returns:
        BinaryDataset with class imbalance
    """
    # Generate imbalanced data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=15,
        n_informative=10,
        weights=[1 - positive_fraction, positive_fraction],
        flip_y=0.01,  # Small amount of label noise
        random_state=random_state,
    )

    # Use Random Forest for more realistic probabilities in imbalanced case
    model = RandomForestClassifier(n_estimators=50, random_state=random_state)
    model.fit(X, y)
    y_prob = model.predict_proba(X)[:, 1]

    description = (
        f"Imbalanced dataset: {n_samples} samples, {positive_fraction:.1%} positive"
    )

    return BinaryDataset(y, y_prob, description)


def make_well_separated_binary_dataset(
    n_samples: int = 300, random_state: int = 42
) -> BinaryDataset:
    """Create a well-separated binary dataset (easy classification).

    Args:
        n_samples: Number of samples
        random_state: Random seed

    Returns:
        BinaryDataset with good class separation
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=1,
        class_sep=2.0,  # Good separation
        random_state=random_state,
    )

    model = LogisticRegression(random_state=random_state)
    model.fit(X, y)
    y_prob = model.predict_proba(X)[:, 1]

    description = f"Well-separated dataset: {n_samples} samples, good class separation"

    return BinaryDataset(y, y_prob, description)


def make_overlapping_binary_dataset(
    n_samples: int = 400, random_state: int = 42
) -> BinaryDataset:
    """Create an overlapping binary dataset (difficult classification).

    Args:
        n_samples: Number of samples
        random_state: Random seed

    Returns:
        BinaryDataset with significant class overlap
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=12,
        n_informative=6,
        n_redundant=6,
        n_clusters_per_class=2,
        class_sep=0.5,  # Poor separation
        flip_y=0.05,  # Some label noise
        random_state=random_state,
    )

    model = LogisticRegression(random_state=random_state)
    model.fit(X, y)
    y_prob = model.predict_proba(X)[:, 1]

    description = f"Overlapping dataset: {n_samples} samples, significant class overlap"

    return BinaryDataset(y, y_prob, description)


def make_realistic_multiclass_dataset(
    n_samples: int = 600,
    n_classes: int = 3,
    class_imbalance: bool = False,
    random_state: int = 42,
) -> MulticlassDataset:
    """Create a realistic multiclass classification dataset.

    Args:
        n_samples: Number of samples
        n_classes: Number of classes (2-5)
        class_imbalance: Whether to create imbalanced classes
        random_state: Random seed

    Returns:
        MulticlassDataset with realistic probabilities
    """
    if class_imbalance:
        # Create imbalanced weights
        weights = [0.6, 0.3, 0.1][:n_classes]
        if len(weights) < n_classes:
            remaining = 1.0 - sum(weights)
            weights.extend(
                [remaining / (n_classes - len(weights))] * (n_classes - len(weights))
            )
    else:
        weights = None

    X, y = make_classification(
        n_samples=n_samples,
        n_features=15,
        n_informative=10,
        n_redundant=5,
        n_classes=n_classes,
        n_clusters_per_class=1,
        weights=weights,
        random_state=random_state,
    )

    # Use Random Forest for realistic multiclass probabilities
    model = RandomForestClassifier(n_estimators=30, random_state=random_state)
    model.fit(X, y)
    y_prob = model.predict_proba(X)

    balance_str = "imbalanced" if class_imbalance else "balanced"
    description = (
        f"Multiclass dataset: {n_samples} samples, {n_classes} classes, {balance_str}"
    )

    return MulticlassDataset(y, y_prob, n_classes, description)


def make_calibrated_binary_dataset(
    n_samples: int = 800, random_state: int = 42
) -> BinaryDataset:
    """Create a well-calibrated binary dataset.

    This dataset is designed so that predicted probabilities closely match
    the true probability of positive class.

    Args:
        n_samples: Number of samples
        random_state: Random seed

    Returns:
        BinaryDataset with well-calibrated probabilities
    """
    rng = np.random.default_rng(random_state)

    # Generate probabilities first, then labels based on those probabilities
    y_prob = rng.beta(2, 2, size=n_samples)  # Beta distribution gives good spread
    y_true = rng.binomial(1, y_prob)  # Labels match probabilities

    description = (
        f"Calibrated dataset: {n_samples} samples, well-calibrated probabilities"
    )

    return BinaryDataset(y_true, y_prob, description)


def make_large_binary_dataset(
    n_samples: int = 5000, random_state: int = 42
) -> BinaryDataset:
    """Create a large binary dataset for performance testing.

    Args:
        n_samples: Number of samples (should be large, e.g., 5000+)
        random_state: Random seed

    Returns:
        Large BinaryDataset for performance testing
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=25,
        n_informative=20,
        n_redundant=5,
        n_clusters_per_class=2,
        weights=[0.4, 0.6],
        random_state=random_state,
    )

    model = LogisticRegression(random_state=random_state)
    model.fit(X, y)
    y_prob = model.predict_proba(X)[:, 1]

    description = f"Large dataset: {n_samples} samples for performance testing"

    return BinaryDataset(y, y_prob, description)


# Standard fixtures for common use
STANDARD_BINARY = make_realistic_binary_dataset(500, 0.3, 0.1, 42)
IMBALANCED_BINARY = make_imbalanced_binary_dataset(1000, 0.05, 42)
WELL_SEPARATED_BINARY = make_well_separated_binary_dataset(300, 42)
OVERLAPPING_BINARY = make_overlapping_binary_dataset(400, 42)
CALIBRATED_BINARY = make_calibrated_binary_dataset(800, 42)
LARGE_BINARY = make_large_binary_dataset(5000, 42)

STANDARD_MULTICLASS = make_realistic_multiclass_dataset(600, 3, False, 42)
IMBALANCED_MULTICLASS = make_realistic_multiclass_dataset(800, 4, True, 42)
