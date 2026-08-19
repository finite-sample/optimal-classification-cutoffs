"""Test fixtures and utilities for optimal cutoff testing.

This module provides standardized data generation, realistic datasets,
assertion helpers, and hypothesis testing strategies for consistent
testing across all test modules.
"""

# Export all data generators (simple numpy-based)
# Export assertion helpers
from .assertions import (
    assert_arrays_close,
    assert_labels_valid,
    assert_method_consistency,
    assert_monotonic_increase,
    assert_optimization_successful,
    assert_probability_matrix_valid,
    assert_valid_confusion_matrix,
    assert_valid_metric_score,
    assert_valid_threshold,
)
from .data_generators import (
    generate_binary_data,
    generate_calibrated_probabilities,
    generate_multiclass_data,
    generate_tied_probabilities,
)

# Export hypothesis strategies
from .hypothesis_strategies import (
    beta_bernoulli_calibrated,
    extreme_probabilities,
    labels_binary_like,
    multiclass_labels_and_probs,
    rational_weights,
    tied_probabilities,
)

# Export all realistic datasets (sklearn-based)
from .realistic_datasets import (
    CALIBRATED_BINARY,
    IMBALANCED_BINARY,
    IMBALANCED_MULTICLASS,
    LARGE_BINARY,
    OVERLAPPING_BINARY,
    # Standard dataset constants
    STANDARD_BINARY,
    STANDARD_MULTICLASS,
    WELL_SEPARATED_BINARY,
    BinaryDataset,
    MulticlassDataset,
    make_calibrated_binary_dataset,
    make_imbalanced_binary_dataset,
    make_large_binary_dataset,
    make_overlapping_binary_dataset,
    make_realistic_binary_dataset,
    make_realistic_multiclass_dataset,
    make_well_separated_binary_dataset,
)

# Make all available at package level
__all__ = [
    "CALIBRATED_BINARY",
    "IMBALANCED_BINARY",
    "IMBALANCED_MULTICLASS",
    "LARGE_BINARY",
    "OVERLAPPING_BINARY",
    # Standard dataset constants
    "STANDARD_BINARY",
    "STANDARD_MULTICLASS",
    "WELL_SEPARATED_BINARY",
    # Realistic datasets (sklearn-based)
    "BinaryDataset",
    "MulticlassDataset",
    "assert_arrays_close",
    "assert_labels_valid",
    "assert_method_consistency",
    "assert_monotonic_increase",
    "assert_optimization_successful",
    "assert_probability_matrix_valid",
    "assert_valid_confusion_matrix",
    "assert_valid_metric_score",
    # Assertion helpers
    "assert_valid_threshold",
    # Hypothesis strategies
    "beta_bernoulli_calibrated",
    "extreme_probabilities",
    # Data generators (simple)
    "generate_binary_data",
    "generate_calibrated_probabilities",
    "generate_multiclass_data",
    "generate_tied_probabilities",
    "labels_binary_like",
    "make_calibrated_binary_dataset",
    "make_imbalanced_binary_dataset",
    "make_large_binary_dataset",
    "make_overlapping_binary_dataset",
    "make_realistic_binary_dataset",
    "make_realistic_multiclass_dataset",
    "make_well_separated_binary_dataset",
    "multiclass_labels_and_probs",
    "rational_weights",
    "tied_probabilities",
]
