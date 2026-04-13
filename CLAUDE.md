# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Version Information
- **Current Version**: 2.0.1
- **Python Support**: 3.12, 3.13, 3.14

## Development Commands

### Package Installation
```bash
# Install in development mode
python -m pip install -e .

# Install with example dependencies
python -m pip install -e ".[examples]"
```

### Testing
```bash
# Run all tests (640+ tests as of v1.0.0)
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_multiclass.py -v

# Run specific test
python -m pytest tests/test_multiclass.py::test_multiclass_confusion_matrix -v

# Run tests for new features
python -m pytest tests/test_piecewise.py -v              # O(n log n) optimization
python -m pytest tests/test_multiclass_coord.py -v      # Coordinate ascent
python -m pytest tests/test_registry_integration.py -v  # Registry system
python -m pytest tests/test_property_based.py -v        # Property-based tests
```

### Code Quality
```bash
# Run linting (passes with 0 errors)
ruff check optimal_cutoffs/

# Apply code formatting
ruff format optimal_cutoffs/

# Run docstring linting
uv run pydoclint optimal_cutoffs/

# Generate new baseline for pydoclint (after fixing violations)
uv run pydoclint --generate-baseline=True optimal_cutoffs/
```

### Examples
```bash
# Run basic binary example
python examples/01_quickstart.py

# Run multiclass example
python examples/03_multiclass.py
```

## Architecture Overview

This library provides optimal threshold selection for binary and multiclass classification problems. The core architecture is built around several key components that work together:

### Binary vs Multiclass Detection
The library automatically detects whether input is binary (1D probabilities) or multiclass (2D probability matrix) and routes to appropriate implementations. Binary classification uses legacy optimizers for backward compatibility, while multiclass uses the newer general framework.

### Multiclass Strategies
The library supports two main multiclass approaches:

**One-vs-Rest (OvR)** - Default strategy where each class gets its own optimized threshold:
- Treats each class as a separate binary problem (class vs all others)
- Optimizes thresholds independently per class
- Handles class imbalance better than simple argmax approaches
- Allows different metrics to be optimized for different classes

**Coordinate Ascent** - Advanced strategy for single-label consistency:
- Couples classes through joint assignment using argmax(P - tau)
- Ensures exactly one prediction per sample (single-label)
- Iteratively optimizes per-class thresholds while fixing others
- Better suited for applications requiring strict single-label predictions

### Core Components

**metrics.py**: Contains the enhanced metric registry system and confusion matrix utilities. The `METRIC_REGISTRY` is a global dictionary that allows registration of custom metrics that accept `(tp, tn, fp, fn)` parameters. The `VECTORIZED_REGISTRY` stores optimized vectorized implementations for O(n log n) algorithms. Built-in metrics (F1, accuracy, precision, recall) have both scalar and vectorized versions. For multiclass, special handling ensures proper micro averaging (avoiding TN double-counting in OvR).

**optimizers.py**: Implements threshold optimization algorithms. `get_optimal_threshold()` serves as the main entry point that auto-detects binary vs multiclass and routes accordingly. Optimization methods include:
- "auto" (automatically selects best method, default)
- "sort_scan" (O(n log n) algorithm for piecewise metrics with vectorized implementations)
- "smart_brute" (evaluates all unique probabilities)
- "minimize" (scipy optimization with enhanced fallbacks)
- "gradient" (simple gradient ascent)
- "coord_ascent" (coordinate ascent for coupled multiclass optimization, single-label consistent)

**piecewise.py**: Implements the O(n log n) sort-and-scan optimization kernel for piecewise-constant metrics. This module provides exact optimization using cumulative sums and vectorized operations, achieving significant performance improvements for large datasets.

**multiclass_coord.py**: Implements coordinate-ascent optimization for multiclass thresholds that ensures single-label consistency. Unlike One-vs-Rest approaches, this couples classes through joint assignment using argmax of shifted scores: argmax_j (p_ij - tau_j).

**wrapper.py**: Provides the high-level `ThresholdOptimizer` class with scikit-learn style fit/predict API. The predict method for multiclass implements sophisticated decision rules that vary by optimization method (OvR vs coordinate ascent).

**validation.py**: Comprehensive input validation utilities ensuring robust API behavior with clear error messages.

**types.py**: Type definitions and protocols for better code maintainability and IDE support.

**cv.py**: Cross-validation utilities that work with the threshold optimization functions to provide robust threshold estimates.

### Input Validation
The library includes comprehensive input validation that ensures:
- Class labels are consecutive integers starting from 0 (required for OvR)
- No NaN or infinite values in probability arrays
- Matching dimensions between labels and probabilities
- Proper array shapes for binary (1D) vs multiclass (2D) inputs

### Prediction Logic for Multiclass
The multiclass prediction strategy balances threshold-based decisions with practical fallbacks:
1. Apply per-class thresholds to get binary predictions for each class
2. For each sample, if multiple classes are above threshold, predict the one with highest probability
3. If no classes are above threshold, predict the class with highest probability (standard argmax)

This approach maintains the benefits of optimized thresholds while ensuring every sample gets a prediction.

## Version 1.0.0 Mathematical Taxonomy Release

### Enhanced Score-Based Workflows
- **Non-Probability Support**: New `require_proba=False` parameter enables optimization on logits and arbitrary score ranges
- **Empirical Score Optimization**: Support for raw model outputs before calibration
- **Flexible Input Validation**: Better handling of scores outside [0,1] range

### New Vectorized Metrics
- **IoU/Jaccard Metric**: `iou_vectorized()` with proper zero-division handling
- **Specificity Metric**: `specificity_vectorized()` for True Negative Rate optimization
- **O(n log n) Support**: Both metrics integrated into sort-and-scan optimization

### Performance Optimizations
- **Improved Tie Handling**: O(1) local nudges replace expensive O(n·u) exhaustive search
- **Better Weight Handling**: Fixed edge cases in weighted optimization for single samples and same-class scenarios
- **Memory Efficiency**: Reduced allocations in vectorized metric evaluation

### Cross-Validation Enhancements
- **StratifiedKFold Default**: Better class balance preservation in CV splits
- **Multiclass Support**: Proper handling of OvR and micro/macro averaging in CV
- **Threshold Format Handling**: Support for scalar, array, tuple, dict thresholds from all modes
- **Error Handling**: Removed dangerous 0.5 fallback, improved error propagation

### Code Quality & Testing
- **640+ Tests**: Comprehensive test suite including new functionality
- **100% Ruff Compliance**: Fixed 140+ linting issues across all modules
- **Improved Documentation**: Better docstrings, type annotations, and examples
- **Numerical Stability**: Better handling of edge cases and floating-point precision

### Architectural Improvements
- **Unified Result Types**: New `ThresholdResult` class for consistent API returns
- **Better Validation**: Enhanced input validation with clear error messages
- **Modular Design**: Cleaner separation between optimization, metrics, and validation

## Development Notes

### Key Design Principles
- **Performance**: Prioritize O(n log n) algorithms where possible
- **Robustness**: Comprehensive input validation and error handling
- **Flexibility**: Support multiple optimization strategies and metrics
- **Compatibility**: Maintain backward compatibility while adding new features

### Testing Strategy
- **Unit tests**: Core functionality with edge cases
- **Property-based tests**: Mathematical properties using Hypothesis
- **Integration tests**: End-to-end workflows and API compatibility
- **Performance tests**: Algorithmic complexity verification
- **Regression tests**: Prevent breaking changes

### Code Organization Best Practices
- **Modular design**: Separate concerns (optimization, metrics, validation)
- **Registry pattern**: Extensible metric system
- **Type safety**: Comprehensive type annotations
- **Documentation**: Docstrings follow NumPy style
- **Error handling**: Informative error messages with context

### Performance Considerations
- Use `method="auto"` for optimal performance (automatically selects best algorithm)
- For large datasets (n > 10,000), the `sort_scan` method provides significant speedups
- Sample weights are supported but may reduce performance for some methods
- The `coord_ascent` method is currently limited to F1 metric and ">" comparison

### Common Patterns
```python
# Recommended usage pattern
from optimal_cutoffs import get_optimal_threshold

# Binary classification - auto method selection
threshold = get_optimal_threshold(y_true, y_prob, metric="f1", method="auto")

# Multiclass classification - coordinate ascent for single-label consistency
thresholds = get_optimal_threshold(y_true, y_prob, metric="f1", method="coord_ascent")

# Performance-critical applications
threshold = get_optimal_threshold(y_true, y_prob, metric="f1", method="sort_scan")
```
