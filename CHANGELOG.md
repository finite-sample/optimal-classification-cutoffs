# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.2] - 2025-12-25

### Added
- **Modern Type Support**: Added `py.typed` marker file for PEP 561 compliance enabling type checker support

### Changed
- **Type Organization**: Modernized type organization following Python 3.12+ best practices
  - Moved `OptimizationResult` from separate types module to `core.py` where it's used
  - Types now follow locality principle - defined where they're used rather than centralized
  - Maintained backward compatibility through `__init__.py` re-exports

### Removed
- **Legacy Type Module**: Removed separate `types.py` module following modern Python practices
- **Outdated Test Imports**: Updated test files to import from main package instead of internal modules

## [1.1.1] - 2025-12-25

### Fixed
- **Documentation Build**: Added nbconvert and pandoc to documentation dependencies for proper Jupyter notebook rendering
- **Code Organization**: Renamed `types_minimal.py` to `types.py` to better reflect its role as the main types module

### Added
- **Examples Integration**: Moved examples to documentation as Jupyter notebooks for better integration

## [1.1.0] - 2025-12-25

### Added
- **Python 3.12+ Requirement**: Updated minimum Python version from 3.11 to 3.12
- **Modern Python Features**: Leveraged Python 3.12+ features for better code readability
  - Match/case statements replace if/elif chains in core routing logic
  - PEP 695 type aliases for cleaner type annotations
  - F-string debug syntax for improved logging

### Changed
- **Minimum Python Version**: Now requires Python 3.12 or higher
- **CI Testing**: Updated to test Python 3.12, 3.13, and 3.14
- **Code Modernization**: Updated core modules with Python 3.12+ syntax
- **Documentation**: Updated installation requirements to reflect Python 3.12+

### Removed
- **Backward Compatibility**: Removed legacy function aliases and deprecated patterns
- **Legacy Documentation**: Cleaned up empty sections and outdated references

## [1.0.0] - 2025-01-01

### Added
- **Mathematical Taxonomy Implementation**: Complete restructuring around mathematical problem types
- **New Module Organization**: `binary.py`, `multilabel.py`, `multiclass.py`, `bayes.py` modules
- **Problem-Type Based API**: Direct functions that encourage understanding of mathematical principles
- **Enhanced Bayes Optimization**: Proper weighted margin rule for multiclass cost matrices
- **Coordinate Ascent Algorithm**: Single-label consistent multiclass optimization
- **Expected Metric Optimization**: Enhanced Dinkelbach algorithm for calibrated probabilities

### Fixed
- **Binary Utility Formula**: Corrected threshold calculation: `τ* = (u_tn - u_fp) / [(u_tp - u_fn) + (u_tn - u_fp)]`
- **Multiclass Margin Rule**: Added weighted margin when cost totals differ across classes
- **Mathematical Correctness**: All algorithms now properly implement theoretical foundations

### Changed
- **BREAKING**: Complete API restructuring around mathematical taxonomy
- **Legacy Compatibility**: Old function names mapped to new implementations
- **Method Name Mapping**: Legacy methods like `sort_scan` → `auto`, `unique_scan` → `independent`
- **Enhanced Validation**: Stricter input requirements with clear error messages
- **Modern Python**: Updated to use union syntax (`X | Y`) for isinstance calls

### Performance
- **O(n log n) Algorithms**: Maintained optimal complexity for piecewise metrics
- **Vectorized Operations**: Enhanced performance for large datasets
- **Memory Efficiency**: Reduced allocations in critical paths

### Documentation
- **Educational Focus**: API designed to teach mathematical concepts
- **Comprehensive Examples**: Updated README with taxonomy-based examples
- **Mathematical Formulas**: Proper documentation of underlying theory

## [0.6.1] - 2025-01-01

### Fixed
- **CRITICAL: Expected Mode Type Mismatch**: Fixed bug in `core.py` where expected mode optimization was incorrectly accessing `.threshold` and `.score` properties on OptimizationResult
  - Changed to use correct API: `result.thresholds[0]` and `result.scores[0]`
  - Two previously failing tests are now working and un-skipped
- **Array Handling Efficiency**: Improved piecewise optimization performance by eliminating unnecessary scalar-to-array conversions
  - Added `_evaluate_metric_scalar_efficient()` helper function
  - Reduced memory allocations and function call overhead in hot paths
- **Safe Division Robustness**: Enhanced `_safe_div()` function to handle all edge cases
  - Now properly handles negative denominators, inf/nan values
  - Provides consistent 0.0 return for problematic divisions
- **Input Validation Gaps**: Strengthened validation functions to catch more edge cases
  - Added None input handling, dtype conversion error catching
  - Enhanced detection of non-finite values (NaN, inf)
  - Added unified validation helper `_validate_threshold_inputs()`

### Changed
- **Function Naming Clarity**: Renamed functions for better API clarity (no backward compatibility)
  - `get_confusion_matrix` → `confusion_matrix_at_threshold`
  - `_confusion_matrix_from_labels` → `confusion_matrix_from_predictions`
  - Updated all internal references and examples

### Developer Notes
- All core unit tests and integration tests pass
- Linting passes with 0 errors (ruff compliant)
- Critical functionality that was previously broken (expected mode) now works correctly
- No breaking changes to public API except function renames (which improve clarity)

## [0.6.0] - 2025-01-01

### Added
- **Enhanced Piecewise Optimization**: Score-based workflows for non-probability inputs
  - New `require_proba` parameter in `optimal_threshold_sortscan()` and related functions
  - Support for logits and arbitrary score ranges (not just [0,1] probabilities)
  - Improved empirical workflows for raw model outputs before calibration
- **New Vectorized Metrics**: Added IoU/Jaccard and Specificity metrics to piecewise registry
  - `iou_vectorized()` function: Intersection over Union = TP / (TP + FP + FN)
  - `specificity_vectorized()` function: True Negative Rate = TN / (TN + FP)
  - Both metrics support O(n log n) sort-and-scan optimization with proper zero-division handling
- **Improved Tie Handling**: O(1) local nudges replace expensive O(n·u) exhaustive search
  - Significantly faster optimization when many samples have identical probabilities
  - Maintains mathematical correctness while improving performance
- **Enhanced Cross-Validation Module**: Complete refactor with multiclass support
  - Added `StratifiedKFold` as default (better class balance preservation)
  - Support for all threshold formats: scalar, array, tuple, dict from expected mode
  - Proper multiclass evaluation with OvR, micro/macro averaging
  - New helper functions: `_extract_thresholds()`, `_evaluate_threshold_on_fold()`
- **Comprehensive Code Quality**: 100% ruff compliance and improved documentation
  - Fixed 140+ linting issues across all modules
  - Improved type annotations and error messages
  - Better docstring consistency following NumPy style

### Changed
- **Weight Handling Improvements**: Better edge case handling in piecewise optimization
  - Fixed single-sample edge cases in weighted optimization
  - Improved handling when all samples belong to same class
  - More robust weight validation and normalization
- **Cross-Validation Behavior**: Replaced dangerous 0.5 fallback with proper error handling
  - CV functions now properly handle all optimization modes (empirical, expected, bayes)
  - Removed hardcoded 0.5 threshold fallback that could mask optimization failures
  - Better error propagation and debugging information

### Fixed
- **Threshold Format Consistency**: Unified handling across empirical/expected/bayes modes
  - Fixed inconsistent return types between different optimization modes
  - Better conversion between scalar/array/dict threshold formats
  - Improved backward compatibility while supporting new features
- **Numerical Stability**: Better handling of edge cases in piecewise optimization
  - Fixed issues with tied probabilities and extreme class imbalances
  - More robust handling of zero-division cases in new metrics
  - Improved floating-point precision in threshold computations

### Performance
- **Tie Handling Optimization**: Dramatic speedup for datasets with many tied probabilities
  - O(1) local nudges instead of O(n·u) exhaustive search where u = unique probabilities
  - Particularly beneficial for discrete probability distributions
- **Improved Caching**: Better memory usage in sort-and-scan algorithms
  - More efficient cumulative sum computations
  - Reduced memory allocations in vectorized metric evaluation

## [0.5.0] - 2024-12-30

### Added
- **Generalized Dinkelbach Framework**: Extended Dinkelbach expected optimization to support any fractional-linear metric
  - New `expected_fractional.py` module with coefficient-based metric representation
  - Support for precision, recall, Jaccard/IoU, Tversky index, accuracy, specificity, F-beta metrics
  - Mathematical framework using `FractionalLinearCoeffs` dataclass for metric coefficients
  - Automatic O(n log n) optimization via sorting and binary search per Dinkelbach iteration
- **Enhanced Expected Mode**: `mode="expected"` now supports 8+ metrics beyond just F-beta
  - Precision, recall, specificity, accuracy, Jaccard, Tversky, F-beta family
  - Automatic fallback to legacy F-beta implementation for unsupported metrics
  - Unified API supporting both binary and multiclass/multilabel optimization
- **Comprehensive Test Coverage**: Added 200+ tests for generalized fractional-linear framework
  - Coefficient correctness tests for all supported metrics
  - Property-based tests using Hypothesis for edge case discovery
  - Integration tests ensuring backward compatibility
  - Edge case handling for extreme probabilities and numerical stability

### Changed
- **Extended Metric Support**: `mode="expected"` no longer limited to F-beta metrics
  - Can optimize expected precision, recall, accuracy, Jaccard, etc. under calibration
  - Maintains mathematical rigor with coefficient-based approach
  - Preserves exact optimization guarantees via Dinkelbach algorithm
- **Improved Framework Design**: Cleaner separation between metric-specific and general optimization
  - `coeffs_for_metric()` provides standardized coefficient mapping
  - `dinkelbach_expected_fractional_binary()` handles any fractional-linear metric
  - `dinkelbach_expected_fractional_ovr()` extends to multiclass with macro/micro/weighted averaging

### Enhanced
- **Mathematical Rigor**: All new metrics maintain theoretical guarantees of Dinkelbach optimization
  - Exact optimization for expected metrics under perfect calibration assumption
  - O(n log n) complexity per iteration with fast convergence (typically <10 iterations)
  - Support for sample weights and comparison operators throughout
- **Backward Compatibility**: Existing F-beta code continues to work unchanged
  - Automatic routing to appropriate implementation (generalized vs legacy)
  - All existing APIs preserve exact behavior for F-beta metrics
  - Comprehensive regression testing ensures no breaking changes

## [0.4.0] - 2024-12-30

### Added
- **Explicit Mode Parameter**: Added `mode` parameter to `get_optimal_threshold()` for clear estimation regime specification
  - `mode="empirical"` (default): Standard empirical optimization using the `method` parameter
  - `mode="bayes"`: Bayes-optimal threshold under calibrated probabilities (requires `utility`)
  - `mode="expected"`: Expected metric optimization (supports multiple metrics, binary and multiclass)
- **Enhanced ThresholdOptimizer**: Extended wrapper class with new parameters
  - Added `mode`, `utility`, and `minimize_cost` parameters for full API coverage
  - Changed primary parameter from `objective` to `metric` with backward compatibility
- **Improved Method Naming**: Renamed `"smart_brute"` to `"unique_scan"` for clarity
  - `"unique_scan"`: Evaluates thresholds at unique probability values
  - Better name reflects the actual algorithm behavior
- **Auto Method Selection**: Changed default optimization method in CV functions from `"smart_brute"` to `"auto"`
  - Automatically selects best method based on metric properties and data size
  - Better performance out-of-the-box for most use cases
- **Comprehensive Test Suite**: Added realistic integration tests with datasets of 100+ samples
  - Tests now use meaningful datasets generated from scikit-learn
  - Added performance tests with 5000+ sample datasets
  - Enhanced test coverage for edge cases and real-world scenarios

### Changed
- **Parameter Naming**: Unified naming convention to use `metric` consistently
  - `ThresholdOptimizer` now uses `metric` parameter instead of `objective`
  - All documentation and examples updated to use `metric` terminology
- **CV Function Defaults**: Cross-validation functions now default to `method="auto"`
  - `cv_threshold_optimization()` and `nested_cv_threshold_optimization()`
  - Provides better performance and method selection automatically
- **Enhanced mode='expected' Support**: Extended expected optimization to support multiple metrics and multiclass
  - Returns tuple `(threshold, expected_score)` for binary classification
  - Returns dict with per-class information for multiclass classification
  - ThresholdOptimizer wrapper properly handles tuple returns

### Deprecated
- **Legacy Parameters**: Several parameters deprecated with clear migration paths
  - `bayes=True` → use `mode="bayes"` instead
  - `method="dinkelbach"` → use `mode="expected"` instead
  - `method="smart_brute"` → use `method="unique_scan"` instead
  - `objective` parameter in `ThresholdOptimizer` → use `metric` instead
- All deprecations emit `DeprecationWarning` with migration instructions

### Fixed
- **Test Suite Robustness**: Fixed 113+ test failures related to deprecated API usage
- **Type Annotations**: Resolved all mypy type checking errors without type ignore comments
- **Sort-Scan Precision**: Improved tolerance handling for edge cases with extreme or tied probabilities
- **Utility Parameter Handling**: Fixed sign convention consistency between Bayes and empirical optimization
- **Wrapper Compatibility**: Fixed ThresholdOptimizer wrapper to handle tuple returns from mode='expected'

### Migration Guide

#### Mode Parameter Usage
```python
# Before (v0.3.x)
threshold = get_optimal_threshold(y, p, method="dinkelbach")
threshold = get_optimal_threshold(None, p, utility={...}, bayes=True)

# After (v0.4.x)
threshold = get_optimal_threshold(y, p, mode="expected")
threshold = get_optimal_threshold(None, p, utility={...}, mode="bayes")
```

#### ThresholdOptimizer Parameter Changes
```python
# Before (v0.3.x)
optimizer = ThresholdOptimizer(objective="f1")

# After (v0.4.x)
optimizer = ThresholdOptimizer(metric="f1")

# With new features
optimizer = ThresholdOptimizer(metric="f1", mode="bayes", utility={"fp": -1, "fn": -5})
```

#### Method Name Changes
```python
# Before (v0.3.x)
threshold = get_optimal_threshold(y, p, method="smart_brute")

# After (v0.4.x)
threshold = get_optimal_threshold(y, p, method="unique_scan")
# Or better, use auto method selection
threshold = get_optimal_threshold(y, p, method="auto")
```

#### Cross-Validation Changes
```python
# Before (v0.3.x) - explicitly needed to specify method
thresholds, scores = cv_threshold_optimization(y, p, method="smart_brute")

# After (v0.4.x) - auto method selection by default
thresholds, scores = cv_threshold_optimization(y, p)  # Uses method="auto"
```

### Technical Details
- Full backward compatibility maintained - existing code will work unchanged
- All deprecated parameters continue to function with warnings
- New test suite with 50+ tests covering new features and deprecation scenarios
- Enhanced type annotations with new `EstimationMode` type
- Comprehensive documentation updates

## [0.3.0] - 2024-12-28

### Added
- **Cost/Benefit-Aware Threshold Optimization**: Complete support for utility-based threshold optimization
  - `bayes_threshold_from_utility()` - Calculate Bayes-optimal thresholds under calibrated probabilities
  - `bayes_threshold_from_costs()` - Convenience wrapper for cost specification
  - `make_linear_counts_metric()` - Create linear utility metrics from confusion matrix counts
  - `make_cost_metric()` - Create cost-sensitive metrics with benefits and penalties
  - Extended `get_optimal_threshold()` API with `utility`, `minimize_cost`, and `bayes` parameters
- **Enhanced Documentation Structure**: Comprehensive documentation with improved navigation
  - Installation guide with troubleshooting
  - Quick start guide with examples
  - Comprehensive user guide covering all features
  - Advanced topics including cost-sensitive optimization
  - Detailed API reference with separate sections
  - FAQ with common issues and solutions
  - Real-world examples and integration patterns
- **Performance Improvements**: Enhanced algorithmic efficiency and robustness
  - Improved edge case handling in sort-scan optimization
  - Better tolerance handling for numerical precision issues
  - Enhanced test coverage with property-based testing

### Changed
- **Version Requirement**: Updated NumPy requirement to `>=1.20.0` for modern API support
- **Code Quality**: Fixed all ruff linting issues for consistent code style
- **Type Annotations**: Improved mypy compliance with complete type coverage
- **Test Robustness**: Enhanced tolerance handling for edge cases in numerical optimization
  - Improved handling of tied probabilities and extreme values
  - Better edge case detection for property-based tests
  - More robust comparison between optimization methods

### Fixed
- **MyPy Compliance**: Fixed type annotation issues in metric factory functions
- **Test Reliability**: Fixed 4 failing tests related to numerical precision and edge cases:
  - `test_sortscan_matches_bruteforce_accuracy` - Enhanced tolerance for boundary cases
  - `test_sortscan_matches_bruteforce_precision` - Improved edge case handling
  - `test_coord_ascent_unsupported_features` - Fixed regex pattern matching
  - `test_piecewise_matches_brute_force` - Added degenerate case filtering
- **Documentation Links**: Fixed broken figure link in README.md
- **Code Style**: Resolved all line length and formatting issues for ruff compliance

### Technical Details
- **Test Suite**: 443 tests passing with comprehensive coverage
- **Quality Checks**: Full compliance with ruff, mypy, and pytest requirements
- **Dependencies**: Maintained compatibility with Python 3.10-3.13
- **Documentation**: Complete Sphinx documentation with proper navigation structure

### Migration Guide
The new cost/benefit-aware optimization is fully backward compatible. Existing code will continue to work unchanged. To use the new features:

```python
# Cost-sensitive optimization (FN costs 5x more than FP)
threshold = get_optimal_threshold(y, p, utility={"fp": -1.0, "fn": -5.0})

# Bayes-optimal for calibrated probabilities (no training data needed)
threshold = get_optimal_threshold(None, p, utility={"fp": -1, "fn": -5}, bayes=True)
```

## [0.2.1] - 2024-09-25

### Previous Release
- O(n log n) sort-and-scan optimization
- Multiclass support with One-vs-Rest
- Cross-validation utilities
- Comprehensive test suite
- Scikit-learn compatible API
