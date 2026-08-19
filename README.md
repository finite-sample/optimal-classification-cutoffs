# Optimal Classification Cutoffs

[![Python application](https://github.com/finite-sample/optimal-classification-cutoffs/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/optimal-classification-cutoffs/actions/workflows/ci.yml)
[![Downloads](https://pepy.tech/badge/optimal-classification-cutoffs)](https://pepy.tech/project/optimal-classification-cutoffs)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://finite-sample.github.io/optimal-classification-cutoffs/)
[![PyPI version](https://img.shields.io/pypi/v/optimal-classification-cutoffs.svg)](https://pypi.org/project/optimal-classification-cutoffs/)
[![Python](https://img.shields.io/pypi/pyversions/optimal-classification-cutoffs.svg)](https://pypi.org/project/optimal-classification-cutoffs/)

**Optimize classification thresholds for improved model performance.**

Most classifiers output probabilities that need thresholds for decisions. The default 0.5 threshold is often suboptimal for real objectives like F1, precision/recall, or business costs. This library finds optimal thresholds using efficient O(n log n) algorithms, with significant improvements possible especially on imbalanced datasets.

## Why Optimize Classification Thresholds

```python
# Default 0.5 threshold
y_pred = (model.predict_proba(X)[:, 1] >= 0.5).astype(int)
# F1 Score: 0.654

# Optimized threshold
from optimal_cutoffs import optimize_thresholds

result = optimize_thresholds(y_true, y_scores, metric="f1")
y_pred = result.predict(y_scores_test)
# F1 Score: 0.891 (improvement depends on dataset characteristics)
```

**Key considerations:** The 0.5 threshold assumes equal costs and balanced classes. Many real-world problems have imbalanced data (e.g., fraud detection: 1% positive rate) and asymmetric costs (e.g., false negatives may be more costly than false positives). In such cases, optimal thresholds often differ significantly from 0.5.

## Installation

```bash
pip install optimal-classification-cutoffs
```

**Optional Performance Boost:**
```bash
# For algorithmic speedups with optional Numba acceleration
pip install optimal-classification-cutoffs[performance]

# For Jupyter examples and visualizations
pip install optimal-classification-cutoffs[examples]
```

> **Python 3.14+ Support**: The package works on all Python versions 3.12+, including cutting-edge Python 3.14. Numba acceleration is optional and will automatically fall back to pure Python when unavailable.

## Quick Start

### Binary Classification

```python
from optimal_cutoffs import optimize_thresholds

# Your existing model probabilities
y_scores = model.predict_proba(X_test)[:, 1]

# Find optimal threshold (exact solution, O(n log n))
result = optimize_thresholds(y_true, y_scores, metric="f1")
print(f"Optimal threshold: {result.threshold:.3f}")  # May differ from 0.5
print(f"Expected F1: {result.scores[0]:.3f}")

# Make optimal predictions
y_pred = result.predict(y_scores_new)
```

### Multiclass Classification: Per-Class Thresholds

```python
import numpy as np
from optimal_cutoffs import optimize_thresholds

# Multiclass probabilities (n_samples, n_classes)
y_scores = model.predict_proba(X_test)

# Automatically detects multiclass, optimizes per-class thresholds
result = optimize_thresholds(y_true, y_scores, metric="f1")
print(f"Per-class thresholds: {result.thresholds}")
print(f"Task detected: {result.task.value}")  # "multiclass"
print(f"Method used: {result.method}")  # "coord_ascent"

# Predictions use optimal thresholds
y_pred = result.predict(y_scores_new)
```

### Cost-Sensitive Decisions: No Thresholds Needed

```python
from optimal_cutoffs import optimize_decisions

# Cost matrix: rows=true class, cols=predicted class
# False negatives cost 10x more than false positives
cost_matrix = [[0, 1], [10, 0]]

result = optimize_decisions(y_score, cost_matrix)
y_pred = result.predict(y_score_new)  # Bayes-optimal decisions
```

## API Overview

**API Overview:**

### Core Functions

```python
from optimal_cutoffs import optimize_thresholds, optimize_decisions

# For threshold-based optimization (F1, precision, recall, etc.)
result = optimize_thresholds(y_true, y_scores, metric="f1")

# For cost matrix optimization (no thresholds)
result = optimize_decisions(y_score, cost_matrix)
```

### Progressive Disclosure: Power When You Need It

```python
from optimal_cutoffs import metrics, bayes, cv, algorithms

# Custom metrics
custom_f2 = lambda tp, tn, fp, fn: (5 * tp) / (5 * tp + 4 * fn + fp)
metrics.register("f2", custom_f2)

# Cross-validation with threshold tuning (operates on labels and scores)
thresholds, scores = cv.cross_validate(y_true, y_scores, metric="f1")

# Advanced algorithms
result = algorithms.multiclass.coordinate_ascent(y_true, y_scores)
```

### Auto-Selection with Explanations

Everything is explainable. The library tells you what it detected and why:

```python
result = optimize_thresholds(y_true, y_scores)  # All defaults

print(f"Task: {result.task.value}")  # "binary" (auto-detected)
print(f"Method: {result.method}")  # "sort_scan" (O(n log n))
print(
    f"Notes: {result.notes}"
)  # ["Detected binary task...", "Selected sort_scan for O(n log n) optimization..."]
```

## Why This Works: Mathematical Foundations

### Piecewise Structure
Most metrics (F1, precision, recall) are **piecewise-constant** in threshold τ. Sorting scores once enables **exact** optimization in O(n log n) time.

### Bayes Decision Theory
Under calibrated probabilities, optimal binary thresholds have **closed form**:
```
τ* = cost_fp / (cost_fp + cost_fn)
```
Independent of class priors, depends only on cost ratio.

### Multiclass Extensions
- **One-vs-Rest:** Independent per-class thresholds (macro averaging)
- **Coordinate Ascent:** Coupled thresholds for single-label consistency
- **General Costs:** Skip thresholds, apply Bayes rule on probability vectors

## Performance

- **O(n log n)** exact optimization for piecewise metrics
- **O(1)** closed-form solutions for cost-sensitive objectives
- **Optional Numba acceleration** with automatic pure Python fallback

The O(n log n) sort-scan algorithm provides significant speedups compared to naive approaches:
- **50-300× faster** than grid search (101 points)
- **1000-8000× faster** than exhaustive unique threshold evaluation
- Performance scales as O(n log n) vs O(n·k) for naive methods
- Exact solutions guaranteed for piecewise-constant metrics

## Complete Example: Real Impact

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from optimal_cutoffs import optimize_thresholds

# Realistic imbalanced dataset (like fraud detection)
X, y = make_classification(n_samples=1000, weights=[0.9, 0.1], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Train any classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_scores = model.predict_proba(X_test)[:, 1]

# ❌ Default threshold
y_pred_default = (y_scores >= 0.5).astype(int)
f1_default = f1_score(y_test, y_pred_default)
print(f"Default F1: {f1_default:.3f}")  # ~0.65

# ✅ Optimal threshold
result = optimize_thresholds(y_test, y_scores, metric="f1")
y_pred_optimal = result.predict(y_scores)
f1_optimal = f1_score(y_test, y_pred_optimal)
print(f"Optimal F1: {f1_optimal:.3f}")  # ~0.89

improvement = (f1_optimal - f1_default) / f1_default * 100
print(f"Improvement: {improvement:+.1f}%")  # ~+40%
```

## When to Use This

**Perfect for:**
- Imbalanced classification (fraud, medical, spam)
- Cost-sensitive decisions (business impact)
- Performance-critical applications (exact solutions)
- Research requiring theoretical optimality

**Not needed for:**
- Perfectly balanced classes with symmetric costs
- Problems requiring probabilistic outputs
- Uncalibrated models (calibrate first)

## Advanced Usage

### Cross-Validation with Thresholds
```python
from optimal_cutoffs import cv

# Cross-validation for threshold selection.
# Returns (thresholds, scores): the threshold chosen on each fold's training part
# and the metric it achieved on that fold's held-out part.
thresholds, scores = cv.cross_validate(
    y_true,
    y_scores,
    metric="f1",
    cv=5,
)
```

### Custom Metrics
```python
from optimal_cutoffs import metrics


# Register custom Fβ score
def f_beta(tp, tn, fp, fn, beta=2.0):
    return (1 + beta**2) * tp / ((1 + beta**2) * tp + beta**2 * fn + fp)


metrics.register("f2", lambda tp, tn, fp, fn: f_beta(tp, tn, fp, fn, 2.0))

# Use like any built-in metric
result = optimize_thresholds(y_true, y_scores, metric="f2")
```

### Multiple Metrics
```python
# Optimize different metrics
f1_result = optimize_thresholds(y_true, y_scores, metric="f1")
precision_result = optimize_thresholds(y_true, y_scores, metric="precision")

print(f"F1 optimal τ: {f1_result.threshold:.3f}")
print(f"Precision optimal τ: {precision_result.threshold:.3f}")
```

## References

- Lipton et al. (2014) *Optimal Thresholding of Classifiers to Maximize F1*
- Elkan (2001) *The Foundations of Cost-Sensitive Learning*
- Dinkelbach (1967) *Nonlinear Fractional Programming*
- Platt (1999) *Probabilistic Outputs for Support Vector Machines*
