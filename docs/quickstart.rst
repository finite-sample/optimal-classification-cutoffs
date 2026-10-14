Quick Start
===========

This guide will get you up and running with optimal-classification-cutoffs in just a few minutes.

Basic Binary Classification
----------------------------

The simplest use case is finding the optimal threshold for binary classification:

.. code-block:: python

   from optimal_cutoffs import optimize_thresholds
   import numpy as np

   # Your binary classification data
   y_true = np.array([0, 0, 1, 1, 0, 1])
   y_prob = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])

   # Find optimal threshold for F1 score
   result = optimize_thresholds(y_true, y_prob, metric='f1')
   print(f"Optimal threshold: {result.threshold:.3f}")

   # Make predictions
   predictions = result.predict(y_prob)
   print(f"Predictions: {predictions}")

Other Metrics
~~~~~~~~~~~~~

You can optimize for different metrics:

.. code-block:: python

   # Optimize for accuracy
   result_acc = optimize_thresholds(y_true, y_prob, metric='accuracy')

   # Optimize for precision
   result_prec = optimize_thresholds(y_true, y_prob, metric='precision')

   # Optimize for recall
   result_rec = optimize_thresholds(y_true, y_prob, metric='recall')

   print(f"Accuracy threshold: {result_acc.threshold:.3f}")
   print(f"Precision threshold: {result_prec.threshold:.3f}")
   print(f"Recall threshold: {result_rec.threshold:.3f}")

Multiclass Classification
-------------------------

For multiclass problems, the library automatically detects the problem type and returns per-class thresholds:

.. code-block:: python

   # Multiclass example with 3 classes
   y_true = np.array([0, 1, 2, 0, 1, 2])
   y_prob = np.array([
       [0.7, 0.2, 0.1],  # Sample 1: likely class 0
       [0.1, 0.8, 0.1],  # Sample 2: likely class 1
       [0.1, 0.1, 0.8],  # Sample 3: likely class 2
       [0.6, 0.3, 0.1],  # Sample 4: likely class 0
       [0.2, 0.7, 0.1],  # Sample 5: likely class 1
       [0.1, 0.2, 0.7]   # Sample 6: likely class 2
   ])

   # Get per-class optimal thresholds
   result = optimize_thresholds(y_true, y_prob, metric='f1')
   print(f"Optimal thresholds per class: {result.thresholds}")
   print(f"Task detected: {result.task.value}")

   # Make predictions
   predictions = result.predict(y_prob)

Progressive Disclosure: Power Tools
-----------------------------------

API 2.0.0 uses progressive disclosure - simple for basic use, powerful when needed:

.. code-block:: python

   from optimal_cutoffs import optimize_thresholds, cv, metrics
   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import RandomForestClassifier

   # Generate sample data
   X = np.random.randn(1000, 5)
   y = (X[:, 0] + X[:, 1] > 0).astype(int)

   # Split data
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

   # Train a classifier
   clf = RandomForestClassifier(random_state=42)
   clf.fit(X_train, y_train)
   y_prob_train = clf.predict_proba(X_train)[:, 1]
   y_prob_test = clf.predict_proba(X_test)[:, 1]

   # Simple: Optimize threshold
   result = optimize_thresholds(y_train, y_prob_train, metric='f1')
   y_pred = result.predict(y_prob_test)

   print(f"Optimal threshold: {result.threshold:.3f}")
   print(f"Test accuracy: {np.mean(y_pred == y_test):.3f}")
   print(f"Method used: {result.method}")

   # Advanced: Cross-validation with threshold tuning
   cv_scores = cv.cross_validate(clf, X, y, metric='f1')

Optimization Methods
--------------------

The library provides several optimization methods:

.. code-block:: python

   # Auto method selection (recommended)
   result = optimize_thresholds(y_true, y_prob, metric='f1', method='auto')

   # Fast O(n log n) algorithm for piecewise metrics
   result = optimize_thresholds(y_true, y_prob, metric='f1', method='sort_scan')

   # Scipy-based continuous optimization
   result = optimize_thresholds(y_true, y_prob, metric='f1', method='minimize')

   # Explainable auto-selection
   print(f"Method selected: {result.method}")
   print(f"Reasoning: {result.notes}")

Cost-Sensitive Optimization
---------------------------

For applications where different types of errors have different costs:

.. code-block:: python

   from optimal_cutoffs import optimize_decisions, bayes

   # Option 1: Use cost matrix (no thresholds needed)
   cost_matrix = [[0, 1], [5, 0]]  # FN costs 5x more than FP
   result = optimize_decisions(y_prob, cost_matrix)
   predictions = result.predict(y_prob)

   # Option 2: Bayes-optimal threshold calculation
   threshold = bayes.threshold(cost_fp=1.0, cost_fn=5.0)
   print(f"Bayes-optimal threshold: {threshold:.3f}")  # = 1/(1+5) = 0.167

Next Steps
----------

* Read the :doc:`user_guide` for detailed explanations and advanced features
* Check out :doc:`examples <auto_examples/index>` for worked examples
* Explore :doc:`advanced` topics like cross-validation and custom metrics
* Understand the :doc:`theory` behind why this approach works better than standard methods
