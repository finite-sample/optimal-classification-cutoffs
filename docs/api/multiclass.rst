Multiclass Support
==================

The library provides comprehensive support for multiclass classification using One-vs-Rest and coordinate ascent strategies.

Multiclass Strategies
---------------------

One-vs-Rest (Default)
~~~~~~~~~~~~~~~~~~~~~

The default multiclass strategy treats each class independently:

* Each class gets its own optimized threshold
* Classes are treated as separate binary problems (class vs all others)
* Handles class imbalance better than simple argmax approaches
* Allows different metrics to be optimized for different classes

.. code-block:: python

   from optimal_cutoffs import optimize_thresholds
   import numpy as np

   # 3-class problem
   y_true = np.array([0, 1, 2, 0, 1, 2])
   y_prob = np.array([
       [0.7, 0.2, 0.1],
       [0.1, 0.8, 0.1],
       [0.1, 0.1, 0.8],
       [0.6, 0.3, 0.1],
       [0.2, 0.7, 0.1],
       [0.1, 0.2, 0.7]
   ])

   # Returns array of per-class thresholds
   thresholds = optimize_thresholds(y_true, y_prob, metric='f1')

Coordinate Ascent (Advanced)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For single-label consistency, use coordinate ascent optimization:

.. code-block:: python

   # Coordinate ascent ensures exactly one prediction per sample
   thresholds = optimize_thresholds(
       y_true, y_prob,
       metric='f1',
       method='coord_ascent'
   )

This method couples classes through joint assignment using ``argmax(P - tau)`` and ensures exactly one prediction per sample.

Multiclass Prediction Logic
---------------------------

The library uses sophisticated decision rules for multiclass prediction:

1. **Apply per-class thresholds** to get binary predictions for each class
2. **Multiple classes above threshold**: Predict the one with highest probability
3. **No classes above threshold**: Predict the class with highest probability (standard argmax)

.. code-block:: python

   from optimal_cutoffs import ThresholdOptimizer

   # Fit optimizer
   optimizer = ThresholdOptimizer(metric='f1')
   optimizer.fit(y_true, y_prob)

   # Make predictions (handles the logic automatically)
   y_pred = optimizer.predict(y_prob_new)

Averaging Strategies
--------------------

Control how metrics are aggregated across classes:

Macro Averaging (Default)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Equal weight to all classes
   thresholds = optimize_thresholds(
       y_true, y_prob,
       metric='f1',
       average='macro'
   )

Micro Averaging
~~~~~~~~~~~~~~~

For empirical fits with ``average="micro"``, automatic selection uses one
cutoff shared by all classes. The cutoff is fitted on the pooled class
indicators. The predictor picks the highest-probability eligible class, or
the highest-probability class if none are eligible. Both rules give the
same class with a shared cutoff. The reported micro F1, precision and
recall therefore equal the fraction of correctly classified observations,
weighted by ``sample_weight`` when supplied.

.. code-block:: python

   result = optimize_thresholds(y_true, y_prob, metric="f1", average="micro")
   predictions = result.predict(y_prob)
   assert np.isclose(result.score, np.mean(predictions == y_true))

Weighted Averaging
~~~~~~~~~~~~~~~~~~

The macro F1 solver gives each class equal weight. With ``average="weighted"``,
each class contributes in proportion to the total weight of its true observations.
This applies to the objective during the margin search and to the reported score.
For independent one-vs-rest fits, the class-specific cutoffs can still be fitted
separately because the support weights are fixed.

.. code-block:: python

   # Weight by class frequency
   thresholds = optimize_thresholds(
       y_true, y_prob,
       metric='f1',
       average='weighted'
   )

Multiclass Metrics
------------------

Exclusive Single-Label Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For standard multiclass accuracy (exactly one prediction per sample):

.. code-block:: python

   from optimal_cutoffs.metrics import multiclass_metric_exclusive

   # Computes accuracy using margin-based decision rule
   accuracy = multiclass_metric_exclusive(
       y_true, y_prob, thresholds,
       metric_name='accuracy',
       comparison='>'
   )

One-vs-Rest Metrics
~~~~~~~~~~~~~~~~~~~

For independent per-class evaluation:

.. code-block:: python

   from optimal_cutoffs.metrics import multiclass_metric

   # Get per-class confusion matrices
   from optimal_cutoffs.metrics import get_multiclass_confusion_matrix

   confusion_matrices = get_multiclass_confusion_matrix(
       y_true, y_prob, thresholds, comparison='>'
   )

   # Compute macro-averaged F1
   macro_f1 = multiclass_metric(confusion_matrices, 'f1', average='macro')

Advanced Multiclass Usage
-------------------------

Class Imbalance Handling
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.utils.class_weight import compute_sample_weight

   # Compute sample weights to handle imbalance
   sample_weights = compute_sample_weight('balanced', y_true)

   thresholds = optimize_thresholds(
       y_true, y_prob,
       metric='f1',
       sample_weight=sample_weights
   )

Custom Class Costs
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Different costs for different classes (not yet implemented)
   # This is a planned feature for future versions

   # Workaround: Optimize each class separately with different costs
   n_classes = y_prob.shape[1]
   custom_thresholds = []

   for class_idx in range(n_classes):
       # Convert to binary problem
       y_binary = (y_true == class_idx).astype(int)
       y_prob_binary = y_prob[:, class_idx]

       # Apply class-specific costs
       class_costs = {"fp": -1.0, "fn": -5.0}  # Customize per class

       result = optimize_thresholds(
           y_binary, y_prob_binary,
           utility=class_costs
       )
       custom_thresholds.append(result.thresholds[0])

Performance Considerations
---------------------------

Memory Usage
~~~~~~~~~~~~

Multiclass problems require more memory:

* **O(n × k)** for probability matrices (n samples, k classes)
* **O(k)** for threshold arrays
* One-vs-Rest creates k binary problems internally

For large problems, consider:

.. code-block:: python

   # Process classes in batches
   def optimize_multiclass_batched(y_true, y_prob, batch_size=1000):
       n_samples = len(y_true)
       all_thresholds = []

       for start_idx in range(0, n_samples, batch_size):
           end_idx = min(start_idx + batch_size, n_samples)

           batch_true = y_true[start_idx:end_idx]
           batch_prob = y_prob[start_idx:end_idx]

           thresholds = optimize_thresholds(batch_true, batch_prob, metric='f1')
           all_thresholds.append(thresholds)

       # Combine results (example: median)
       return np.median(all_thresholds, axis=0)

Speed Optimization
~~~~~~~~~~~~~~~~~~

For multiclass problems:

* Use ``method='auto'`` for intelligent algorithm selection
* Consider ``method='sort_scan'`` for large datasets with piecewise metrics
* Use ``method='smart_brute'`` for small datasets or highest precision

.. code-block:: python

   # Fast optimization for large multiclass datasets
   thresholds = optimize_thresholds(
       y_true, y_prob,
       metric='f1',
       method='auto',  # Automatically selects best method
       comparison='>'  # Slightly faster than '>='
   )
