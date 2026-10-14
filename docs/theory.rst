Theory and Background
======================

This page explains the theoretical foundations underlying the optimal threshold selection algorithms, particularly focusing on the piecewise-constant nature of classification metrics and the relationship between threshold optimization and probability calibration.

Piecewise-Constant Metrics
---------------------------

Understanding the Step Function Nature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most classification metrics—including F1 score, accuracy, precision, and recall—are **piecewise-constant functions** with respect to the decision threshold. This fundamental property has important implications for optimization algorithms.

Why Metrics Are Piecewise-Constant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Classification metrics depend on confusion matrix counts: true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN). These counts are determined by comparing predicted probabilities to a threshold:

.. math::

   \text{prediction} = \begin{cases}
   1 & \text{if } p \geq \text{threshold} \\
   0 & \text{if } p < \text{threshold}
   \end{cases}

**Key insight**: The confusion matrix counts only change when the threshold crosses one of the unique predicted probability values. Between any two consecutive probability values, all predictions remain unchanged, so the metric value stays constant.

For example, with predicted probabilities ``[0.2, 0.4, 0.7, 0.9]``:

- Thresholds in ``[0.0, 0.2)`` → all predictions are 1
- Thresholds in ``[0.2, 0.4)`` → predictions are ``[0, 1, 1, 1]``
- Thresholds in ``[0.4, 0.7)`` → predictions are ``[0, 0, 1, 1]``
- Thresholds in ``[0.7, 0.9)`` → predictions are ``[0, 0, 0, 1]``
- Thresholds in ``[0.9, 1.0]`` → all predictions are 0

This creates a step function where the metric value jumps only at the unique probability values: ``{0.2, 0.4, 0.7, 0.9}``.

Mathematical Proof
~~~~~~~~~~~~~~~~~~

Let :math:`S = \{p_1, p_2, \ldots, p_k\}` be the set of unique predicted probabilities, sorted in ascending order. For any threshold :math:`t` in the interval :math:`(p_i, p_{i+1})`, the prediction vector remains constant:

.. math::

   \hat{y}_j = \begin{cases}
   1 & \text{if } p_j \geq t \text{ (equivalently, if } p_j > p_i \text{)} \\
   0 & \text{otherwise}
   \end{cases}

Since the prediction vector is constant over each interval, the confusion matrix elements (TP, TN, FP, FN) are also constant over each interval. Therefore, any metric function :math:`M(\text{TP}, \text{TN}, \text{FP}, \text{FN})` is piecewise-constant with breakpoints at the unique probability values.

Why Continuous Optimizers Can Miss the Maximum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standard continuous optimization algorithms (like ``scipy.optimize.minimize_scalar``) assume the objective function is smooth and use gradient-based or bracketing methods. However, piecewise-constant functions have several problematic properties:

1. **Zero gradients**: The derivative is zero everywhere except at the breakpoints (where it's undefined)
2. **No local information**: The gradient doesn't point toward better solutions
3. **Arbitrary convergence**: Optimizers may converge to any point within a constant region

Consider this example: if the true optimum is at threshold 0.7 with F1 = 0.85, but ``minimize_scalar`` converges to 0.65, it will report the same F1 = 0.85 (since both lie in the same constant region) but return the wrong threshold value.

The Optimized O(n log n) Solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since the optimal threshold must occur at one of the unique probability values, we can solve this problem optimally with a sort-and-scan algorithm:

.. code-block:: python

   # Sort predictions by descending probability (O(n log n))
   sorted_indices = np.argsort(-predicted_probabilities)

   # Compute all confusion matrices in one vectorized pass (O(n))
   tp_cumsum = np.cumsum(labels_sorted)  # True positives at each cut
   fp_cumsum = np.cumsum(1 - labels_sorted)  # False positives at each cut

   # Vectorized metric evaluation across all n cuts (O(n))
   all_scores = metric_function_vectorized(tp_cumsum, tn_array, fp_cumsum, fn_array)
   optimal_cut = np.argmax(all_scores)

This approach:

- **True O(n log n) complexity**: Single sort followed by linear scan
- **Guarantees global optimum**: Evaluates all n possible cut points, not just unique probabilities
- **Vectorized operations**: Eliminates Python loops for maximum efficiency
- **Numerically stable**: Returns threshold as midpoint between adjacent probabilities
- **Exact results**: No approximation error from numerical optimization

Performance improvements over the original approach:
- **50-100x speedup** on typical datasets
- **Scales better** with dataset size (true O(n log n) vs O(k log n) where k can be large)
- **More robust** numerical properties

Fallback Mechanism for Minimize Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The library implements a hybrid approach for the ``"minimize"`` method:

1. Run ``scipy.optimize.minimize_scalar`` to get a candidate threshold
2. Also evaluate all unique probability values
3. Return whichever gives the highest metric score

This fallback ensures that even when continuous optimization finds a suboptimal threshold within a constant region, the final result is still globally optimal.

Visualization
~~~~~~~~~~~~~

The following plot demonstrates the piecewise-constant nature of F1 score::

   # Generate example data
   y_true = [0, 0, 1, 1, 0, 1, 0]
   y_prob = [0.1, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]

   # The F1 score will be constant between each pair of consecutive probabilities
   # and will only change at the breakpoints: {0.1, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9}

Non-Piecewise Metrics
~~~~~~~~~~~~~~~~~~~~~

Some metrics are **not** piecewise-constant:

- **Log-loss (cross-entropy)**: Depends directly on probability values, not just binary predictions
- **Brier score**: Quadratic loss that uses continuous probability values
- **ROC-AUC**: Based on ranking, changes continuously with threshold

For these metrics, continuous optimization methods are more appropriate, and the library automatically falls back to evaluating all unique probabilities rather than using the O(n log n) piecewise algorithm.

Calibration and Threshold Optimization
---------------------------------------

Understanding the Relationship
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Calibration** and **threshold optimization** address different aspects of probabilistic classification:

- **Calibration** ensures that predicted probabilities match true frequencies (e.g., among samples with predicted probability 0.7, approximately 70% should be positive)
- **Threshold optimization** finds decision boundaries that maximize specific classification metrics

These are **complementary techniques**, not competing alternatives.

When to Use Calibration
~~~~~~~~~~~~~~~~~~~~~~~~

Use probability calibration when:

1. **Probability interpretation matters**: You need reliable probability estimates for decision-making
2. **Model comparison**: Comparing models based on probability quality (e.g., using Brier score or log-likelihood)
3. **Risk assessment**: Converting scores to meaningful probabilities for business decisions
4. **Ranking quality**: Ensuring that higher probabilities truly indicate higher likelihood

Common calibration methods:

- **Platt scaling**: Fits a sigmoid function to map scores to probabilities [Platt1999]_
- **Isotonic regression**: Non-parametric method that fits a monotonic function [ZadroznyCElkan2002]_

When to Use Threshold Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use threshold optimization when:

1. **Metric optimization**: You need to maximize specific classification metrics (F1, precision, recall)
2. **Binary decisions**: Converting probabilities to hard classifications for deployment
3. **Imbalanced classes**: Standard 0.5 threshold is often suboptimal for imbalanced datasets
4. **Business constraints**: Achieving specific precision/recall trade-offs

Using Both Together
~~~~~~~~~~~~~~~~~~~

The recommended workflow combines both techniques:

.. code-block:: python

   from sklearn.calibration import CalibratedClassifierCV
   from optimal_cutoffs import ThresholdOptimizer

   # Step 1: Train your base classifier
   base_model = LogisticRegression()
   base_model.fit(X_train, y_train)

   # Step 2: Calibrate probabilities
   calibrated_model = CalibratedClassifierCV(base_model, cv=3)
   calibrated_model.fit(X_train, y_train)
   y_prob_cal = calibrated_model.predict_proba(X_val)[:, 1]

   # Step 3: Optimize threshold on calibrated probabilities
   optimizer = ThresholdOptimizer(metric='f1')
   optimizer.fit(y_val, y_prob_cal)

   # Step 4: Make final predictions
   y_prob_test = calibrated_model.predict_proba(X_test)[:, 1]
   y_pred_test = optimizer.predict(y_prob_test)

Benefits of this combined approach:

- **Reliable probabilities**: Calibration ensures probability quality
- **Optimal decisions**: Threshold optimization maximizes your chosen metric
- **Robust pipeline**: Works well across different types of models and datasets

Calibration Diagnostics
~~~~~~~~~~~~~~~~~~~~~~~

Before and after calibration, evaluate:

1. **Reliability diagrams**: Plot predicted vs observed frequencies in probability bins
2. **Calibration metrics**: Brier score, expected calibration error (ECE)
3. **Sharpness**: How spread out the predicted probabilities are

Example diagnostic code::

   from sklearn.calibration import calibration_curve

   # Before calibration
   fraction_of_positives, mean_predicted_value = calibration_curve(
       y_true, y_prob_uncalibrated, n_bins=10
   )

   # After calibration
   fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(
       y_true, y_prob_calibrated, n_bins=10
   )

When Calibration May Not Be Needed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Skip calibration if:

1. **Only rankings matter**: Using ROC-AUC or ranking-based evaluation
2. **Already well-calibrated**: Model probabilities are already reliable (check with diagnostics)
3. **Computational constraints**: Limited time/resources for the calibration step
4. **Threshold-only usage**: Only need binary predictions, never probability estimates

References
----------

.. [Platt1999] Platt, J. (1999). "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods". In *Advances in Large Margin Classifiers*.

.. [ZadroznyCElkan2002] Zadrozny, B. & Elkan, C. (2002). "Transforming classifier scores into accurate multiclass probability estimates". In *Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining*.
