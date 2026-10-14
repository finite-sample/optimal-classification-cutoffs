Installation
============

Requirements
------------

* Python 3.12 or higher
* NumPy >= 1.20.0
* SciPy >= 1.5.0
* scikit-learn (for some metrics and utilities)

Install from PyPI
-----------------

The easiest way to install optimal-classification-cutoffs is using pip:

.. code-block:: bash

   pip install optimal-classification-cutoffs

Install from Source
-------------------

For the latest development version, you can install directly from the GitHub repository:

.. code-block:: bash

   pip install git+https://github.com/finite-sample/optimal-classification-cutoffs.git

Development Installation
------------------------

For development purposes, clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/finite-sample/optimal-classification-cutoffs.git
   cd optimal-classification-cutoffs
   pip install -e .

The examples are scripts in ``docs/examples``. Run one with:

.. code-block:: bash

   uv run --group docs python docs/examples/plot_binary_threshold.py

Verification
------------

To verify your installation, run:

.. code-block:: python

   import optimal_cutoffs
   print(optimal_cutoffs.__version__)

   # Quick test
   import numpy as np
   from optimal_cutoffs import optimize_thresholds

   y_true = np.array([0, 0, 1, 1])
   y_prob = np.array([0.1, 0.4, 0.35, 0.8])
   result = optimize_thresholds(y_true, y_prob, metric='f1')
   print(f"Installation successful! Optimal threshold: {result.thresholds[0]}")

Troubleshooting
---------------

**Import Errors**
   If you encounter import errors, ensure all dependencies are installed:

   .. code-block:: bash

      pip install numpy>=1.20.0 scipy>=1.5.0 scikit-learn

**Version Conflicts**
   For clean environment setup, consider using virtual environments:

   .. code-block:: bash

      python -m venv optimal_cutoffs_env
      source optimal_cutoffs_env/bin/activate  # On Windows: optimal_cutoffs_env\Scripts\activate
      pip install optimal-classification-cutoffs
