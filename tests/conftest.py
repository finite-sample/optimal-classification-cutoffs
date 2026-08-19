"""Pytest configuration and shared fixtures for optimal_cutoffs tests.

This module provides pytest configuration, shared fixtures, and test utilities
that are available across all test modules.
"""

import warnings

import numpy as np
import pytest


def pytest_configure(config):
    """Configure pytest settings and custom markers."""
    # Register custom markers
    config.addinivalue_line("markers", "slow: mark test as slow running (>1 second)")
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "edge_case: mark test as an edge case test")
    config.addinivalue_line("markers", "validation: mark test as a validation test")
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "stress: mark test as a stress test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and handle slow tests."""
    # Add markers based on test location
    for item in items:
        # Add markers based on file path
        if "unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "edge_cases/" in str(item.fspath):
            item.add_marker(pytest.mark.edge_case)
        elif "validation/" in str(item.fspath):
            item.add_marker(pytest.mark.validation)
        elif "performance/" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "slow/" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.stress)

        # Add slow marker to tests that are explicitly marked or in slow directory
        if (
            item.get_closest_marker("slow") or "slow/" in str(item.fspath)
        ) and not hasattr(item, "_slow_marked"):
            item.add_marker(pytest.mark.slow)
            item._slow_marked = True


def pytest_runtest_setup(item):
    """Setup for individual test runs."""
    # Skip slow tests unless explicitly requested
    if item.get_closest_marker("slow") and not item.config.getoption(
        "--runslow", default=False
    ):
        pytest.skip("need --runslow option to run slow tests")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runstress", action="store_true", default=False, help="run stress tests"
    )


@pytest.fixture(scope="session")
def random_state():
    """Provide a consistent random state for reproducible tests."""
    return np.random.RandomState(42)


@pytest.fixture
def suppress_warnings():
    """Suppress common warnings during tests."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        yield


@pytest.fixture
def sample_binary_data(random_state):
    """Provide sample binary classification data."""
    n_samples = 100
    y_true = random_state.binomial(1, 0.5, n_samples)
    pred_prob = random_state.uniform(0, 1, n_samples)

    # Ensure both classes are present
    if np.sum(y_true) == 0:
        y_true[0] = 1
    elif np.sum(y_true) == n_samples:
        y_true[0] = 0

    return y_true, pred_prob


@pytest.fixture
def sample_multiclass_data(random_state):
    """Provide sample multiclass classification data."""
    n_samples = 100
    n_classes = 3

    # Generate labels ensuring all classes are present
    y_true = random_state.choice(n_classes, size=n_samples)
    for c in range(n_classes):
        if not np.any(y_true == c):
            y_true[random_state.randint(0, n_samples)] = c

    # Generate probability matrix
    pred_prob = random_state.dirichlet(np.ones(n_classes), size=n_samples)

    return y_true, pred_prob


@pytest.fixture
def sample_weights(random_state):
    """Provide sample weights for testing."""
    return random_state.uniform(0.5, 2.0, 100)


@pytest.fixture
def performance_timer():
    """Provide a timer for performance testing."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.perf_counter()

        def stop(self):
            self.end_time = time.perf_counter()

        def elapsed(self):
            if self.start_time is None or self.end_time is None:
                raise ValueError("Timer not properly started/stopped")
            return self.end_time - self.start_time

    return Timer()


@pytest.fixture
def memory_monitor():
    """Provide memory monitoring for performance tests."""
    try:
        import os

        import psutil

        class MemoryMonitor:
            def __init__(self):
                self.process = psutil.Process(os.getpid())
                self.baseline = None

            def start(self):
                self.baseline = self.process.memory_info().rss / 1024 / 1024  # MB

            def current_usage(self):
                return self.process.memory_info().rss / 1024 / 1024  # MB

            def peak_usage(self):
                if self.baseline is None:
                    raise ValueError("Monitor not started")
                return self.current_usage() - self.baseline

        return MemoryMonitor()

    except ImportError:
        pytest.skip("psutil not available for memory monitoring")


# Parametrized fixtures for common test scenarios
@pytest.fixture(params=["f1", "accuracy", "precision", "recall"])
def metric_name(request):
    """Parametrized fixture for different metrics."""
    return request.param


@pytest.fixture(params=[">", ">="])
def comparison_operator(request):
    """Parametrized fixture for comparison operators."""
    return request.param


@pytest.fixture(params=["unique_scan", "minimize", "gradient"])
def optimization_method(request):
    """Parametrized fixture for optimization methods."""
    return request.param


@pytest.fixture(params=[10, 100, 1000])
def dataset_size(request):
    """Parametrized fixture for different dataset sizes."""
    return request.param


# Utility functions available to all tests
def assert_valid_probability(prob):
    """Assert that a value is a valid probability."""
    assert isinstance(prob, int | float | np.number)
    assert 0.0 <= prob <= 1.0
    assert np.isfinite(prob)


def assert_valid_threshold(threshold):
    """Assert that a threshold is valid."""
    if isinstance(threshold, list | tuple | np.ndarray):
        for t in threshold:
            assert_valid_probability(t)
    else:
        assert_valid_probability(threshold)


def assert_confusion_matrix_valid(tp, tn, fp, fn, total_samples=None):
    """Assert that confusion matrix values are valid."""
    assert tp >= 0
    assert tn >= 0
    assert fp >= 0
    assert fn >= 0
    assert all(np.isfinite(x) for x in [tp, tn, fp, fn])

    if total_samples is not None:
        assert abs((tp + tn + fp + fn) - total_samples) < 1e-10


# Test data generators that can be reused
def generate_separable_data(n_samples=100, random_state=None):
    """Generate perfectly separable binary data."""
    rng = np.random.RandomState(random_state)

    # Half negative, half positive
    n_pos = n_samples // 2
    n_neg = n_samples - n_pos

    y_true = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
    pred_prob = np.concatenate(
        [
            rng.uniform(0.0, 0.4, n_neg),  # Negative class
            rng.uniform(0.6, 1.0, n_pos),  # Positive class
        ]
    )

    # Shuffle
    indices = rng.permutation(n_samples)
    return y_true[indices], pred_prob[indices]


def generate_noisy_data(n_samples=100, noise_level=0.1, random_state=None):
    """Generate binary data with controlled noise."""
    rng = np.random.RandomState(random_state)

    # Generate base probabilities
    pred_prob = rng.uniform(0, 1, n_samples)

    # Generate labels with noise
    true_prob = pred_prob + noise_level * rng.normal(0, 1, n_samples)
    y_true = (true_prob > 0.5).astype(int)

    # Ensure both classes present
    if np.sum(y_true) == 0:
        y_true[0] = 1
    elif np.sum(y_true) == n_samples:
        y_true[0] = 0

    return y_true, pred_prob


def generate_imbalanced_data(n_samples=1000, imbalance_ratio=0.1, random_state=None):
    """Generate imbalanced binary data."""
    rng = np.random.RandomState(random_state)

    n_positive = max(1, int(n_samples * imbalance_ratio))
    n_negative = n_samples - n_positive

    y_true = np.concatenate([np.zeros(n_negative), np.ones(n_positive)])

    # Generate probabilities slightly favoring correct class
    neg_probs = rng.uniform(0.0, 0.6, n_negative)
    pos_probs = rng.uniform(0.4, 1.0, n_positive)
    pred_prob = np.concatenate([neg_probs, pos_probs])

    # Shuffle
    indices = rng.permutation(n_samples)
    return y_true[indices], pred_prob[indices]
