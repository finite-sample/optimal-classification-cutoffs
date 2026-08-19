"""Centralized Numba import utilities for JIT compilation support.

This module provides a single location for Numba imports and fallback logic,
ensuring consistent behavior across all modules that use JIT compilation.
"""

from typing import Any

try:
    from numba import float64, int32, jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Define dummy decorators for when numba is not available
    def jit(*args, **kwargs):
        """Stand in for `numba.jit`, returning the function unchanged."""

        def decorator(func):
            return func

        return decorator

    def prange(*args, **kwargs):
        """Stand in for `numba.prange`, which is `range` without numba."""
        return range(*args, **kwargs)

    float64 = float
    int32 = int


def numba_with_fallback(**numba_kwargs: Any):
    """Decorator that auto-generates Python fallback from Numba code.

    When Numba is available: applies JIT compilation with specified kwargs
    When Numba unavailable: returns original function (Numba code is valid Python!)

    Args:
        **numba_kwargs: Keyword arguments to pass to numba.jit (e.g., nopython=True,
            cache=True)

    Returns:
        Decorator function that applies JIT or returns original function

    Examples:
        >>> @numba_with_fallback(nopython=True, cache=True)
        ... def fast_function(x, y):
        ...     return x + y
    """

    def decorator(func):
        if NUMBA_AVAILABLE:
            return jit(**numba_kwargs)(func)
        # Return original function - Numba code is valid Python!
        return func

    return decorator


__all__ = [
    "NUMBA_AVAILABLE",
    "float64",
    "int32",
    "jit",
    "numba_with_fallback",
    "prange",
]
