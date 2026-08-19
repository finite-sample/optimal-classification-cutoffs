"""Tests for API 2.0.0 specific features.

This module tests the new features introduced in API 2.0.0:
- Task and Average enums
- Explainable auto-selection
- Clean namespace organization
- Match/case routing performance
"""

import numpy as np
import pytest

from optimal_cutoffs import (
    Average,
    Task,
    optimize_decisions,
    optimize_thresholds,
)


class TestEnums:
    """Test Task and Average enum functionality."""

    def test_task_enum_values(self):
        """Test Task enum has correct values."""
        assert Task.AUTO.value == "auto"
        assert Task.BINARY.value == "binary"
        assert Task.MULTICLASS.value == "multiclass"
        assert Task.MULTILABEL.value == "multilabel"

    def test_average_enum_values(self):
        """Test Average enum has correct values."""
        assert Average.AUTO.value == "auto"
        assert Average.MACRO.value == "macro"
        assert Average.MICRO.value == "micro"
        assert Average.WEIGHTED.value == "weighted"
        assert Average.NONE.value == "none"

    def test_task_enum_in_result(self):
        """Test that results contain detected task enum."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8, 0.9])

        result = optimize_thresholds(y_true, y_score, task=Task.AUTO)
        assert isinstance(result.task, Task)
        assert result.task == Task.BINARY

    def test_average_enum_in_result(self):
        """Test that results contain selected average enum."""
        y_true = np.array([0, 1, 2, 1, 0])
        y_score = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.2, 0.7, 0.1],
                [0.9, 0.05, 0.05],
            ]
        )

        result = optimize_thresholds(
            y_true, y_score, task=Task.MULTICLASS, average=Average.AUTO
        )
        assert isinstance(result.average, Average)
        assert result.average == Average.MACRO


class TestExplainableAutoSelection:
    """Test explainable auto-selection features."""

    def test_auto_selection_notes(self):
        """Test that auto-selection provides explanatory notes."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8, 0.9])

        result = optimize_thresholds(y_true, y_score, task=Task.AUTO, method="auto")

        # Should have notes explaining decisions
        assert hasattr(result, "notes")
        assert isinstance(result.notes, list)
        assert len(result.notes) > 0

        # Check for expected explanatory content
        notes_text = " ".join(result.notes)
        assert any(
            word in notes_text.lower() for word in ["binary", "detected", "auto"]
        )

    def test_method_selection_explanation(self):
        """Test method auto-selection provides explanation."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8, 0.9])

        # F1 should trigger sort_scan method
        result = optimize_thresholds(y_true, y_score, method="auto", metric="f1")
        assert result.method == "sort_scan"
        assert any("O(n log n)" in note for note in result.notes)

        # Accuracy is also piecewise-constant with a vectorized implementation, so it
        # gets the exact O(n log n) optimizer too.
        result = optimize_thresholds(y_true, y_score, method="auto", metric="accuracy")
        assert result.method == "sort_scan"
        assert any("O(n log n)" in note for note in result.notes)

        # A metric registered without a piecewise/vectorized implementation falls back
        # to scipy minimisation.
        from optimal_cutoffs import metrics

        metrics.register(
            "_nonpiecewise_probe", lambda tp, tn, fp, fn: tp / (tp + fp + fn + 1e-9)
        )
        result = optimize_thresholds(
            y_true, y_score, method="auto", metric="_nonpiecewise_probe"
        )
        assert result.method == "minimize"
        assert any("scipy" in note for note in result.notes)

    def test_multiclass_method_explanation(self):
        """Test multiclass method selection explanation."""
        y_true = np.array([0, 1, 2, 1, 0])
        y_score = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.2, 0.7, 0.1],
                [0.9, 0.05, 0.05],
            ]
        )

        result = optimize_thresholds(
            y_true, y_score, task=Task.AUTO, method="auto", metric="f1"
        )

        # Should explain multiclass detection and method choice
        notes_text = " ".join(result.notes)
        assert "multiclass" in notes_text.lower()
        assert (
            "coordinate ascent" in notes_text.lower()
            or "coord_ascent" in notes_text.lower()
        )

    def test_warnings_field(self):
        """Test that warnings are captured when relevant."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8, 0.9])

        result = optimize_thresholds(y_true, y_score)

        # Should have warnings attribute
        assert hasattr(result, "warnings")
        assert isinstance(result.warnings, list)


class TestNamespaceOrganization:
    """Test clean namespace organization."""

    def test_metrics_namespace(self):
        """Test metrics namespace functionality."""
        import optimal_cutoffs

        # Should be able to access metrics namespace
        assert hasattr(optimal_cutoffs, "metrics")

        # Test key functions
        assert hasattr(optimal_cutoffs.metrics, "get")
        assert hasattr(optimal_cutoffs.metrics, "list_available")
        assert hasattr(optimal_cutoffs.metrics, "register")

        # Test it works
        f1_func = optimal_cutoffs.metrics.get("f1")
        assert callable(f1_func)

        available = optimal_cutoffs.metrics.list_available()
        assert "f1" in available
        assert "precision" in available
        assert "recall" in available

    def test_bayes_namespace(self):
        """Test bayes namespace functionality."""
        import optimal_cutoffs

        # Should be able to access bayes namespace
        assert hasattr(optimal_cutoffs, "bayes")
        assert hasattr(optimal_cutoffs.bayes, "threshold")

        # Test it works
        threshold = optimal_cutoffs.bayes.threshold(cost_fp=1.0, cost_fn=5.0)
        assert isinstance(threshold, float)
        assert 0.0 < threshold < 0.5  # Should be < 0.5 since FN costs more

    def test_cv_namespace(self):
        """Test cv namespace functionality."""
        import optimal_cutoffs

        # Should be able to access cv namespace
        assert hasattr(optimal_cutoffs, "cv")
        assert hasattr(optimal_cutoffs.cv, "cross_validate")

    def test_algorithms_namespace(self):
        """Test algorithms namespace functionality."""
        import optimal_cutoffs

        # Should be able to access algorithms namespace
        assert hasattr(optimal_cutoffs, "algorithms")
        assert hasattr(optimal_cutoffs.algorithms, "binary")
        assert hasattr(optimal_cutoffs.algorithms, "multiclass")
        assert hasattr(optimal_cutoffs.algorithms, "multilabel")


class TestOptimizationResult:
    """Test enhanced OptimizationResult functionality."""

    def test_result_has_required_fields(self):
        """Test that results have all required fields."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8, 0.9])

        result = optimize_thresholds(y_true, y_score)

        # Core fields
        assert hasattr(result, "thresholds")
        assert hasattr(result, "scores")
        assert hasattr(result, "predict")

        # Explainability fields
        assert hasattr(result, "task")
        assert hasattr(result, "method")
        assert hasattr(result, "average")
        assert hasattr(result, "notes")
        assert hasattr(result, "warnings")
        assert hasattr(result, "metric")

    def test_binary_threshold_property(self):
        """Test binary threshold property works."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8, 0.9])

        result = optimize_thresholds(y_true, y_score)

        # Should work for binary
        assert hasattr(result, "threshold")
        threshold = result.threshold
        assert isinstance(threshold, float)
        assert threshold == result.thresholds[0]

    def test_multiclass_threshold_property_raises(self):
        """Test that .threshold raises for multiclass."""
        y_true = np.array([0, 1, 2, 1, 0])
        y_score = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.2, 0.7, 0.1],
                [0.9, 0.05, 0.05],
            ]
        )

        result = optimize_thresholds(y_true, y_score)

        # Should raise for multiclass
        with pytest.raises(ValueError, match=r"Use \.thresholds"):
            _ = result.threshold

    def test_predict_function_works(self):
        """Test that predict function works."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8, 0.9])

        result = optimize_thresholds(y_true, y_score)

        # Should be able to predict
        predictions = result.predict(y_score)
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == y_score.shape
        assert np.all(np.isin(predictions, [0, 1]))


class TestAPI20ExportCount:
    """Test that API 2.0.0 has exactly 8 exports as designed."""

    def test_clean_api_export_count(self):
        """Test that __all__ defines the clean API."""
        import optimal_cutoffs

        # Test that __all__ has exactly 10 items (our clean API)
        expected_all = {
            "__version__",
            "optimize_thresholds",
            "optimize_decisions",
            "OptimizationResult",
            "Task",
            "Average",
            "metrics",
            "bayes",
            "cv",
            "algorithms",
        }

        assert set(optimal_cutoffs.__all__) == expected_all, (
            f"Expected 10 __all__ exports, got {len(optimal_cutoffs.__all__)}: "
            f"{optimal_cutoffs.__all__}"
        )

        # Test that star import only gets the clean API
        # (This is what users will see with "from optimal_cutoffs import *")
        star_imports = set(optimal_cutoffs.__all__)
        assert len(star_imports) == 10, "Star import should only expose 10 items"

    def test_no_backward_compatibility(self):
        """Test that old API functions are not available."""
        import optimal_cutoffs

        # These old functions should NOT be available
        old_functions = [
            "get_optimal_threshold",
            "cv_threshold_optimization",
            "optimize_f1_binary",
            "optimize_macro_multilabel",
            "optimize_micro_multiclass",
        ]

        for func_name in old_functions:
            assert not hasattr(optimal_cutoffs, func_name), (
                f"Old function {func_name} should not be available in API 2.0.0"
            )


class TestOptimizeDecisions:
    """Test the optimize_decisions function."""

    def test_optimize_decisions_basic(self):
        """Test basic optimize_decisions functionality."""
        # Simple 2x2 cost matrix
        y_prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
        cost_matrix = np.array([[0, 1], [5, 0]])  # FN costs 5x more than FP

        result = optimize_decisions(y_prob, cost_matrix)

        # Should have predict function
        assert hasattr(result, "predict")
        assert callable(result.predict)

        # Should be able to make predictions
        decisions = result.predict(y_prob)
        assert isinstance(decisions, np.ndarray)
        assert decisions.shape == (3,)
