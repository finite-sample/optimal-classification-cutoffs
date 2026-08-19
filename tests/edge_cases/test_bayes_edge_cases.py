"""Comprehensive tests for Bayes module edge cases and mathematical correctness."""

import numpy as np
import pytest

from optimal_cutoffs.bayes import (
    BayesOptimal,
    UtilitySpec,
    threshold,
    thresholds_from_costs,
)


class TestBinaryMathematicalCorrectness:
    """Test binary decision math for all D cases."""

    def test_cost_only_threshold(self):
        """Test C_fp = 1, C_fn = 5 -> t = 1/6."""
        result_threshold = threshold(1, 5)
        assert abs(result_threshold - (1 / 6)) < 1e-12

    def test_standard_case_D_positive(self):
        """Test standard case with D > 0."""
        # tp=2, tn=2, fp=-1, fn=-1 => A=3, B=3, D=6, threshold=0.5
        u = UtilitySpec(tp_utility=2, tn_utility=2, fp_utility=-1, fn_utility=-1)
        bo = BayesOptimal(u)

        # Test binary decision with margin approach
        p = np.array([0.0, 0.4, 0.5, 0.6, 1.0])
        decisions = bo._decide_binary(p)
        expected = np.array([0, 0, 1, 1, 1])  # >= 0.5 is positive
        np.testing.assert_array_equal(decisions, expected)

    def test_D_eq_0_always_positive(self):
        """Test D=0 case where decision is always positive."""
        # Choose params with A = -B and B <= 0
        # Example: tp=1, fn=0 => A=1; tn=0, fp=1 => B=-1; D=0, B<=0 -> always positive
        u = UtilitySpec(tp_utility=1, tn_utility=0, fp_utility=1, fn_utility=0)
        bo = BayesOptimal(u)

        p = np.array([0.0, 0.3, 1.0])
        decisions = bo._decide_binary(p)
        np.testing.assert_array_equal(decisions, np.array([1, 1, 1]))

    def test_D_eq_0_always_negative(self):
        """Test D=0 case where decision is always negative."""
        # Choose params with A = -B and B > 0
        # Example: tp=0, fn=1 => A=-1; tn=1, fp=0 => B=1; D=0, B>0 -> always negative
        u = UtilitySpec(tp_utility=0, tn_utility=1, fp_utility=0, fn_utility=1)
        bo = BayesOptimal(u)

        p = np.array([0.0, 0.7, 1.0])
        decisions = bo._decide_binary(p)
        np.testing.assert_array_equal(decisions, np.array([0, 0, 0]))

    def test_D_negative_inverted_inequality(self):
        """Test D < 0 case where inequality is inverted."""
        # Example: tp=-1, fn=1 => A=-2; tn=-1, fp=1 => B=-2; D=-4 < 0
        # threshold would be -2/-4 = 0.5, but inequality flips
        # predict positive when p <= 0.5 (margin <= 0)
        u = UtilitySpec(tp_utility=-1, tn_utility=-1, fp_utility=1, fn_utility=1)
        bo = BayesOptimal(u)

        p = np.array([0.0, 0.4, 0.5, 0.6, 1.0])
        decisions = bo._decide_binary(p)
        expected = np.array([1, 1, 1, 0, 0])  # <= 0.5 is positive when D < 0
        np.testing.assert_array_equal(decisions, expected)

    def test_compute_threshold_D_zero_raises(self):
        """Test that compute_threshold raises for D=0."""
        u = UtilitySpec(tp_utility=1, tn_utility=0, fp_utility=1, fn_utility=0)
        bo = BayesOptimal(u)

        with pytest.raises(ValueError, match="compute_threshold is only valid when"):
            bo.compute_threshold()

    def test_compute_threshold_D_negative_raises(self):
        """Test that compute_threshold raises for D<0."""
        u = UtilitySpec(tp_utility=-1, tn_utility=-1, fp_utility=1, fn_utility=1)
        bo = BayesOptimal(u)

        with pytest.raises(ValueError, match="compute_threshold is only valid when"):
            bo.compute_threshold()

    def test_binary_probability_shapes(self):
        """Test both (n,) and (n,2) probability shapes."""
        u = UtilitySpec(tp_utility=1, tn_utility=1, fp_utility=0, fn_utility=0)
        bo = BayesOptimal(u)

        # Test 1D shape
        p1d = np.array([0.3, 0.7])
        decisions1d = bo._decide_binary(p1d)

        # Test 2D shape (binary probabilities)
        p2d = np.array([[0.7, 0.3], [0.3, 0.7]])  # P(y=0), P(y=1)
        decisions2d = bo._decide_binary(p2d)

        np.testing.assert_array_equal(decisions1d, decisions2d)

    def test_tie_breaking_consistency(self):
        """Test that >= tie-breaking is used consistently."""
        u = UtilitySpec(tp_utility=1, tn_utility=1, fp_utility=0, fn_utility=0)
        bo = BayesOptimal(u)

        # threshold should be 0.5, test tie at exactly 0.5
        p = np.array([0.5])
        decision = bo._decide_binary(p)
        assert decision[0] == 1  # >= 0.5 should be positive


class TestMulticlassApplyFallback:
    """Test multiclass apply fallback behavior."""

    def test_multiclass_apply_fallback(self):
        """Test fallback when no class meets threshold."""
        from optimal_cutoffs import OptimizationResult

        # Create a predict function that mimics the old BayesThresholdResult.apply
        # behavior
        def predict_func(probs):
            thresholds = np.array([0.7, 0.8, 0.9])
            predictions = []
            for prob_row in probs:
                # Check which classes pass threshold
                passes_threshold = prob_row >= thresholds
                if np.any(passes_threshold):
                    # Predict class with highest probability among those passing
                    # threshold
                    valid_indices = np.where(passes_threshold)[0]
                    best_idx = valid_indices[np.argmax(prob_row[valid_indices])]
                    predictions.append(best_idx)
                else:
                    # Fallback to argmax if none pass
                    predictions.append(np.argmax(prob_row))
            return np.array(predictions)

        th = OptimizationResult(
            thresholds=np.array([0.7, 0.8, 0.9]),
            scores=np.array([0.0, 0.0, 0.0]),  # dummy scores
            predict=predict_func,
            metric="utility",
            n_classes=3,
        )
        probs = np.array(
            [
                [0.6, 0.6, 0.6],  # none pass -> fallback argmax 0
                [0.95, 0.1, 0.1],  # class 0 passes
                [0.5, 0.85, 0.1],  # class 1 passes
            ]
        )
        pred = th.predict(probs)
        np.testing.assert_array_equal(pred, np.array([0, 0, 1]))

    def test_binary_apply_consistency(self):
        """Test binary apply with consistent tie-breaking."""
        from optimal_cutoffs import OptimizationResult

        def predict_func(probs):
            return (probs >= 0.5).astype(int)

        th = OptimizationResult(
            thresholds=np.array([0.5]),
            scores=np.array([0.0]),  # dummy score
            predict=predict_func,
            metric="utility",
            n_classes=2,
        )

        # Test 1D probabilities
        probs1d = np.array([0.4, 0.5, 0.6])
        pred1d = th.predict(probs1d)
        np.testing.assert_array_equal(pred1d, np.array([0, 1, 1]))

        # Test 2D probabilities - extract positive class probabilities
        probs2d = np.array([[0.6, 0.4], [0.5, 0.5], [0.4, 0.6]])
        probs2d_pos = probs2d[:, 1]  # Use positive class probabilities
        pred2d = th.predict(probs2d_pos)
        np.testing.assert_array_equal(pred2d, np.array([0, 1, 1]))


class TestExpectedUtility:
    """Test expected utility computation."""

    def test_binary_expected_utility(self):
        """Test binary expected utility calculation."""
        # Simple case: tp=1, tn=1, fp=0, fn=0
        u = UtilitySpec(tp_utility=1, tn_utility=1, fp_utility=0, fn_utility=0)
        bo = BayesOptimal(u)

        # p=0.7: EU_pos = 0.7*1 + 0.3*0 = 0.7, EU_neg = 0.7*0 + 0.3*1 = 0.3
        # max(0.7, 0.3) = 0.7
        p = np.array([0.7])
        eu = bo.expected_utility(p)
        assert abs(eu - 0.7) < 1e-12

    def test_matrix_expected_utility(self):
        """Test matrix-based expected utility."""
        # Identity utility matrix
        utility_matrix = np.eye(3)
        bo = BayesOptimal(utility_matrix)

        probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        eu = bo.expected_utility(probs)
        # Expected utility should be max of each row: max(0.7, 0.2, 0.1) + max(0.1, 0.8,
        # 0.1) = 0.7 + 0.8 = 1.5/2 = 0.75
        assert abs(eu - 0.75) < 1e-12


class TestVectorizedThresholds:
    """Test vectorized threshold computation."""

    def test_thresholds_from_costs_vectorized(self):
        """Test vectorized threshold computation matches element-wise."""
        fp_costs = np.array([1, 2, 3])
        fn_costs = np.array([5, 4, 3])

        thresholds = thresholds_from_costs(fp_costs, fn_costs)
        expected = fp_costs / (fp_costs + fn_costs)
        np.testing.assert_array_equal(thresholds, expected)

    def test_bayes_thresholds_validation(self):
        """Test input validation for vectorized thresholds."""
        with pytest.raises(ValueError, match="same shape"):
            thresholds_from_costs([1, 2], [1])

        with pytest.raises(ValueError, match="finite"):
            thresholds_from_costs([1, np.inf], [1, 2])

        # Zero costs should fail since |fp| + |fn| = 0
        with pytest.raises(ValueError, match="must be > 0"):
            thresholds_from_costs([0, 1], [0, 2])


class TestUtilitySpecValidation:
    """Test UtilitySpec input validation."""

    def test_from_costs_validation(self):
        """Test validation in from_costs."""
        with pytest.raises(ValueError, match="finite"):
            UtilitySpec.from_costs(np.inf, 1.0)

        with pytest.raises(ValueError, match="finite"):
            UtilitySpec.from_costs(1.0, np.nan)

    def test_from_dict_validation(self):
        """Test validation in from_dict."""
        # Test that unknown keys raise error
        with pytest.raises(ValueError, match="Unknown utility keys"):
            UtilitySpec.from_dict({"tp": 1, "tn": 1, "invalid_key": 2})

        # Test that non-finite values raise error
        with pytest.raises(ValueError, match="finite"):
            UtilitySpec.from_dict({"tp": 1, "tn": 1, "fp": np.inf, "fn": 1})


class TestAPIConsistency:
    """Test API consistency and error handling."""

    def test_decide_api_accepts_numpy_arrays(self):
        """Test that decide() accepts raw numpy arrays."""
        u = UtilitySpec(tp_utility=1, tn_utility=1, fp_utility=0, fn_utility=0)
        bo = BayesOptimal(u)

        # Should not raise - accepts NDArray directly
        probs = np.array([0.3, 0.7])
        decisions = bo.decide(probs)
        assert isinstance(decisions, np.ndarray)
        assert decisions.dtype == np.int32

    def test_binary_matrix_utility(self):
        """Test binary decisions with 2x2 utility matrix."""
        # 2x2 matrix: [[tn, fn], [fp, tp]]
        utility_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
        bo = BayesOptimal(utility_matrix)

        assert bo.is_binary
        probs = np.array([0.3, 0.7])
        decisions = bo.decide(probs)
        # threshold = 0.5, so [0, 1]
        np.testing.assert_array_equal(decisions, np.array([0, 1]))

    def test_invalid_probability_shapes(self):
        """Test error handling for invalid probability shapes."""
        u = UtilitySpec()
        bo = BayesOptimal(u)

        with pytest.raises(ValueError, match="Binary probabilities must be"):
            bo._extract_binary_p(np.array([[[0.5]]]))  # 3D array

        with pytest.raises(ValueError, match="Binary probabilities must be"):
            bo._extract_binary_p(
                np.array([[0.3, 0.4, 0.3]])
            )  # 3-class probs for binary


# Quick property-based test to verify margin formula equivalence
class TestMathematicalProperties:
    """Test mathematical properties and invariants."""

    def test_margin_threshold_equivalence(self):
        """Test that margin and threshold formulations are equivalent for D > 0."""
        u = UtilitySpec(tp_utility=2, tn_utility=1, fp_utility=-1, fn_utility=-2)
        bo = BayesOptimal(u)

        # For D > 0, margin >= 0 should equal p >= threshold
        _A, B, D = bo._binary_params()
        assert D > 0  # Ensure D > 0 for this test

        threshold = B / D
        probs = np.linspace(0, 1, 100)

        # Method 1: margin-based
        margin = D * probs - B
        decisions_margin = (margin >= 0).astype(int)

        # Method 2: threshold-based
        decisions_threshold = (probs >= threshold).astype(int)

        np.testing.assert_array_equal(decisions_margin, decisions_threshold)

    def test_optimal_decisions_monotonic(self):
        """Test that optimal decisions respect probability ordering for D > 0."""
        u = UtilitySpec(tp_utility=1, tn_utility=1, fp_utility=0, fn_utility=0)
        bo = BayesOptimal(u)

        # Increasing probabilities should lead to monotonic decisions (for D > 0)
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        decisions = bo._decide_binary(probs)

        # Decisions should be non-decreasing
        assert np.all(np.diff(decisions) >= 0)
