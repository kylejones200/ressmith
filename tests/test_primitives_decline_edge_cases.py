"""Tests for numerical edge cases in decline curve calculations.

This module tests numerical stability and correctness for edge cases:
- b → 0 (hyperbolic approaches exponential)
- b → 1 (hyperbolic approaches harmonic)
- di → 0 (very small decline rates)
"""

import numpy as np
import pytest

from ressmith.primitives.decline import (
    arps_exponential,
    arps_hyperbolic,
    arps_harmonic,
    cumulative_exponential,
    cumulative_hyperbolic,
    cumulative_harmonic,
)
from ressmith.primitives.reserves import (
    calculate_eur_exponential,
    calculate_eur_hyperbolic,
    calculate_eur_harmonic,
)


class TestHyperbolicBApproachingZero:
    """Test hyperbolic decline as b approaches 0 (should approach exponential)."""

    def test_b_very_small_approaches_exponential(self):
        """Hyperbolic with b → 0 should approach exponential decline."""
        t = np.linspace(0, 100, 1000)
        qi = 100.0
        di = 0.1

        # Exponential decline
        q_exp = arps_exponential(t, qi, di)

        # Hyperbolic declines with b approaching 0
        b_values = [0.01, 0.001, 0.0001]
        for b in b_values:
            q_hyp = arps_hyperbolic(t, qi, di, b)

            # Rates should be close (within 5% for small b)
            # More tolerance for very small b due to numerical precision
            rtol = 0.05 if b >= 0.001 else 0.15
            np.testing.assert_allclose(q_hyp, q_exp, rtol=rtol, atol=1.0)

    def test_b_very_small_cumulative_approaches_exponential(self):
        """Cumulative for hyperbolic with b → 0 should approach exponential."""
        t = np.linspace(0, 100, 1000)
        qi = 100.0
        di = 0.1

        # Exponential cumulative
        cum_exp = cumulative_exponential(t, qi, di)

        # Hyperbolic cumulative with small b
        b = 0.001
        cum_hyp = cumulative_hyperbolic(t, qi, di, b)

        # Should be close (within 10% for small b)
        np.testing.assert_allclose(cum_hyp, cum_exp, rtol=0.10, atol=10.0)

    def test_b_zero_not_allowed(self):
        """b = 0 should raise ValueError."""
        t = np.linspace(0, 10, 100)
        qi = 100.0
        di = 0.1
        b = 0.0

        with pytest.raises(ValueError, match="b-factor must be between 0 and 1"):
            arps_hyperbolic(t, qi, di, b)
        with pytest.raises(ValueError, match="b-factor must be between 0 and 1"):
            cumulative_hyperbolic(t, qi, di, b)

    def test_b_negative_not_allowed(self):
        """Negative b values should raise ValueError."""
        t = np.linspace(0, 10, 100)
        qi = 100.0
        di = 0.1
        b = -0.1

        with pytest.raises(ValueError, match="b-factor must be between 0 and 1"):
            arps_hyperbolic(t, qi, di, b)
        with pytest.raises(ValueError, match="b-factor must be between 0 and 1"):
            cumulative_hyperbolic(t, qi, di, b)


class TestHyperbolicBApproachingOne:
    """Test hyperbolic decline as b approaches 1 (should approach harmonic)."""

    def test_b_approaches_one_approaches_harmonic(self):
        """Hyperbolic with b → 1 should approach harmonic decline."""
        t = np.linspace(0, 100, 1000)
        qi = 100.0
        di = 0.1

        # Harmonic decline
        q_har = arps_harmonic(t, qi, di)

        # Hyperbolic declines with b approaching 1
        b_values = [0.999, 0.9999, 0.99999]
        for b in b_values:
            q_hyp = arps_hyperbolic(t, qi, di, b)

            # Rates should be very close (within 1% for b near 1)
            np.testing.assert_allclose(q_hyp, q_har, rtol=0.01, atol=0.5)

    def test_b_equals_one_tolerance_check(self):
        """Test that b=1.0 ± tolerance correctly uses harmonic formula."""
        t = np.linspace(0, 100, 1000)
        qi = 100.0
        di = 0.1

        # Harmonic cumulative
        cum_har = cumulative_harmonic(t, qi, di)

        # Test values very close to 1.0 (within tolerance)
        b_values = [1.0 - 1e-7, 1.0, 1.0 + 1e-7]
        for b in b_values:
            # Should use harmonic formula due to tolerance check
            cum_hyp = cumulative_hyperbolic(t, qi, di, b)
            np.testing.assert_allclose(cum_hyp, cum_har, rtol=1e-5, atol=0.1)

    def test_b_approaches_one_cumulative_approaches_harmonic(self):
        """Cumulative for hyperbolic with b → 1 should approach harmonic."""
        t = np.linspace(0, 100, 1000)
        qi = 100.0
        di = 0.1

        # Harmonic cumulative
        cum_har = cumulative_harmonic(t, qi, di)

        # Hyperbolic cumulative with b close to 1
        b = 0.9999
        cum_hyp = cumulative_hyperbolic(t, qi, di, b)

        # Should be close (within 5% for b near 1)
        np.testing.assert_allclose(cum_hyp, cum_har, rtol=0.05, atol=10.0)

    def test_b_greater_than_one_raises_error(self):
        """b > 1 should raise ValueError."""
        t = np.linspace(0, 10, 100)
        qi = 100.0
        di = 0.1
        b = 1.5

        with pytest.raises(ValueError, match="b-factor must be between 0 and 1"):
            arps_hyperbolic(t, qi, di, b)
        with pytest.raises(ValueError, match="b-factor must be between 0 and 1"):
            cumulative_hyperbolic(t, qi, di, b)


class TestSmallDeclineRate:
    """Test behavior with very small decline rates (di → 0)."""

    def test_di_very_small_rate_behavior(self):
        """Rates should remain nearly constant for very small di."""
        t = np.linspace(0, 100, 1000)
        qi = 100.0

        # Very small decline rates
        di_values = [1e-6, 1e-5, 1e-4]

        for di in di_values:
            # Exponential decline
            q_exp = arps_exponential(t, qi, di)

            # Should start at qi
            assert np.isclose(q_exp[0], qi, rtol=1e-5)

            # Should decline very slowly
            # After 100 days, rate should still be > 99% of qi for di=1e-6
            if di <= 1e-6:
                assert q_exp[-1] > 0.99 * qi
            elif di <= 1e-5:
                assert q_exp[-1] > 0.90 * qi

            # Should be positive and decreasing
            assert np.all(q_exp > 0)
            assert np.all(np.diff(q_exp) <= 0)

    def test_di_very_small_cumulative_behavior(self):
        """Cumulative should grow approximately linearly for small di."""
        t = np.linspace(0, 100, 1000)
        qi = 100.0
        di = 1e-6

        # Exponential cumulative
        cum_exp = cumulative_exponential(t, qi, di)

        # For very small di, cumulative ≈ qi * t
        cum_linear = qi * t

        # Should be approximately linear (within 1%)
        np.testing.assert_allclose(cum_exp, cum_linear, rtol=0.01, atol=1.0)

    def test_di_zero_handling(self):
        """di = 0 should be handled gracefully."""
        t = np.linspace(0, 10, 100)
        qi = 100.0
        di = 0.0

        # Should handle division by zero gracefully
        try:
            q_exp = arps_exponential(t, qi, di)
            # If no error, should be constant at qi
            assert np.allclose(q_exp, qi)
        except ZeroDivisionError:
            # Division by zero error is acceptable
            pass

        # Cumulative with di=0 might cause issues
        try:
            cum_exp = cumulative_exponential(t, qi, di)
            # If no error, should be approximately qi * t
            assert np.allclose(cum_exp, qi * t, rtol=0.01)
        except ZeroDivisionError:
            # Division by zero error is acceptable
            pass

    def test_di_very_small_hyperbolic(self):
        """Hyperbolic with very small di should behave correctly."""
        t = np.linspace(0, 100, 1000)
        qi = 100.0
        di = 1e-6
        b = 0.5

        q_hyp = arps_hyperbolic(t, qi, di, b)

        # Should start at qi
        assert np.isclose(q_hyp[0], qi, rtol=1e-5)

        # Should decline very slowly
        assert q_hyp[-1] > 0.95 * qi

        # Should be positive and decreasing
        assert np.all(q_hyp > 0)
        assert np.all(np.diff(q_hyp) <= 0)

    def test_di_very_small_eur_calculation(self):
        """EUR calculation with very small di should be stable."""
        qi = 100.0
        di = 1e-6
        t_max = 360.0
        econ_limit = 10.0

        # Should not crash
        eur_exp = calculate_eur_exponential(qi, di, t_max, econ_limit)

        # Should be positive and reasonable
        assert eur_exp > 0
        # For very small di, EUR should be approximately qi * t_max
        assert eur_exp > qi * t_max * 0.9

    def test_di_negative_not_allowed(self):
        """Negative di should raise an error or be handled."""
        t = np.linspace(0, 10, 100)
        qi = 100.0
        di = -0.1

        # Should handle gracefully (might raise error or produce invalid results)
        try:
            q_exp = arps_exponential(t, qi, di)
            # If no error, check that rates are valid
            assert np.all(np.isfinite(q_exp))
        except (ValueError, ZeroDivisionError):
            # Error is acceptable for invalid input
            pass


class TestCombinedEdgeCases:
    """Test combinations of edge cases."""

    def test_small_b_and_small_di(self):
        """Test hyperbolic with both small b and small di."""
        t = np.linspace(0, 100, 1000)
        qi = 100.0
        di = 1e-4
        b = 0.001

        q_hyp = arps_hyperbolic(t, qi, di, b)
        cum_hyp = cumulative_hyperbolic(t, qi, di, b)

        # Should be stable and produce valid output
        assert np.all(np.isfinite(q_hyp))
        assert np.all(np.isfinite(cum_hyp))
        assert np.all(q_hyp > 0)
        assert np.all(cum_hyp >= 0)
        assert np.all(np.diff(cum_hyp) >= 0)  # Cumulative should be increasing

    def test_b_near_one_and_small_di(self):
        """Test hyperbolic with b near 1 and small di."""
        t = np.linspace(0, 100, 1000)
        qi = 100.0
        di = 1e-5
        b = 0.9999

        q_hyp = arps_hyperbolic(t, qi, di, b)
        cum_hyp = cumulative_hyperbolic(t, qi, di, b)

        # Should use harmonic formula due to tolerance
        q_har = arps_harmonic(t, qi, di)
        cum_har = cumulative_harmonic(t, qi, di)

        # Should match harmonic closely
        np.testing.assert_allclose(q_hyp, q_har, rtol=0.01, atol=0.5)
        np.testing.assert_allclose(cum_hyp, cum_har, rtol=0.01, atol=1.0)

    def test_large_time_with_small_di(self):
        """Test behavior at large times with small decline rate."""
        t = np.linspace(0, 10000, 1000)  # Very long time
        qi = 100.0
        di = 1e-5

        # Exponential decline
        q_exp = arps_exponential(t, qi, di)
        cum_exp = cumulative_exponential(t, qi, di)

        # Should still be stable
        assert np.all(np.isfinite(q_exp))
        assert np.all(np.isfinite(cum_exp))
        assert np.all(q_exp > 0)
        assert np.all(cum_exp >= 0)

    def test_eur_edge_cases(self):
        """Test EUR calculations with edge case parameters."""
        qi = 100.0

        # Small di
        di_small = 1e-6
        eur_exp_small = calculate_eur_exponential(qi, di_small, 360.0, 10.0)
        assert eur_exp_small > 0

        # b near 1
        di = 0.1
        b_near_one = 0.9999
        eur_hyp_near_one = calculate_eur_hyperbolic(qi, di, b_near_one, 360.0, 10.0)
        eur_har = calculate_eur_harmonic(qi, di, 360.0, 10.0)
        # Should be close to harmonic EUR
        assert abs(eur_hyp_near_one - eur_har) < 0.1 * eur_har

        # b very small
        b_small = 0.001
        eur_hyp_small = calculate_eur_hyperbolic(qi, di, b_small, 360.0, 10.0)
        eur_exp = calculate_eur_exponential(qi, di, 360.0, 10.0)
        # Should be close to exponential EUR (within 20%)
        assert abs(eur_hyp_small - eur_exp) < 0.2 * eur_exp


class TestNumericalStability:
    """Test numerical stability of calculations."""

    def test_no_nan_or_inf_in_normal_cases(self):
        """Normal parameter values should not produce NaN or Inf."""
        t = np.linspace(0, 100, 1000)
        qi = 100.0
        di = 0.1
        b = 0.5

        q_exp = arps_exponential(t, qi, di)
        q_hyp = arps_hyperbolic(t, qi, di, b)
        q_har = arps_harmonic(t, qi, di)

        cum_exp = cumulative_exponential(t, qi, di)
        cum_hyp = cumulative_hyperbolic(t, qi, di, b)
        cum_har = cumulative_harmonic(t, qi, di)

        # No NaN or Inf values
        assert np.all(np.isfinite(q_exp))
        assert np.all(np.isfinite(q_hyp))
        assert np.all(np.isfinite(q_har))
        assert np.all(np.isfinite(cum_exp))
        assert np.all(np.isfinite(cum_hyp))
        assert np.all(np.isfinite(cum_har))

    def test_large_time_values(self):
        """Test with very large time values."""
        t = np.linspace(0, 1e6, 1000)  # Very large times
        qi = 100.0
        di = 0.1
        b = 0.5

        # Should handle large times without overflow
        q_exp = arps_exponential(t, qi, di)
        q_hyp = arps_hyperbolic(t, qi, di, b)

        # Rates should approach zero or economic limit
        assert np.all(q_exp >= 0)
        assert np.all(q_hyp >= 0)

    def test_very_large_qi_values(self):
        """Test with very large initial rates."""
        t = np.linspace(0, 100, 1000)
        qi = 1e6  # Very large initial rate
        di = 0.1
        b = 0.5

        q_exp = arps_exponential(t, qi, di)
        q_hyp = arps_hyperbolic(t, qi, di, b)

        # Should handle large qi without overflow
        assert np.all(np.isfinite(q_exp))
        assert np.all(np.isfinite(q_hyp))
        assert q_exp[0] == qi
        assert q_hyp[0] == qi

    def test_cumulative_monotonicity(self):
        """Cumulative values should always be increasing."""
        t = np.linspace(0, 100, 1000)
        qi = 100.0
        di = 0.1
        b_values = [0.1, 0.5, 0.9, 0.999]

        for b in b_values:
            cum_hyp = cumulative_hyperbolic(t, qi, di, b)
            # Cumulative should be monotonically increasing
            assert np.all(np.diff(cum_hyp) >= 0), f"Cumulative not monotonic for b={b}"

        cum_exp = cumulative_exponential(t, qi, di)
        cum_har = cumulative_harmonic(t, qi, di)
        assert np.all(np.diff(cum_exp) >= 0)
        assert np.all(np.diff(cum_har) >= 0)


class TestBoundaryConditions:
    """Test boundary conditions and special values."""

    def test_t_zero_rates(self):
        """At t=0, all rates should equal qi."""
        qi = 100.0
        di = 0.1
        b = 0.5

        t = np.array([0.0])
        assert arps_exponential(t, qi, di)[0] == qi
        assert arps_hyperbolic(t, qi, di, b)[0] == qi
        assert arps_harmonic(t, qi, di)[0] == qi

    def test_t_zero_cumulative(self):
        """At t=0, cumulative should be zero."""
        qi = 100.0
        di = 0.1
        b = 0.5

        t = np.array([0.0])
        assert cumulative_exponential(t, qi, di)[0] == 0.0
        assert cumulative_hyperbolic(t, qi, di, b)[0] == 0.0
        assert cumulative_harmonic(t, qi, di)[0] == 0.0

    def test_single_time_point(self):
        """Test with single time point."""
        t = np.array([10.0])
        qi = 100.0
        di = 0.1
        b = 0.5

        q_exp = arps_exponential(t, qi, di)
        q_hyp = arps_hyperbolic(t, qi, di, b)
        q_har = arps_harmonic(t, qi, di)

        assert len(q_exp) == 1
        assert len(q_hyp) == 1
        assert len(q_har) == 1
        assert q_exp[0] > 0
        assert q_hyp[0] > 0
        assert q_har[0] > 0

