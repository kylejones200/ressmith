"""Comprehensive numerical stability tests.

Tests for:
- Extreme parameter values
- Very long time horizons
- Division by zero scenarios
- Overflow/underflow conditions
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


class TestExtremeParameterValues:
    """Test with extreme parameter values."""

    def test_extremely_large_qi(self):
        """Test with very large initial rates."""
        t = np.linspace(0, 100, 1000)
        qi = 1e10  # Extremely large
        di = 0.1
        b = 0.5

        q_exp = arps_exponential(t, qi, di)
        q_hyp = arps_hyperbolic(t, qi, di, b)

        # Should handle without overflow
        assert np.all(np.isfinite(q_exp))
        assert np.all(np.isfinite(q_hyp))
        assert q_exp[0] == qi
        assert q_hyp[0] == qi

    def test_extremely_small_qi(self):
        """Test with very small initial rates."""
        t = np.linspace(0, 100, 1000)
        qi = 1e-10  # Extremely small
        di = 0.1
        b = 0.5

        q_exp = arps_exponential(t, qi, di)
        q_hyp = arps_hyperbolic(t, qi, di, b)

        # Should handle without underflow
        assert np.all(np.isfinite(q_exp))
        assert np.all(np.isfinite(q_hyp))
        assert q_exp[0] == qi
        assert q_hyp[0] == qi

    def test_extremely_large_di(self):
        """Test with very large decline rates."""
        t = np.linspace(0, 100, 1000)
        qi = 100.0
        di = 10.0  # Very large decline
        b = 0.5

        q_exp = arps_exponential(t, qi, di)
        q_hyp = arps_hyperbolic(t, qi, di, b)

        # Should decline very rapidly
        assert np.all(np.isfinite(q_exp))
        assert np.all(np.isfinite(q_hyp))
        assert q_exp[-1] < qi * 0.01  # Should be very small
        assert q_hyp[-1] < qi * 0.01

    def test_extremely_small_di(self):
        """Test with very small decline rates."""
        t = np.linspace(0, 100, 1000)
        qi = 100.0
        di = 1e-10  # Extremely small
        b = 0.5

        q_exp = arps_exponential(t, qi, di)
        q_hyp = arps_hyperbolic(t, qi, di, b)

        # Should remain nearly constant
        assert np.all(np.isfinite(q_exp))
        assert np.all(np.isfinite(q_hyp))
        assert q_exp[-1] > qi * 0.99
        assert q_hyp[-1] > qi * 0.99

    def test_extreme_b_values(self):
        """Test with b very close to boundaries."""
        t = np.linspace(0, 100, 1000)
        qi = 100.0
        di = 0.1

        # b very close to 0
        b_near_zero = 1e-10
        q_hyp_near_zero = arps_hyperbolic(t, qi, di, b_near_zero)
        assert np.all(np.isfinite(q_hyp_near_zero))

        # b very close to 1
        b_near_one = 1.0 - 1e-10
        q_hyp_near_one = arps_hyperbolic(t, qi, di, b_near_one)
        assert np.all(np.isfinite(q_hyp_near_one))


class TestVeryLongTimeHorizons:
    """Test with very long time horizons."""

    def test_very_long_time_exponential(self):
        """Test exponential with very long time."""
        t = np.linspace(0, 1e6, 10000)  # 1 million days
        qi = 100.0
        di = 0.001

        q_exp = arps_exponential(t, qi, di)
        cum_exp = cumulative_exponential(t, qi, di)

        # Should handle without overflow
        assert np.all(np.isfinite(q_exp))
        assert np.all(np.isfinite(cum_exp))
        assert np.all(q_exp >= 0)
        assert np.all(cum_exp >= 0)

    def test_very_long_time_hyperbolic(self):
        """Test hyperbolic with very long time."""
        t = np.linspace(0, 1e6, 10000)
        qi = 100.0
        di = 0.001
        b = 0.5

        q_hyp = arps_hyperbolic(t, qi, di, b)
        cum_hyp = cumulative_hyperbolic(t, qi, di, b)

        # Should handle without overflow
        assert np.all(np.isfinite(q_hyp))
        assert np.all(np.isfinite(cum_hyp))
        assert np.all(q_hyp >= 0)
        assert np.all(cum_hyp >= 0)

    def test_very_long_time_harmonic(self):
        """Test harmonic with very long time."""
        t = np.linspace(0, 1e6, 10000)
        qi = 100.0
        di = 0.001

        q_har = arps_harmonic(t, qi, di)
        cum_har = cumulative_harmonic(t, qi, di)

        # Should handle without overflow
        assert np.all(np.isfinite(q_har))
        assert np.all(np.isfinite(cum_har))
        assert np.all(q_har >= 0)
        assert np.all(cum_har >= 0)

    def test_eur_very_long_horizon(self):
        """Test EUR calculation with very long time horizon."""
        qi = 100.0
        di = 0.001
        t_max = 1e6  # Very long
        econ_limit = 1.0

        eur_exp = calculate_eur_exponential(qi, di, t_max, econ_limit)
        eur_hyp = calculate_eur_hyperbolic(qi, di, 0.5, t_max, econ_limit)
        eur_har = calculate_eur_harmonic(qi, di, t_max, econ_limit)

        # Should be finite and positive
        assert np.isfinite(eur_exp)
        assert np.isfinite(eur_hyp)
        assert np.isfinite(eur_har)
        assert eur_exp > 0
        assert eur_hyp > 0
        assert eur_har > 0


class TestDivisionByZero:
    """Test division by zero scenarios."""

    def test_division_by_zero_in_hyperbolic(self):
        """Test hyperbolic with parameters that could cause division by zero."""
        t = np.linspace(0, 100, 1000)
        qi = 100.0
        di = 0.0  # Zero decline rate
        b = 0.5

        # Should handle gracefully
        try:
            q_hyp = arps_hyperbolic(t, qi, di, b)
            # If no error, should be constant at qi
            assert np.allclose(q_hyp, qi)
        except (ZeroDivisionError, ValueError):
            # Error is acceptable for di=0
            pass

    def test_division_by_zero_in_cumulative(self):
        """Test cumulative with zero decline rate."""
        t = np.linspace(0, 100, 1000)
        qi = 100.0
        di = 0.0

        try:
            cum_exp = cumulative_exponential(t, qi, di)
            # If no error, should be approximately qi * t
            expected = qi * t
            assert np.allclose(cum_exp, expected, rtol=0.01)
        except (ZeroDivisionError, ValueError):
            # Error is acceptable
            pass

    def test_division_by_zero_in_eur(self):
        """Test EUR calculation with zero decline rate."""
        qi = 100.0
        di = 0.0
        t_max = 360.0
        econ_limit = 10.0

        try:
            eur_exp = calculate_eur_exponential(qi, di, t_max, econ_limit)
            # If no error, should be approximately qi * t_max
            assert np.isfinite(eur_exp)
            assert eur_exp > 0
        except (ZeroDivisionError, ValueError):
            # Error is acceptable
            pass

    def test_division_by_zero_b_near_one(self):
        """Test division by zero when b approaches 1."""
        t = np.linspace(0, 100, 1000)
        qi = 100.0
        di = 0.1
        b = 1.0  # Exactly 1.0

        # Should use harmonic formula (no division by zero)
        q_hyp = arps_hyperbolic(t, qi, di, b)
        q_har = arps_harmonic(t, qi, di)

        # Should match harmonic
        np.testing.assert_allclose(q_hyp, q_har, rtol=1e-5)


class TestOverflowUnderflow:
    """Test overflow and underflow conditions."""

    def test_overflow_prevention(self):
        """Test that calculations don't overflow."""
        t = np.linspace(0, 1000, 10000)
        qi = 1e6
        di = 0.1
        b = 0.5

        q_exp = arps_exponential(t, qi, di)
        q_hyp = arps_hyperbolic(t, qi, di, b)
        cum_exp = cumulative_exponential(t, qi, di)
        cum_hyp = cumulative_hyperbolic(t, qi, di, b)

        # No overflow (no Inf values)
        assert np.all(np.isfinite(q_exp))
        assert np.all(np.isfinite(q_hyp))
        assert np.all(np.isfinite(cum_exp))
        assert np.all(np.isfinite(cum_hyp))

    def test_underflow_prevention(self):
        """Test that calculations don't underflow to zero incorrectly."""
        t = np.linspace(0, 100, 1000)
        qi = 1e-10  # Very small
        di = 0.01
        b = 0.5

        q_exp = arps_exponential(t, qi, di)
        q_hyp = arps_hyperbolic(t, qi, di, b)

        # Should still be finite (not underflow to zero)
        assert np.all(np.isfinite(q_exp))
        assert np.all(np.isfinite(q_hyp))
        # Early values should still be close to qi
        assert q_exp[0] == qi
        assert q_hyp[0] == qi

    def test_numerical_precision(self):
        """Test numerical precision with extreme values."""
        t = np.linspace(0, 100, 1000)
        qi = 100.0
        di = 1e-15  # Extremely small
        b = 0.5

        q_exp = arps_exponential(t, qi, di)
        q_hyp = arps_hyperbolic(t, qi, di, b)

        # Should remain stable (not lose precision)
        assert np.all(np.isfinite(q_exp))
        assert np.all(np.isfinite(q_hyp))
        # Should remain close to qi
        assert np.allclose(q_exp, qi, rtol=1e-10)
        assert np.allclose(q_hyp, qi, rtol=1e-10)
