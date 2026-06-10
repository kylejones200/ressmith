"""
Optional integrations with external libraries.

- decline_curve: Use the decline-curve package for Arps (exponential,
  hyperbolic, harmonic) and ML-based production forecasting.
  Install with: pip install petrosmith[dca]
"""

from petrosmith.integrations.decline_curve import (
    forecast_with_dca,
    decline_curve_available,
)

__all__ = [
    "forecast_with_dca",
    "decline_curve_available",
]
