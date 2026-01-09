"""
Layer 1: Objects

Immutable dataclasses representing the core domain.
No imports from other ressmith layers.
Only standard library, numpy, pandas, and timesmith.typing allowed.
"""

from ressmith.objects.domain import (
    CumSeries,
    DeclineSpec,
    EconResult,
    EconSpec,
    ForecastResult,
    ForecastSpec,
    ProductionSeries,
    RateSeries,
    WellMeta,
)

__all__ = [
    "WellMeta",
    "ProductionSeries",
    "RateSeries",
    "CumSeries",
    "DeclineSpec",
    "ForecastSpec",
    "EconSpec",
    "ForecastResult",
    "EconResult",
]

