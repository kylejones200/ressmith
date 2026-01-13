"""
Core domain objects for ResSmith.

Immutable dataclasses representing wells, production data, forecasts, and economics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

# Note: timesmith.typing is available for type hints if needed
# from timesmith.typing import SeriesLike, PanelLike


@dataclass(frozen=True)
class WellMeta:
    """Well metadata."""

    well_id: str
    basin: str
    operator: str
    fluid: str
    units: dict[str, str]
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProductionSeries:
    """Multi-phase production time series."""

    time_index: pd.DatetimeIndex
    oil: np.ndarray
    gas: np.ndarray
    water: np.ndarray
    choke: np.ndarray | None = None
    pressure: np.ndarray | None = None
    units: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate alignment of arrays."""
        n = len(self.time_index)
        if len(self.oil) != n or len(self.gas) != n or len(self.water) != n:
            raise ValueError(f"All arrays must have length {n} matching time_index")
        if self.choke is not None and len(self.choke) != n:
            raise ValueError(f"choke array must have length {n}")
        if self.pressure is not None and len(self.pressure) != n:
            raise ValueError(f"pressure array must have length {n}")


@dataclass(frozen=True)
class RateSeries:
    """Single-phase rate time series."""

    time_index: pd.DatetimeIndex
    rate: np.ndarray
    units: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate alignment."""
        if len(self.rate) != len(self.time_index):
            raise ValueError(
                f"rate array length {len(self.rate)} must match "
                f"time_index length {len(self.time_index)}"
            )


@dataclass(frozen=True)
class CumSeries:
    """Cumulative production time series."""

    time_index: pd.DatetimeIndex
    cumulative: np.ndarray
    units: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate alignment."""
        if len(self.cumulative) != len(self.time_index):
            raise ValueError(
                f"cumulative array length {len(self.cumulative)} must match "
                f"time_index length {len(self.time_index)}"
            )


@dataclass(frozen=True)
class DeclineSpec:
    """Decline model specification."""

    model_name: str
    parameters: dict[str, Any]
    start_date: datetime
    phase: str | None = None  # 'oil', 'gas', 'water', or None for total


@dataclass(frozen=True)
class ForecastSpec:
    """Forecast specification."""

    horizon: int  # Number of periods
    frequency: str  # Pandas frequency string, e.g., 'D', 'M'
    scenarios: dict[str, Any] | None = None


@dataclass(frozen=True)
class EconSpec:
    """Economics specification."""

    price_assumptions: dict[str, float]  # e.g., {'oil': 50.0, 'gas': 3.0}
    opex: float  # Operating expense per period
    capex: float  # Capital expenditure (one-time)
    discount_rate: float
    taxes: dict[str, float] | None = None
    units: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ForecastResult:
    """Forecast results."""

    yhat: pd.Series  # Point forecasts
    intervals: pd.DataFrame | None = None  # Optional prediction intervals
    metadata: dict[str, Any] = field(default_factory=dict)
    model_spec: DeclineSpec | None = None


@dataclass(frozen=True)
class EconResult:
    """Economics evaluation results."""

    cashflows: pd.DataFrame
    npv: float
    irr: float | None = None
    payout_time: float | None = None  # Periods to payout
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeclineSegment:
    """A single segment in a segmented decline curve."""

    kind: str  # "exponential", "harmonic", or "hyperbolic"
    parameters: dict[str, float]  # ARPS parameters: qi, di, b
    t_start: float = 0.0
    t_end: float | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    start_cum: float | None = None
    end_cum: float | None = None

    def __post_init__(self) -> None:
        """Validate segment parameters."""
        qi = self.parameters.get("qi", 0.0)
        di = self.parameters.get("di", 0.0)
        b = self.parameters.get("b", 0.0)

        if qi <= 0:
            raise ValueError("Initial rate (qi) must be positive")
        if di <= 0:
            raise ValueError("Decline rate (di) must be positive")
        if self.kind == "hyperbolic" and (b < 0 or b > 2.0):
            raise ValueError(f"b-factor must be between 0 and 2.0, got {b}")
        if self.kind == "exponential" and abs(b) > 1e-6:
            raise ValueError("b-factor must be 0 for exponential decline")
        if self.kind == "harmonic" and abs(b - 1.0) > 1e-6:
            raise ValueError("b-factor must be 1.0 for harmonic decline")


@dataclass(frozen=True)
class SegmentedDeclineResult:
    """Result of segmented decline curve analysis."""

    segments: list[DeclineSegment]
    forecast: pd.Series
    continuity_errors: list[str] = field(default_factory=list)
