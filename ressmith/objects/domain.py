"""
Core domain objects for ResSmith.

Immutable dataclasses representing wells, production data, forecasts, and economics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
from timesmith.typing import SeriesLike


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
    choke: Optional[np.ndarray] = None
    pressure: Optional[np.ndarray] = None
    units: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate alignment of arrays."""
        n = len(self.time_index)
        if len(self.oil) != n or len(self.gas) != n or len(self.water) != n:
            raise ValueError(
                f"All arrays must have length {n} matching time_index"
            )
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
    phase: Optional[str] = None  # 'oil', 'gas', 'water', or None for total


@dataclass(frozen=True)
class ForecastSpec:
    """Forecast specification."""

    horizon: int  # Number of periods
    frequency: str  # Pandas frequency string, e.g., 'D', 'M'
    scenarios: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class EconSpec:
    """Economics specification."""

    price_assumptions: dict[str, float]  # e.g., {'oil': 50.0, 'gas': 3.0}
    opex: float  # Operating expense per period
    capex: float  # Capital expenditure (one-time)
    discount_rate: float
    taxes: Optional[dict[str, float]] = None
    units: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ForecastResult:
    """Forecast results."""

    yhat: pd.Series  # Point forecasts
    intervals: Optional[pd.DataFrame] = None  # Optional prediction intervals
    metadata: dict[str, Any] = field(default_factory=dict)
    model_spec: Optional[DeclineSpec] = None


@dataclass(frozen=True)
class EconResult:
    """Economics evaluation results."""

    cashflows: pd.DataFrame
    npv: float
    irr: Optional[float] = None
    payout_time: Optional[float] = None  # Periods to payout
    metadata: dict[str, Any] = field(default_factory=dict)

