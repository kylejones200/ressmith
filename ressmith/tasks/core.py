"""
Core task classes for ResSmith.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from ressmith.objects.domain import (
    EconResult,
    EconSpec,
    ForecastResult,
    ForecastSpec,
    ProductionSeries,
    RateSeries,
)
from ressmith.objects.validate import (
    assert_alignment,
    assert_monotonic_time,
    assert_positive_rates,
    assert_rate_series_alignment,
)
from ressmith.primitives.base import BaseDeclineModel, BaseEconModel
from ressmith.primitives.economics import (
    cashflow_from_forecast,
    irr,
    npv,
)

logger = logging.getLogger(__name__)


class FitDeclineTask:
    """Task for fitting decline models to data."""

    def __init__(
        self,
        model: BaseDeclineModel,
        column_mapping: Optional[dict[str, str]] = None,
        phase: str = "oil",
    ) -> None:
        """
        Initialize fit task.

        Parameters
        ----------
        model : BaseDeclineModel
            Decline model to fit
        column_mapping : dict, optional
            Mapping from standard names to DataFrame columns
        phase : str
            Phase to fit: 'oil', 'gas', 'water' (default: 'oil')
        """
        self.model = model
        self.column_mapping = column_mapping or {}
        self.phase = phase

    def run(
        self,
        data: pd.DataFrame | pd.Series,
        horizon: Optional[int] = None,
    ) -> tuple[BaseDeclineModel, ForecastResult]:
        """
        Run fit task.

        Parameters
        ----------
        data : pd.DataFrame or pd.Series
            Input data
        horizon : int, optional
            Forecast horizon for in-sample forecast

        Returns
        -------
        tuple
            (fitted_model, forecast_result)
        """
        # Convert to Layer 1 objects
        if isinstance(data, pd.Series):
            time_index = data.index
            if not isinstance(time_index, pd.DatetimeIndex):
                raise ValueError("Series index must be DatetimeIndex")
            rate = data.values
            rate_series = RateSeries(
                time_index=time_index, rate=rate, units={}
            )
            # Validate
            assert_monotonic_time(time_index)
            assert_positive_rates(rate)
            assert_rate_series_alignment(rate_series)
            # Fit
            fitted_model = self.model.fit(rate_series)
        else:
            # DataFrame case
            time_col = self.column_mapping.get("time", data.index.name or "index")
            if time_col == "index":
                time_index = data.index
            else:
                time_index = pd.to_datetime(data[time_col])

            if not isinstance(time_index, pd.DatetimeIndex):
                raise ValueError("Time column must be datetime")

            # Extract phase data
            phase_col = self.column_mapping.get(self.phase, self.phase)
            if phase_col not in data.columns:
                raise ValueError(f"Phase column '{phase_col}' not found")

            rate = data[phase_col].values
            oil = data.get("oil", np.zeros_like(rate))
            gas = data.get("gas", np.zeros_like(rate))
            water = data.get("water", np.zeros_like(rate))

            prod_series = ProductionSeries(
                time_index=time_index,
                oil=oil.values if hasattr(oil, "values") else oil,
                gas=gas.values if hasattr(gas, "values") else gas,
                water=water.values if hasattr(water, "values") else water,
                units={},
            )
            # Validate
            assert_monotonic_time(time_index)
            assert_positive_rates(rate, name=self.phase)
            assert_alignment(prod_series)
            # Fit
            fitted_model = self.model.fit(prod_series)

        # Generate forecast if horizon specified
        if horizon is not None:
            forecast_spec = ForecastSpec(horizon=horizon, frequency="D")
            forecast_result = fitted_model.predict(forecast_spec)
        else:
            # Return empty forecast result
            forecast_result = ForecastResult(
                yhat=pd.Series(dtype=float), metadata={}
            )

        return fitted_model, forecast_result


class ForecastTask:
    """Task for generating forecasts from fitted models."""

    def __init__(self, model: BaseDeclineModel) -> None:
        """Initialize forecast task."""
        self.model = model

    def run(self, spec: ForecastSpec) -> tuple[ForecastResult, pd.DataFrame]:
        """
        Run forecast task.

        Parameters
        ----------
        spec : ForecastSpec
            Forecast specification

        Returns
        -------
        tuple
            (forecast_result, diagnostics_table)
        """
        # Generate forecast
        result = self.model.predict(spec)

        # Compute diagnostics (residual stats if we had in-sample data)
        diagnostics = pd.DataFrame(
            {
                "metric": ["horizon", "frequency"],
                "value": [spec.horizon, spec.frequency],
            }
        )

        return result, diagnostics


class EconTask:
    """Task for economics evaluation."""

    def __init__(self, econ_model: Optional[BaseEconModel] = None) -> None:
        """Initialize economics task."""
        self.econ_model = econ_model

    def run(
        self, forecast: ForecastResult, spec: EconSpec
    ) -> tuple[EconResult, dict[str, Any]]:
        """
        Run economics task.

        Parameters
        ----------
        forecast : ForecastResult
            Forecast results
        spec : EconSpec
            Economics specification

        Returns
        -------
        tuple
            (econ_result, summary_dict)
        """
        # Build cashflows
        cashflows = cashflow_from_forecast(forecast, spec)

        # Compute NPV and IRR
        net_cf = cashflows["net_cashflow"].values
        npv_value = npv(net_cf, spec.discount_rate)
        irr_value = irr(net_cf)

        # Compute payout time
        cumulative_cf = np.cumsum(net_cf)
        payout_idx = np.where(cumulative_cf >= 0)[0]
        payout_time = float(payout_idx[0]) if len(payout_idx) > 0 else None

        # Create result
        result = EconResult(
            cashflows=cashflows,
            npv=npv_value,
            irr=irr_value,
            payout_time=payout_time,
            metadata={},
        )

        # Summary
        summary = {
            "npv": npv_value,
            "irr": irr_value,
            "payout_time": payout_time,
            "total_revenue": cashflows["revenue"].sum(),
            "total_opex": cashflows["opex"].sum(),
            "total_capex": cashflows["capex"].sum(),
        }

        return result, summary


class BatchTask:
    """Task for batch processing multiple wells."""

    def __init__(
        self,
        task_func: Callable[[str], Any],
        max_workers: int = 4,
        parallel: bool = True,
    ) -> None:
        """
        Initialize batch task.

        Parameters
        ----------
        task_func : callable
            Function that takes well_id and returns results dict
        max_workers : int
            Maximum parallel workers (default: 4)
        parallel : bool
            Enable parallel execution (default: True)
        """
        self.task_func = task_func
        self.max_workers = max_workers
        self.parallel = parallel

    def run(self, well_ids: list[str]) -> pd.DataFrame:
        """
        Run batch task.

        Parameters
        ----------
        well_ids : list
            List of well IDs to process

        Returns
        -------
        pd.DataFrame
            Results table with columns: well_id, fit_params, eur, npv, success, error
        """
        results = []

        if self.parallel:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_well = {
                    executor.submit(self.task_func, well_id): well_id
                    for well_id in well_ids
                }
                for future in as_completed(future_to_well):
                    well_id = future_to_well[future]
                    try:
                        result = future.result()
                        result["well_id"] = well_id
                        result["success"] = True
                        result["error"] = None
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing {well_id}: {e}")
                        results.append(
                            {
                                "well_id": well_id,
                                "success": False,
                                "error": str(e),
                                "fit_params": None,
                                "eur": None,
                                "npv": None,
                            }
                        )
        else:
            for well_id in well_ids:
                try:
                    result = self.task_func(well_id)
                    result["well_id"] = well_id
                    result["success"] = True
                    result["error"] = None
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {well_id}: {e}")
                    results.append(
                        {
                            "well_id": well_id,
                            "success": False,
                            "error": str(e),
                            "fit_params": None,
                            "eur": None,
                            "npv": None,
                        }
                    )

        return pd.DataFrame(results)

