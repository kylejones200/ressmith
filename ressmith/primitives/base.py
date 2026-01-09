"""
Base classes for ResSmith primitives.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

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


class BaseObject(ABC):
    """Base class with parameter management."""

    def __init__(self, **params: Any) -> None:
        """Initialize with parameters."""
        self._params = params.copy()

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters."""
        if deep:
            return deepcopy(self._params)
        return self._params.copy()

    def set_params(self, **params: Any) -> "BaseObject":
        """Set parameters and return self."""
        self._params.update(params)
        return self

    def clone(self) -> "BaseObject":
        """Create a copy with same parameters."""
        return type(self)(**self.get_params())

    def __repr__(self) -> str:
        """String representation."""
        params_str = ", ".join(
            f"{k}={v!r}" for k, v in self._params.items()
        )
        return f"{self.__class__.__name__}({params_str})"

    @property
    def tags(self) -> dict[str, Any]:
        """Model tags describing capabilities."""
        return {}


class BaseEstimator(BaseObject):
    """Base class with fitted state."""

    def __init__(self, **params: Any) -> None:
        """Initialize estimator."""
        super().__init__(**params)
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """Check if estimator is fitted."""
        return self._fitted

    def _check_fitted(self) -> None:
        """Raise error if not fitted."""
        if not self._fitted:
            raise ValueError(
                f"{self.__class__.__name__} must be fitted before use"
            )


class BaseDeclineModel(BaseEstimator):
    """Base class for decline curve models."""

    def fit(
        self, data: ProductionSeries | RateSeries | pd.DataFrame, **fit_params: Any
    ) -> "BaseDeclineModel":
        """
        Fit the decline model to data.

        Parameters
        ----------
        data : ProductionSeries or RateSeries
            Production or rate data to fit
        **fit_params : dict
            Additional fitting parameters

        Returns
        -------
        self : BaseDeclineModel
            Fitted model
        """
        self._fitted = True
        return self

    @abstractmethod
    def predict(
        self, spec: ForecastSpec
    ) -> ForecastResult:
        """
        Generate forecast.

        Parameters
        ----------
        spec : ForecastSpec
            Forecast specification

        Returns
        -------
        ForecastResult
            Forecast results
        """
        pass

    @property
    def tags(self) -> dict[str, Any]:
        """Model capability tags."""
        return {
            "supports_oil": False,
            "supports_gas": False,
            "supports_water": False,
            "supports_multiphase": False,
            "supports_irregular_time": False,
            "requires_positive": True,
            "supports_censoring": False,
            "supports_intervals": False,
        }


class BaseEconModel(BaseObject):
    """Base class for economics models."""

    def evaluate(
        self,
        forecast: ForecastResult,
        spec: EconSpec,
        **eval_params: Any,
    ) -> EconResult:
        """
        Evaluate economics for a forecast.

        Parameters
        ----------
        forecast : ForecastResult
            Forecast results
        spec : EconSpec
            Economics specification

        Returns
        -------
        EconResult
            Economics results
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.evaluate not implemented"
        )

