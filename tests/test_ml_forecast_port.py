"""Smoke tests for pydca/decline-curve code ported to ressmith.primitives.ml_forecast."""

import numpy as np
import pandas as pd
import pytest

from ressmith import dca
from ressmith.primitives.ml_forecast import (
    Forecaster,
    validate_production_data,
)


def test_forecaster_arps_hyperbolic() -> None:
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    s = pd.Series(np.linspace(100.0, 50.0, 24), index=dates)
    f = Forecaster(s)
    out = f.forecast("arps", kind="hyperbolic", horizon=6)
    assert len(out) == len(s) + 6
    assert (out.values >= 0).all()


def test_forecaster_linear_trend() -> None:
    dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    s = pd.Series(np.arange(12, dtype=float) * 5 + 50, index=dates)
    f = Forecaster(s)
    out = f.forecast("linear_trend", horizon=3)
    assert len(out) == 3


def test_dca_evaluate() -> None:
    idx = pd.date_range("2020-01-01", periods=5, freq="MS")
    a = pd.Series([1.0, 2, 3, 4, 5], index=idx)
    b = pd.Series([1.1, 2.1, 2.9, 4.0, 5.1], index=idx)
    m = dca.evaluate(a, b)
    assert "rmse" in m and m["rmse"] >= 0


def test_validate_production_data_ok() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=10, freq="MS"),
            "oil_rate": np.linspace(100, 50, 10),
        }
    )
    r = validate_production_data(df, well_id_column=None)
    assert r.is_valid


@pytest.mark.parametrize("model", ["arima", "timesfm", "chronos"])
def test_dca_single_well_ml_fallback_or_run(model: str) -> None:
    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    s = pd.Series(100 * (0.98 ** np.arange(36)), index=dates)
    out = dca.single_well(s, model=model, horizon=6)
    assert isinstance(out, pd.Series)
    assert len(out) > 0
