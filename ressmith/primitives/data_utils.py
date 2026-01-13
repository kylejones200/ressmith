"""
Data extraction utilities for models.

Helper functions to extract rate and time_index from various input types.
"""

import pandas as pd

from ressmith.objects.domain import ProductionSeries, RateSeries

# Optional timesmith validation
try:
    from timesmith.typing.validators import assert_series_like

    HAS_TIMESMITH_VALIDATORS = True
except ImportError:
    HAS_TIMESMITH_VALIDATORS = False


def extract_rate_data(
    data: ProductionSeries | RateSeries | pd.DataFrame,
) -> tuple[pd.Series, pd.DatetimeIndex]:
    """
    Extract rate data and time index from various input types.

    Parameters
    ----------
    data : ProductionSeries, RateSeries, or pd.DataFrame
        Input data

    Returns
    -------
    tuple
        (rate_series, time_index)
    """
    if isinstance(data, pd.DataFrame):
        if "oil" in data.columns:
            rate = data["oil"]
        elif len(data.columns) == 1:
            rate = data.iloc[:, 0]
        else:
            raise ValueError("DataFrame must have 'oil' column or single column")
        time_index = data.index
        if HAS_TIMESMITH_VALIDATORS:
            rate_series = pd.Series(rate, index=time_index)
            assert_series_like(rate_series, name="data")
        elif not isinstance(time_index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")
        return rate, time_index
    elif isinstance(data, ProductionSeries):
        rate = pd.Series(data.oil, index=data.time_index)
        return rate, data.time_index
    elif isinstance(data, RateSeries):
        rate = pd.Series(data.rate, index=data.time_index)
        return rate, data.time_index
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")
