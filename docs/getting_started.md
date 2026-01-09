This guide walks through a full ResSmith workflow using a single well. The example uses a synthetic decline series. The goal is clarity and structure.

We begin with a pandas DataFrame that represents monthly oil production.

```python
import numpy as np
import pandas as pd

idx = pd.date_range("2019-01-01", periods=48, freq="M")
rate = 800 * (1 + 0.02 * idx.month) ** -1
rate = rate + np.random.normal(0, 20, size=len(rate))

df = pd.DataFrame(
    {"date": idx, "oil_rate": rate}
).set_index("date")
```

ResSmith validates temporal inputs using shared typing from Timesmith.

```python
from timesmith.typing.validators import assert_series_like

assert_series_like(df["oil_rate"])
```

We fit a decline model.

```python
from ressmith.workflows import fit_forecast

forecast = fit_forecast(
    data=df,
    rate_col="oil_rate",
    model="arps_hyperbolic",
    horizon=24
)
```

The result contains a forecasted rate series and model metadata.

```python
print(forecast.yhat.tail())
```

We now evaluate economics.

```python
from ressmith.workflows import evaluate_economics
from ressmith.objects import EconSpec

econ = EconSpec(
    oil_price=70.0,
    opex_per_unit=12.0,
    discount_rate=0.1
)

econ_result = evaluate_economics(
    forecast=forecast,
    econ_spec=econ
)
```

The output includes cashflows and value metrics.

```python
print(econ_result.npv)
```

This is the full ResSmith loop. Rates become forecasts. Forecasts become cashflow. Each step stays explicit and testable.

