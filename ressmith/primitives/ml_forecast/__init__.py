"""
ML and statistical forecasters, spatial kriging, and production data QA.

Ported from decline-curve (pydca). Optional dependencies:
``pip install ressmith[stats]``, ``ressmith[llm]``, ``ressmith[ml]``, ``ressmith[spatial]``.
"""

from ressmith.primitives.ml_forecast.data_contract import (
    CANONICAL_SCHEMA,
    ProductionDataValidator,
    ValidationResult,
    normalize_to_canonical_schema,
    validate_production_data,
)
from ressmith.primitives.ml_forecast.data_qa import (
    QAResult,
    apply_rate_cut,
    run_data_qa,
)
from ressmith.primitives.ml_forecast.ensemble import (
    EnsembleForecaster,
    EnsembleWeights,
    ensemble_forecast,
)
from ressmith.primitives.ml_forecast.forecast_arima import forecast_arima
from ressmith.primitives.ml_forecast.forecast_chronos import (
    check_chronos_availability,
    forecast_chronos,
    forecast_chronos_probabilistic,
)
from ressmith.primitives.ml_forecast.forecast_deepar import DeepARForecaster
from ressmith.primitives.ml_forecast.forecast_statistical import (
    calculate_confidence_intervals,
    holt_winters_forecast,
    linear_trend_forecast,
    moving_average_forecast,
    simple_exponential_smoothing,
)
from ressmith.primitives.ml_forecast.forecast_tft import TFTForecaster
from ressmith.primitives.ml_forecast.forecast_timesfm import (
    check_timesfm_availability,
    forecast_timesfm,
)
from ressmith.primitives.ml_forecast.forecaster import Forecaster
from ressmith.primitives.ml_forecast.spatial_kriging import (
    KrigingResult,
    create_eur_map,
    improve_eur_with_kriging,
    krige_eur,
)

__all__ = [
    "CANONICAL_SCHEMA",
    "DeepARForecaster",
    "EnsembleForecaster",
    "EnsembleWeights",
    "Forecaster",
    "KrigingResult",
    "ProductionDataValidator",
    "QAResult",
    "TFTForecaster",
    "ValidationResult",
    "apply_rate_cut",
    "calculate_confidence_intervals",
    "check_chronos_availability",
    "check_timesfm_availability",
    "create_eur_map",
    "ensemble_forecast",
    "forecast_arima",
    "forecast_chronos",
    "forecast_chronos_probabilistic",
    "forecast_timesfm",
    "holt_winters_forecast",
    "improve_eur_with_kriging",
    "krige_eur",
    "linear_trend_forecast",
    "moving_average_forecast",
    "normalize_to_canonical_schema",
    "run_data_qa",
    "simple_exponential_smoothing",
    "validate_production_data",
]
