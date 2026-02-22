"""Well test validation workflows.

Provides workflows for validating well test data quality and results,
checking for common issues and ensuring tests make engineering sense.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ressmith.primitives.well_test import (
    WellTestResult,
    analyze_buildup_test,
    analyze_drawdown_test,
)

logger = logging.getLogger(__name__)


@dataclass
class WellTestValidationResult:
    """Container for well test validation results.

    Attributes
    ----------
    is_valid : bool
        Whether the test passes validation
    warnings : list[str]
        List of warning messages
    errors : list[str]
        List of error messages
    quality_score : float
        Quality score (0-1, higher is better)
    validation_checks : dict[str, bool]
        Results of individual validation checks
    """

    is_valid: bool
    warnings: list[str]
    errors: list[str]
    quality_score: float
    validation_checks: dict[str, bool]


def validate_well_test_data(
    time: np.ndarray,
    pressure: np.ndarray,
    test_type: str = "buildup",
    production_rate: float | None = None,
    production_time: float | None = None,
) -> WellTestValidationResult:
    """Validate well test data quality.

    Checks for common data quality issues:
    - Insufficient data points
    - Non-monotonic pressure behavior
    - Unrealistic pressure values
    - Missing required parameters
    - Data gaps or outliers

    Parameters
    ----------
    time : np.ndarray
        Time array (hours)
    pressure : np.ndarray
        Pressure array (psi)
    test_type : str
        Test type ('buildup' or 'drawdown') (default: 'buildup')
    production_rate : float, optional
        Production rate (required for buildup tests)
    production_time : float, optional
        Production time before shut-in (required for buildup tests)

    Returns
    -------
    WellTestValidationResult
        Validation results with warnings and errors

    Examples
    --------
    >>> import numpy as np
    >>> time = np.array([0.1, 0.5, 1, 2, 5, 10, 24])
    >>> pressure = np.array([3000, 3200, 3400, 3600, 3800, 3900, 3950])
    >>> result = validate_well_test_data(time, pressure, test_type='buildup', production_rate=1000, production_time=720)
    >>> print(f"Test valid: {result.is_valid}")
    >>> print(f"Quality score: {result.quality_score:.2f}")
    """
    logger.info(f"Validating {test_type} test data")

    warnings = []
    errors = []
    validation_checks = {}
    quality_score = 1.0

    # Check 1: Sufficient data points
    if len(time) < 5:
        errors.append(f"Insufficient data points: {len(time)} (minimum 5 required)")
        validation_checks["sufficient_data"] = False
        quality_score -= 0.3
    else:
        validation_checks["sufficient_data"] = True

    # Check 2: Time and pressure arrays have same length
    if len(time) != len(pressure):
        errors.append(
            f"Time and pressure arrays have different lengths: {len(time)} vs {len(pressure)}"
        )
        validation_checks["array_lengths_match"] = False
        quality_score -= 0.3
    else:
        validation_checks["array_lengths_match"] = True

    # Check 3: Time is monotonically increasing
    if len(time) > 1:
        time_diff = np.diff(time)
        if np.any(time_diff <= 0):
            errors.append("Time array is not monotonically increasing")
            validation_checks["monotonic_time"] = False
            quality_score -= 0.2
        else:
            validation_checks["monotonic_time"] = True
    else:
        validation_checks["monotonic_time"] = True

    # Check 4: Pressure values are realistic
    if np.any(pressure <= 0):
        errors.append("Pressure values must be positive")
        validation_checks["positive_pressure"] = False
        quality_score -= 0.2
    else:
        validation_checks["positive_pressure"] = True

    if np.any(pressure > 20000):
        warnings.append("Some pressure values exceed 20,000 psi (unusually high)")
        validation_checks["realistic_pressure"] = False
        quality_score -= 0.1
    else:
        validation_checks["realistic_pressure"] = True

    # Check 5: Expected pressure behavior for test type
    if test_type == "buildup":
        if production_rate is None or production_rate <= 0:
            errors.append("Production rate required for buildup test validation")
            validation_checks["required_params"] = False
            quality_score -= 0.2
        else:
            validation_checks["required_params"] = True

        if production_time is None or production_time <= 0:
            errors.append("Production time required for buildup test validation")
            validation_checks["required_params"] = False
            quality_score -= 0.2
        else:
            validation_checks["required_params"] = True

        # Buildup: pressure should increase
        if len(pressure) > 1:
            pressure_increase = pressure[-1] - pressure[0]
            if pressure_increase < 0:
                warnings.append(
                    "Pressure decreased during buildup test (unexpected behavior)"
                )
                validation_checks["expected_behavior"] = False
                quality_score -= 0.15
            elif pressure_increase < 50:
                warnings.append(
                    f"Small pressure increase during buildup: {pressure_increase:.1f} psi (may indicate poor test quality)"
                )
                validation_checks["expected_behavior"] = True
                quality_score -= 0.05
            else:
                validation_checks["expected_behavior"] = True

    elif test_type == "drawdown":
        # Drawdown: pressure should decrease
        if len(pressure) > 1:
            pressure_decrease = pressure[0] - pressure[-1]
            if pressure_decrease < 0:
                warnings.append(
                    "Pressure increased during drawdown test (unexpected behavior)"
                )
                validation_checks["expected_behavior"] = False
                quality_score -= 0.15
            elif pressure_decrease < 50:
                warnings.append(
                    f"Small pressure decrease during drawdown: {pressure_decrease:.1f} psi (may indicate poor test quality)"
                )
                validation_checks["expected_behavior"] = True
                quality_score -= 0.05
            else:
                validation_checks["expected_behavior"] = True

    # Check 6: Data gaps
    if len(time) > 1:
        time_diffs = np.diff(time)
        max_gap = np.max(time_diffs)
        mean_gap = np.mean(time_diffs)
        if max_gap > 10 * mean_gap:
            warnings.append(
                f"Large time gap detected: {max_gap:.1f} hours (may indicate missing data)"
            )
            validation_checks["no_large_gaps"] = False
            quality_score -= 0.1
        else:
            validation_checks["no_large_gaps"] = True

    # Check 7: Outliers
    if len(pressure) > 3:
        pressure_median = np.median(pressure)
        pressure_std = np.std(pressure)
        outliers = np.abs(pressure - pressure_median) > 3 * pressure_std
        if np.any(outliers):
            n_outliers = np.sum(outliers)
            warnings.append(
                f"{n_outliers} potential outlier(s) detected in pressure data"
            )
            validation_checks["no_outliers"] = False
            quality_score -= 0.1
        else:
            validation_checks["no_outliers"] = True

    # Check 8: Test duration
    test_duration = time[-1] - time[0] if len(time) > 1 else 0
    if test_duration < 1:
        warnings.append(
            f"Short test duration: {test_duration:.2f} hours (may not capture full response)"
        )
        validation_checks["adequate_duration"] = False
        quality_score -= 0.1
    elif test_duration > 1000:
        warnings.append(
            f"Very long test duration: {test_duration:.1f} hours (may include operational issues)"
        )
        validation_checks["adequate_duration"] = True
        quality_score -= 0.05
    else:
        validation_checks["adequate_duration"] = True

    quality_score = max(0.0, min(1.0, quality_score))
    is_valid = len(errors) == 0 and quality_score >= 0.5

    return WellTestValidationResult(
        is_valid=is_valid,
        warnings=warnings,
        errors=errors,
        quality_score=quality_score,
        validation_checks=validation_checks,
    )


def validate_well_test_results(
    test_result: WellTestResult,
    expected_ranges: dict[str, tuple[float, float]] | None = None,
) -> WellTestValidationResult:
    """Validate well test analysis results.

    Checks if test results make engineering sense:
    - Permeability in reasonable range
    - Skin factor in reasonable range
    - Consistency between parameters
    - Boundary detection makes sense

    Parameters
    ----------
    test_result : WellTestResult
        Well test analysis result
    expected_ranges : dict, optional
        Expected ranges for parameters:
        - permeability: (min, max) in md
        - skin: (min, max)
        - reservoir_pressure: (min, max) in psi

    Returns
    -------
    WellTestValidationResult
        Validation results with warnings and errors

    Examples
    --------
    >>> from ressmith.primitives.well_test import analyze_buildup_test
    >>> import numpy as np
    >>> time = np.array([0.1, 0.5, 1, 2, 5, 10, 24])
    >>> pressure = np.array([3000, 3200, 3400, 3600, 3800, 3900, 3950])
    >>> result = analyze_buildup_test(time, pressure, 1000, 720)
    >>> validation = validate_well_test_results(result)
    >>> print(f"Results valid: {validation.is_valid}")
    """
    logger.info("Validating well test results")

    warnings = []
    errors = []
    validation_checks = {}
    quality_score = 1.0

    # Default expected ranges
    if expected_ranges is None:
        expected_ranges = {
            "permeability": (0.001, 10000.0),  # md
            "skin": (-10.0, 50.0),
            "reservoir_pressure": (100.0, 20000.0),  # psi
        }

    # Check 1: Permeability in reasonable range
    k_min, k_max = expected_ranges.get("permeability", (0.001, 10000.0))
    if test_result.permeability < k_min:
        errors.append(
            f"Permeability too low: {test_result.permeability:.3f} md (expected >= {k_min} md)"
        )
        validation_checks["permeability_range"] = False
        quality_score -= 0.3
    elif test_result.permeability > k_max:
        warnings.append(
            f"Permeability very high: {test_result.permeability:.1f} md (expected <= {k_max} md)"
        )
        validation_checks["permeability_range"] = False
        quality_score -= 0.1
    else:
        validation_checks["permeability_range"] = True

    # Check 2: Skin factor in reasonable range
    skin_min, skin_max = expected_ranges.get("skin", (-10.0, 50.0))
    if test_result.skin < skin_min:
        warnings.append(
            f"Skin factor very negative: {test_result.skin:.2f} (may indicate stimulation or error)"
        )
        validation_checks["skin_range"] = False
        quality_score -= 0.1
    elif test_result.skin > skin_max:
        warnings.append(
            f"Skin factor very high: {test_result.skin:.2f} (may indicate damage or error)"
        )
        validation_checks["skin_range"] = False
        quality_score -= 0.1
    else:
        validation_checks["skin_range"] = True

    # Check 3: Reservoir pressure in reasonable range
    if test_result.reservoir_pressure is not None:
        p_min, p_max = expected_ranges.get("reservoir_pressure", (100.0, 20000.0))
        if test_result.reservoir_pressure < p_min:
            errors.append(
                f"Reservoir pressure too low: {test_result.reservoir_pressure:.1f} psi"
            )
            validation_checks["pressure_range"] = False
            quality_score -= 0.2
        elif test_result.reservoir_pressure > p_max:
            warnings.append(
                f"Reservoir pressure very high: {test_result.reservoir_pressure:.1f} psi"
            )
            validation_checks["pressure_range"] = False
            quality_score -= 0.1
        else:
            validation_checks["pressure_range"] = True
    else:
        warnings.append("Reservoir pressure not estimated")
        validation_checks["pressure_range"] = False
        quality_score -= 0.1

    # Check 4: Consistency check: high permeability with high skin is unusual
    if test_result.permeability > 100 and abs(test_result.skin) > 20:
        warnings.append(
            "High permeability with high absolute skin factor (unusual combination)"
        )
        validation_checks["parameter_consistency"] = False
        quality_score -= 0.1
    else:
        validation_checks["parameter_consistency"] = True

    # Check 5: Boundary detection
    if test_result.boundary_distance is not None:
        if test_result.boundary_distance < 10:
            warnings.append(
                f"Boundary very close: {test_result.boundary_distance:.1f} ft (may be wellbore effect)"
            )
            validation_checks["boundary_reasonable"] = False
            quality_score -= 0.05
        elif test_result.boundary_distance > 10000:
            warnings.append(
                f"Boundary very far: {test_result.boundary_distance:.1f} ft (may be test limit)"
            )
            validation_checks["boundary_reasonable"] = False
            quality_score -= 0.05
        else:
            validation_checks["boundary_reasonable"] = True
    else:
        validation_checks["boundary_reasonable"] = True

    quality_score = max(0.0, min(1.0, quality_score))
    is_valid = len(errors) == 0 and quality_score >= 0.5

    return WellTestValidationResult(
        is_valid=is_valid,
        warnings=warnings,
        errors=errors,
        quality_score=quality_score,
        validation_checks=validation_checks,
    )


def validate_and_analyze_well_test(
    time: np.ndarray,
    pressure: np.ndarray,
    test_type: str = "buildup",
    production_rate: float | None = None,
    production_time: float | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Validate well test data and analyze if valid.

    Complete workflow that validates test data, analyzes if valid,
    and validates the results.

    Parameters
    ----------
    time : np.ndarray
        Time array (hours)
    pressure : np.ndarray
        Pressure array (psi)
    test_type : str
        Test type ('buildup' or 'drawdown') (default: 'buildup')
    production_rate : float, optional
        Production rate (required for buildup tests)
    production_time : float, optional
        Production time before shut-in (required for buildup tests)
    **kwargs
        Additional parameters for test analysis

    Returns
    -------
    dict
        Dictionary with:
        - data_validation: WellTestValidationResult for data
        - test_result: WellTestResult (if data is valid)
        - results_validation: WellTestValidationResult for results (if analyzed)
        - is_valid: Overall validity

    Examples
    --------
    >>> import numpy as np
    >>> time = np.array([0.1, 0.5, 1, 2, 5, 10, 24])
    >>> pressure = np.array([3000, 3200, 3400, 3600, 3800, 3900, 3950])
    >>> result = validate_and_analyze_well_test(
    ...     time, pressure, test_type='buildup',
    ...     production_rate=1000, production_time=720
    ... )
    >>> print(f"Overall valid: {result['is_valid']}")
    >>> if result['test_result']:
    ...     print(f"Permeability: {result['test_result'].permeability:.2f} md")
    """
    logger.info(f"Validating and analyzing {test_type} test")

    # Step 1: Validate data
    data_validation = validate_well_test_data(
        time, pressure, test_type, production_rate, production_time
    )

    test_result = None
    results_validation = None

    # Step 2: Analyze if data is valid
    if data_validation.is_valid:
        try:
            if test_type == "buildup":
                if production_rate is None or production_time is None:
                    raise ValueError(
                        "production_rate and production_time required for buildup test"
                    )
                test_result = analyze_buildup_test(
                    time, pressure, production_rate, production_time, **kwargs
                )
            elif test_type == "drawdown":
                test_result = analyze_drawdown_test(time, pressure, **kwargs)
            else:
                raise ValueError(f"Unknown test type: {test_type}")

            # Step 3: Validate results
            results_validation = validate_well_test_results(test_result)

        except Exception as e:
            logger.error(f"Test analysis failed: {e}")
            data_validation.errors.append(f"Analysis failed: {str(e)}")
            data_validation.is_valid = False

    is_valid = data_validation.is_valid and (
        results_validation is None or results_validation.is_valid
    )

    return {
        "data_validation": data_validation,
        "test_result": test_result,
        "results_validation": results_validation,
        "is_valid": is_valid,
    }
