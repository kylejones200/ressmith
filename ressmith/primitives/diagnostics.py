"""
Fit quality diagnostics for decline curve models.

This module provides comprehensive quality assessment for decline curve fits,
including metrics computation and quality checks.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FitDiagnostics:
    """Result of fit quality diagnostics.

    Attributes
    ----------
    rmse : float
        Root mean squared error
    mae : float
        Mean absolute error
    mape : float
        Mean absolute percentage error
    r_squared : float
        R² score
    residuals : np.ndarray
        Residuals (observed - predicted)
    quality_flags : dict
        Dictionary of quality check results
    warnings : list
        List of warning messages
    """

    rmse: float
    mae: float
    mape: float
    r_squared: float
    residuals: np.ndarray
    quality_flags: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert diagnostics to dictionary."""
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "r_squared": self.r_squared,
            "quality_flags": self.quality_flags,
            "warnings": self.warnings,
        }


def compute_fit_metrics(
    q_obs: np.ndarray,
    q_pred: np.ndarray,
) -> dict[str, float]:
    """
    Compute rate-based error metrics.

    Parameters
    ----------
    q_obs : np.ndarray
        Observed rates
    q_pred : np.ndarray
        Predicted rates

    Returns
    -------
    dict
        Dictionary of metrics (RMSE, MAE, MAPE, R²)
    """
    residuals = q_obs - q_pred

    # RMSE
    rmse = float(np.sqrt(np.mean(residuals**2)))

    # MAE
    mae = float(np.mean(np.abs(residuals)))

    # MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    mask = q_obs > 0
    if mask.any():
        mape = float(np.mean(np.abs(residuals[mask] / q_obs[mask])) * 100)
    else:
        mape = float(np.inf)

    # R²
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((q_obs - q_obs.mean()) ** 2)
    r_squared = float(1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0)

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r_squared": r_squared,
    }


def check_fit_quality(
    q_obs: np.ndarray,
    q_pred: np.ndarray,
    residuals: np.ndarray,
) -> dict[str, bool]:
    """
    Perform quality checks on fit.

    Parameters
    ----------
    q_obs : np.ndarray
        Observed rates
    q_pred : np.ndarray
        Predicted rates
    residuals : np.ndarray
        Residuals

    Returns
    -------
    dict
        Dictionary of quality flags
    """
    flags: dict[str, bool] = {}

    flags["has_negative_predictions"] = bool(np.any(q_pred < 0))

    if len(q_pred) > 1:
        diff = np.diff(q_pred)
        declining_ratio = np.sum(diff < 0) / len(diff)
        flags["mostly_declining"] = declining_ratio > 0.7

    if len(residuals) > 3:
        mean_residual = np.mean(residuals)
        flags["low_bias"] = abs(mean_residual) < np.std(residuals) * 0.5

        std_residual = np.std(residuals)
        outliers = np.abs(residuals) > 3 * std_residual
        flags["few_outliers"] = np.sum(outliers) < len(residuals) * 0.1

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((q_obs - q_obs.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    flags["good_r_squared"] = r_squared > 0.7

    return flags


def compute_diagnostics(
    q_obs: np.ndarray,
    q_pred: np.ndarray,
) -> FitDiagnostics:
    """
    Compute comprehensive fit diagnostics.

    Parameters
    ----------
    q_obs : np.ndarray
        Observed rates
    q_pred : np.ndarray
        Predicted rates

    Returns
    -------
    FitDiagnostics
        Diagnostics result
    """
    if len(q_obs) != len(q_pred):
        raise ValueError("Observed and predicted arrays must have same length")

    residuals = q_obs - q_pred
    metrics = compute_fit_metrics(q_obs, q_pred)
    quality_flags = check_fit_quality(q_obs, q_pred, residuals)

    warnings: list[str] = []
    if quality_flags.get("has_negative_predictions", False):
        warnings.append("Model predicts negative rates")
    if not quality_flags.get("mostly_declining", True):
        warnings.append("Model does not show monotonic decline")
    if not quality_flags.get("low_bias", True):
        warnings.append("Model shows systematic bias")
    if not quality_flags.get("good_r_squared", False):
        warnings.append(f"Low R² score: {metrics['r_squared']:.3f}")

    return FitDiagnostics(
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        mape=metrics["mape"],
        r_squared=metrics["r_squared"],
        residuals=residuals,
        quality_flags=quality_flags,
        warnings=warnings,
    )
