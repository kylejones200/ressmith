"""Tests for fit diagnostics."""

import numpy as np
import pytest

from ressmith.primitives.diagnostics import (
    check_fit_quality,
    compute_diagnostics,
    compute_fit_metrics,
)


def test_compute_fit_metrics_perfect_fit():
    """Test metrics computation with perfect fit."""
    q_obs = np.array([100.0, 90.0, 80.0, 70.0, 60.0])
    q_pred = q_obs.copy()

    metrics = compute_fit_metrics(q_obs, q_pred)

    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["mape"] == 0.0
    assert metrics["r_squared"] == 1.0


def test_compute_fit_metrics_with_error():
    """Test metrics computation with prediction error."""
    q_obs = np.array([100.0, 90.0, 80.0, 70.0, 60.0])
    q_pred = np.array([105.0, 85.0, 85.0, 65.0, 65.0])

    metrics = compute_fit_metrics(q_obs, q_pred)

    assert metrics["rmse"] > 0.0
    assert metrics["mae"] > 0.0
    assert metrics["mape"] > 0.0
    assert 0.0 <= metrics["r_squared"] <= 1.0


def test_check_fit_quality():
    """Test quality checks."""
    q_obs = np.array([100.0, 90.0, 80.0, 70.0, 60.0])
    q_pred = np.array([100.0, 90.0, 80.0, 70.0, 60.0])
    residuals = q_obs - q_pred

    flags = check_fit_quality(q_obs, q_pred, residuals)

    assert "has_negative_predictions" in flags
    assert "mostly_declining" in flags
    assert "low_bias" in flags
    assert "few_outliers" in flags
    assert "good_r_squared" in flags


def test_compute_diagnostics():
    """Test comprehensive diagnostics computation."""
    q_obs = np.array([100.0, 90.0, 80.0, 70.0, 60.0])
    q_pred = np.array([100.0, 90.0, 80.0, 70.0, 60.0])

    diag = compute_diagnostics(q_obs, q_pred)

    assert diag.rmse == 0.0
    assert diag.mae == 0.0
    assert diag.r_squared == 1.0
    assert len(diag.residuals) == len(q_obs)
    assert len(diag.quality_flags) > 0
    # Perfect fit should have no warnings (except possibly outlier check if std=0)
    # With perfect fit, residuals are all zero, so outlier check may behave differently
    assert "has_negative_predictions" in diag.quality_flags
    assert diag.quality_flags["has_negative_predictions"] is False


def test_compute_diagnostics_with_warnings():
    """Test diagnostics with poor fit."""
    q_obs = np.array([100.0, 90.0, 80.0, 70.0, 60.0])
    q_pred = np.array([50.0, 50.0, 50.0, 50.0, 50.0])  # Poor fit

    diag = compute_diagnostics(q_obs, q_pred)

    assert diag.rmse > 0.0
    assert diag.r_squared < 0.7  # Should trigger warning
    assert len(diag.warnings) > 0


def test_compute_diagnostics_negative_predictions():
    """Test diagnostics with negative predictions."""
    q_obs = np.array([100.0, 90.0, 80.0, 70.0, 60.0])
    q_pred = np.array([100.0, 90.0, -10.0, 70.0, 60.0])  # Negative value

    diag = compute_diagnostics(q_obs, q_pred)

    assert diag.quality_flags["has_negative_predictions"] is True
    assert "negative" in " ".join(diag.warnings).lower()


def test_diagnostics_to_dict():
    """Test diagnostics serialization."""
    q_obs = np.array([100.0, 90.0, 80.0, 70.0, 60.0])
    q_pred = np.array([100.0, 90.0, 80.0, 70.0, 60.0])

    diag = compute_diagnostics(q_obs, q_pred)
    diag_dict = diag.to_dict()

    assert "rmse" in diag_dict
    assert "mae" in diag_dict
    assert "mape" in diag_dict
    assert "r_squared" in diag_dict
    assert "quality_flags" in diag_dict
    assert "warnings" in diag_dict

