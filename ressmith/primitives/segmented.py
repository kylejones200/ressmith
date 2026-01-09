"""
Segmented decline curve primitives.

Functions for fitting and predicting segmented decline curves.
"""

from typing import Literal, Optional

import numpy as np
import pandas as pd

from ressmith.objects.domain import DeclineSegment
from ressmith.primitives.decline import (
    arps_exponential,
    arps_harmonic,
    arps_hyperbolic,
    fit_arps_exponential,
    fit_arps_harmonic,
    fit_arps_hyperbolic,
)


def fit_segment(
    t: np.ndarray,
    q: np.ndarray,
    kind: Literal["exponential", "harmonic", "hyperbolic"],
) -> dict[str, float]:
    """
    Fit a single segment to production data.

    Parameters
    ----------
    t : np.ndarray
        Time array (days from start)
    q : np.ndarray
        Rate array
    kind : str
        ARPS decline type: 'exponential', 'harmonic', or 'hyperbolic'

    Returns
    -------
    dict
        Fitted parameters: {'qi': float, 'di': float, 'b': float}
    """
    if len(t) < 3:
        raise ValueError(f"Segment must have at least 3 data points, got {len(t)}")

    if kind == "exponential":
        params = fit_arps_exponential(t, q)
        params["b"] = 0.0
    elif kind == "harmonic":
        params = fit_arps_harmonic(t, q)
        params["b"] = 1.0
    elif kind == "hyperbolic":
        params = fit_arps_hyperbolic(t, q)
    else:
        raise ValueError(f"Unknown decline kind: {kind}")

    return params


def predict_segment(
    t: np.ndarray,
    params: dict[str, float],
    kind: Literal["exponential", "harmonic", "hyperbolic"],
) -> np.ndarray:
    """
    Predict rates for a segment.

    Parameters
    ----------
    t : np.ndarray
        Time array
    params : dict
        ARPS parameters: {'qi': float, 'di': float, 'b': float}
    kind : str
        ARPS decline type

    Returns
    -------
    np.ndarray
        Predicted rates
    """
    qi = params["qi"]
    di = params["di"]
    b = params.get("b", 0.0)

    if kind == "exponential":
        return arps_exponential(t, qi, di)
    elif kind == "harmonic":
        return arps_harmonic(t, qi, di)
    elif kind == "hyperbolic":
        return arps_hyperbolic(t, qi, di, b)
    else:
        raise ValueError(f"Unknown decline kind: {kind}")


def check_continuity(
    segment1: DeclineSegment,
    segment2: DeclineSegment,
    t_transition: float,
    tolerance: float = 0.01,
) -> tuple[bool, Optional[str]]:
    """
    Check continuity between two segments at transition point.

    Parameters
    ----------
    segment1 : DeclineSegment
        First segment
    segment2 : DeclineSegment
        Second segment
    t_transition : float
        Time of transition (relative to segment1 start)
    tolerance : float
        Relative tolerance for continuity check (default: 0.01 = 1%)

    Returns
    -------
    tuple
        (is_continuous, error_message)
    """
    # Calculate rate at end of segment1
    t1_local = t_transition - segment1.t_start
    q1_end = predict_segment(
        np.array([t1_local]), segment1.parameters, segment1.kind
    )[0]

    # Calculate rate at start of segment2
    t2_local = 0.0
    q2_start = predict_segment(
        np.array([t2_local]), segment2.parameters, segment2.kind
    )[0]

    # Check continuity with tolerance
    abs_tolerance = max(q1_end, q2_start) * tolerance
    if abs(q1_end - q2_start) > abs_tolerance:
        return False, (
            f"Rate discontinuity at transition: "
            f"segment1 end={q1_end:.2f}, segment2 start={q2_start:.2f}"
        )

    return True, None

