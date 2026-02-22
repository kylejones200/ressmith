"""Programmatic diagnostic plot generation for RTA analysis.

Provides first-class workflows for generating standard diagnostic plots
that engineers use daily: log-log plots, square-root time plots,
boundary-dominated flow plots, etc.
"""

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None  # type: ignore[assignment, unused-ignore]

from ressmith.workflows.diagnostics_plots import (
    calculate_flow_regime_slopes,
    generate_diagnostic_plot_data,
    identify_flow_regime_from_plots,
    prepare_log_log_data,
    prepare_sqrt_time_data,
)

logger = logging.getLogger(__name__)


def plot_log_log_diagnostic(
    time: np.ndarray | pd.Series,
    rate: np.ndarray | pd.Series,
    title: str = "Log-Log Diagnostic Plot",
    **kwargs: Any,
) -> tuple["Figure", "Axes"] | None:
    """Generate log-log diagnostic plot.

    Standard RTA diagnostic plot: log(rate) vs log(time).

    Parameters
    ----------
    time : np.ndarray or pd.Series
        Production time (days)
    rate : np.ndarray or pd.Series
        Production rate (STB/day or MCF/day)
    title : str
        Plot title (default: 'Log-Log Diagnostic Plot')
    **kwargs
        Additional plotting arguments

    Returns
    -------
    tuple[Figure, Axes] | None
        Matplotlib figure and axes, or None if matplotlib not available

    Examples
    --------
    >>> import numpy as np
    >>> time = np.array([1, 10, 30, 60, 90, 120])
    >>> rate = np.array([1000, 800, 600, 500, 450, 400])
    >>> fig, ax = plot_log_log_diagnostic(time, rate)
    """
    logger.info("Generating log-log diagnostic plot")

    if not HAS_MATPLOTLIB or plt is None:
        logger.warning("matplotlib not available. Install with: pip install matplotlib")
        return None

    # Prepare data
    plot_data = prepare_log_log_data(time, rate)

    # Create plot
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 6)))
    ax.loglog(plot_data["time"], plot_data["rate"], "o-", label="Production Data")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Rate (STB/day or MCF/day)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # Add flow regime annotation
    flow_regime = identify_flow_regime_from_plots(time, rate)
    ax.text(
        0.05,
        0.95,
        f"Flow Regime: {flow_regime}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    return fig, ax


def plot_sqrt_time_diagnostic(
    time: np.ndarray | pd.Series,
    rate: np.ndarray | pd.Series,
    title: str = "Square-Root Time Diagnostic Plot (Linear Flow)",
    **kwargs: Any,
) -> tuple["Figure", "Axes"] | None:
    """Generate square-root time diagnostic plot.

    Standard RTA diagnostic plot for linear flow: rate vs sqrt(time).

    Parameters
    ----------
    time : np.ndarray or pd.Series
        Production time (days)
    rate : np.ndarray or pd.Series
        Production rate (STB/day or MCF/day)
    title : str
        Plot title (default: 'Square-Root Time Diagnostic Plot (Linear Flow)')
    **kwargs
        Additional plotting arguments

    Returns
    -------
    tuple[Figure, Axes] | None
        Matplotlib figure and axes, or None if matplotlib not available

    Examples
    --------
    >>> import numpy as np
    >>> time = np.array([1, 10, 30, 60, 90, 120])
    >>> rate = np.array([1000, 800, 600, 500, 450, 400])
    >>> fig, ax = plot_sqrt_time_diagnostic(time, rate)
    """
    logger.info("Generating square-root time diagnostic plot")

    if not HAS_MATPLOTLIB or plt is None:
        logger.warning("matplotlib not available. Install with: pip install matplotlib")
        return None

    # Prepare data
    plot_data = prepare_sqrt_time_data(time, rate)

    # Create plot
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 6)))
    ax.plot(plot_data["sqrt_time"], plot_data["rate"], "o-", label="Production Data")
    ax.set_xlabel("Square Root of Time (√days)")
    ax.set_ylabel("Rate (STB/day or MCF/day)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add linear flow indicator
    slopes = calculate_flow_regime_slopes(time, rate)
    if not np.isnan(slopes["sqrt_time_slope"]):
        ax.text(
            0.05,
            0.95,
            f"Slope: {slopes['sqrt_time_slope']:.2f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    return fig, ax


def plot_boundary_dominated_flow(
    time: np.ndarray | pd.Series,
    rate: np.ndarray | pd.Series,
    cumulative: np.ndarray | pd.Series | None = None,
    title: str = "Boundary-Dominated Flow Diagnostic Plot",
    **kwargs: Any,
) -> tuple["Figure", "Axes"] | None:
    """Generate boundary-dominated flow diagnostic plot.

    Standard RTA diagnostic plot: rate vs cumulative production.

    Parameters
    ----------
    time : np.ndarray or pd.Series
        Production time (days)
    rate : np.ndarray or pd.Series
        Production rate (STB/day or MCF/day)
    cumulative : np.ndarray or pd.Series, optional
        Cumulative production (if None, calculated from rate)
    title : str
        Plot title (default: 'Boundary-Dominated Flow Diagnostic Plot')
    **kwargs
        Additional plotting arguments

    Returns
    -------
    tuple[Figure, Axes] | None
        Matplotlib figure and axes, or None if matplotlib not available

    Examples
    --------
    >>> import numpy as np
    >>> time = np.array([1, 10, 30, 60, 90, 120])
    >>> rate = np.array([1000, 800, 600, 500, 450, 400])
    >>> fig, ax = plot_boundary_dominated_flow(time, rate)
    """
    logger.info("Generating boundary-dominated flow diagnostic plot")

    if not HAS_MATPLOTLIB or plt is None:
        logger.warning("matplotlib not available. Install with: pip install matplotlib")
        return None

    # Calculate cumulative if not provided
    if cumulative is None:
        if isinstance(time, pd.Series):
            time = time.values
        if isinstance(rate, pd.Series):
            rate = rate.values

        time_deltas = np.diff(np.concatenate([[0], time]))
        cumulative = np.cumsum(rate * time_deltas)

    # Create plot
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 6)))
    ax.plot(cumulative, rate, "o-", label="Production Data")
    ax.set_xlabel("Cumulative Production (STB or MCF)")
    ax.set_ylabel("Rate (STB/day or MCF/day)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return fig, ax


def plot_linear_flow_diagnostic(
    time: np.ndarray | pd.Series,
    rate: np.ndarray | pd.Series,
    title: str = "Linear Flow Diagnostic Plot",
    **kwargs: Any,
) -> tuple["Figure", "Axes"] | None:
    """Generate linear flow diagnostic plot.

    Standard RTA diagnostic plot: rate vs 1/sqrt(time) for linear flow.

    Parameters
    ----------
    time : np.ndarray or pd.Series
        Production time (days)
    rate : np.ndarray or pd.Series
        Production rate (STB/day or MCF/day)
    title : str
        Plot title (default: 'Linear Flow Diagnostic Plot')
    **kwargs
        Additional plotting arguments

    Returns
    -------
    tuple[Figure, Axes] | None
        Matplotlib figure and axes, or None if matplotlib not available

    Examples
    --------
    >>> import numpy as np
    >>> time = np.array([1, 10, 30, 60, 90, 120])
    >>> rate = np.array([1000, 800, 600, 500, 450, 400])
    >>> fig, ax = plot_linear_flow_diagnostic(time, rate)
    """
    logger.info("Generating linear flow diagnostic plot")

    if not HAS_MATPLOTLIB or plt is None:
        logger.warning("matplotlib not available. Install with: pip install matplotlib")
        return None

    # Convert to arrays
    if isinstance(time, pd.Series):
        time = time.values
    if isinstance(rate, pd.Series):
        rate = rate.values

    valid_mask = (time > 0) & (rate > 0)
    time_valid = time[valid_mask]
    rate_valid = rate[valid_mask]

    # Calculate 1/sqrt(time)
    inv_sqrt_time = 1.0 / np.sqrt(time_valid)

    # Create plot
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 6)))
    ax.plot(inv_sqrt_time, rate_valid, "o-", label="Production Data")
    ax.set_xlabel("1 / √Time (1/√days)")
    ax.set_ylabel("Rate (STB/day or MCF/day)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add linear flow indicator
    slopes = calculate_flow_regime_slopes(time, rate)
    if not np.isnan(slopes["sqrt_time_slope"]):
        ax.text(
            0.05,
            0.95,
            f"Linear Flow Slope: {slopes['sqrt_time_slope']:.2f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    return fig, ax


def generate_all_diagnostic_plots(
    time: np.ndarray | pd.Series,
    rate: np.ndarray | pd.Series,
    cumulative: np.ndarray | pd.Series | None = None,
    save_path: str | None = None,
    **kwargs: Any,
) -> dict[str, tuple["Figure", "Axes"]]:
    """Generate all standard diagnostic plots for RTA analysis.

    Creates a comprehensive set of diagnostic plots:
    - Log-log plot
    - Square-root time plot
    - Linear flow plot
    - Boundary-dominated flow plot

    Parameters
    ----------
    time : np.ndarray or pd.Series
        Production time (days)
    rate : np.ndarray or pd.Series
        Production rate (STB/day or MCF/day)
    cumulative : np.ndarray or pd.Series, optional
        Cumulative production
    save_path : str, optional
        Path to save plots (if None, plots are not saved)
    **kwargs
        Additional plotting arguments

    Returns
    -------
    dict
        Dictionary mapping plot names to (Figure, Axes) tuples

    Examples
    --------
    >>> import numpy as np
    >>> time = np.array([1, 10, 30, 60, 90, 120])
    >>> rate = np.array([1000, 800, 600, 500, 450, 400])
    >>> plots = generate_all_diagnostic_plots(time, rate)
    >>> print(f"Generated {len(plots)} diagnostic plots")
    """
    logger.info("Generating all diagnostic plots")

    plots = {}

    # Log-log plot
    fig_log, ax_log = plot_log_log_diagnostic(time, rate, **kwargs)
    if fig_log is not None:
        plots["log_log"] = (fig_log, ax_log)
        if save_path:
            fig_log.savefig(f"{save_path}_log_log.png", dpi=300, bbox_inches="tight")

    # Square-root time plot
    fig_sqrt, ax_sqrt = plot_sqrt_time_diagnostic(time, rate, **kwargs)
    if fig_sqrt is not None:
        plots["sqrt_time"] = (fig_sqrt, ax_sqrt)
        if save_path:
            fig_sqrt.savefig(f"{save_path}_sqrt_time.png", dpi=300, bbox_inches="tight")

    # Linear flow plot
    fig_linear, ax_linear = plot_linear_flow_diagnostic(time, rate, **kwargs)
    if fig_linear is not None:
        plots["linear_flow"] = (fig_linear, ax_linear)
        if save_path:
            fig_linear.savefig(f"{save_path}_linear_flow.png", dpi=300, bbox_inches="tight")

    # Boundary-dominated flow plot
    fig_bdf, ax_bdf = plot_boundary_dominated_flow(time, rate, cumulative, **kwargs)
    if fig_bdf is not None:
        plots["boundary_dominated"] = (fig_bdf, ax_bdf)
        if save_path:
            fig_bdf.savefig(f"{save_path}_boundary_dominated.png", dpi=300, bbox_inches="tight")

    logger.info(f"Generated {len(plots)} diagnostic plots")
    return plots


def plot_diagnostic_with_flow_regime_identification(
    data: pd.DataFrame,
    time_col: str | None = None,
    rate_col: str = "oil",
    plot_type: str = "all",
    **kwargs: Any,
) -> dict[str, Any]:
    """Generate diagnostic plots with automated flow regime identification.

    Comprehensive workflow that generates diagnostic plots and identifies
    flow regimes automatically.

    Parameters
    ----------
    data : pd.DataFrame
        Production data
    time_col : str, optional
        Time column name (if None, uses index)
    rate_col : str
        Rate column name (default: 'oil')
    plot_type : str
        Type of plots to generate: 'log_log', 'sqrt_time', 'linear_flow',
        'boundary_dominated', 'all' (default: 'all')
    **kwargs
        Additional plotting arguments

    Returns
    -------
    dict
        Dictionary with:
        - plots: Dictionary of generated plots
        - flow_regime: Identified flow regime
        - slopes: Calculated slopes
        - plot_data: Prepared plot data

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> time = np.array([1, 10, 30, 60, 90, 120])
    >>> rate = np.array([1000, 800, 600, 500, 450, 400])
    >>> df = pd.DataFrame({'time': time, 'oil': rate})
    >>> result = plot_diagnostic_with_flow_regime_identification(df, time_col='time')
    >>> print(f"Flow regime: {result['flow_regime']}")
    """
    logger.info("Generating diagnostic plots with flow regime identification")

    # Extract time and rate
    if time_col is None:
        if isinstance(data.index, pd.DatetimeIndex):
            time = np.array(
                [(data.index[i] - data.index[0]).days for i in range(len(data))]
            )
        else:
            time = np.arange(len(data))
    else:
        time = data[time_col].values

    rate = data[rate_col].values

    # Generate plot data
    plot_data = generate_diagnostic_plot_data(time, rate, plot_type=plot_type)

    # Generate plots
    plots = {}
    if plot_type in ("log_log", "all"):
        fig, ax = plot_log_log_diagnostic(time, rate, **kwargs)
        if fig is not None:
            plots["log_log"] = (fig, ax)

    if plot_type in ("sqrt_time", "all"):
        fig, ax = plot_sqrt_time_diagnostic(time, rate, **kwargs)
        if fig is not None:
            plots["sqrt_time"] = (fig, ax)

    if plot_type in ("linear_flow", "all"):
        fig, ax = plot_linear_flow_diagnostic(time, rate, **kwargs)
        if fig is not None:
            plots["linear_flow"] = (fig, ax)

    if plot_type in ("boundary_dominated", "all"):
        cumulative = None
        if "cumulative" in data.columns:
            cumulative = data["cumulative"].values
        fig, ax = plot_boundary_dominated_flow(time, rate, cumulative, **kwargs)
        if fig is not None:
            plots["boundary_dominated"] = (fig, ax)

    return {
        "plots": plots,
        "flow_regime": plot_data["flow_regime"],
        "slopes": plot_data["slopes"],
        "plot_data": plot_data,
    }
