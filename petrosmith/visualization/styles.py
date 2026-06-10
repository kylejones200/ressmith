"""
Minimalist plotting styles for PetroSmith

Clean, professional visualizations without chart junk.
"""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def set_minimalist_style():
    """
    Set global matplotlib style for minimalist, clean plots.
    
    Removes chart junk:
    - No gridlines
    - Only left and bottom spines
    - Clean, readable fonts
    - Descriptive titles instead of axis labels
    """
    plt.style.use('default')
    
    # Set clean defaults
    plt.rcParams.update({
        # Figure
        'figure.facecolor': 'white',
        'figure.edgecolor': 'white',
        
        # Axes
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.axisbelow': True,
        
        # Grid (disabled by default)
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'grid.linestyle': '-',
        
        # Ticks
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 0,
        'ytick.minor.size': 0,
        
        # Font
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        
        # Legend
        'legend.frameon': False,
        'legend.loc': 'best',
        
        # Lines
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        
        # Save
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',
    })


def apply_clean_axes(ax: Axes, title: str = None, show_grid: bool = False) -> Axes:
    """
    Apply minimalist styling to a matplotlib axes object.
    
    Args:
        ax: Matplotlib axes object
        title: Descriptive title (should describe what is being plotted)
        show_grid: Whether to show subtle gridlines (default: False)
        
    Returns:
        Modified axes object
        
    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot(x, y)
        >>> apply_clean_axes(ax, "Oil Production Rate vs Time (2020-2025)")
    """
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Keep left and bottom spines clean
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    # Remove gridlines unless explicitly requested
    if show_grid:
        ax.grid(True, alpha=0.3, linewidth=0.5, linestyle='-', which='major')
        ax.set_axisbelow(True)
    else:
        ax.grid(False)
    
    # Set title if provided
    if title:
        ax.set_title(title, fontsize=12, fontweight='normal', pad=10)
    
    # Tick parameters
    ax.tick_params(
        axis='both',
        which='major',
        direction='out',
        length=4,
        width=1.0,
        colors='black'
    )
    
    # Remove minor ticks
    ax.tick_params(axis='both', which='minor', length=0)
    
    return ax


def create_minimalist_figure(nrows: int = 1, ncols: int = 1, figsize: tuple = None):
    """
    Create a figure with minimalist styling applied.
    
    Args:
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        figsize: Figure size (width, height) in inches
        
    Returns:
        Figure and axes objects
        
    Example:
        >>> fig, ax = create_minimalist_figure(figsize=(10, 6))
        >>> ax.plot(x, y)
        >>> apply_clean_axes(ax, "Pressure vs Depth (Well A-1)")
    """
    if figsize is None:
        figsize = (8, 6) if nrows == 1 and ncols == 1 else (12, 8)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor='white')
    
    # Apply clean styling to all axes
    if nrows == 1 and ncols == 1:
        apply_clean_axes(axes)
    else:
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        for ax in axes_flat:
            apply_clean_axes(ax)
    
    return fig, axes


def remove_axis_labels(ax: Axes) -> Axes:
    """
    Remove x and y axis labels (use when title is descriptive enough).
    
    Args:
        ax: Matplotlib axes object
        
    Returns:
        Modified axes object
    """
    ax.set_xlabel('')
    ax.set_ylabel('')
    return ax


def set_descriptive_title(ax: Axes, x_var: str, y_var: str, units_x: str = None, units_y: str = None) -> Axes:
    """
    Set a descriptive title that makes axis labels unnecessary.
    
    Args:
        ax: Matplotlib axes object
        x_var: X variable name
        y_var: Y variable name  
        units_x: X units (optional)
        units_y: Y units (optional)
        
    Returns:
        Modified axes object
        
    Example:
        >>> set_descriptive_title(ax, "Time", "Oil Rate", "days", "STB/day")
        # Sets title: "Oil Rate (STB/day) vs Time (days)"
    """
    title_parts = [y_var]
    if units_y:
        title_parts.append(f"({units_y})")
    title_parts.append("vs")
    title_parts.append(x_var)
    if units_x:
        title_parts.append(f"({units_x})")
    
    title = " ".join(title_parts)
    ax.set_title(title, fontsize=12, fontweight='normal', pad=10)
    
    # Remove axis labels since title is descriptive
    remove_axis_labels(ax)
    
    return ax
