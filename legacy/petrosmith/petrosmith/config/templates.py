"""
Configuration template generator
"""

from pathlib import Path
from typing import Literal


TEMPLATES = {
    "basic": """# PetroSmith Config-Driven Analysis
# Basic Template

project:
  name: "My Analysis"
  description: "Description of analysis"
  output_dir: "./results"

data:
  input_file: "data.csv"
  x_column: "x"
  y_column: "y"
  z_column: "value"
  delimiter: ","

preprocessing:
  remove_outliers: false
  transform: null  # Options: log, boxcox, normal_score, sqrt

variogram:
  n_lags: 15
  max_lag: null  # Auto-calculated if null
  models:
    - spherical
    - exponential
    - gaussian
  auto_fit: true
  fit_method: "wls"

kriging:
  method: "ordinary"  # Options: ordinary, simple, universal
  neighborhood:
    max_neighbors: 25
    min_neighbors: 3
  grid:
    x_min: 0.0
    x_max: 100.0
    y_min: 0.0
    y_max: 100.0
    resolution: 1.0

validation:
  cross_validation: true
  n_folds: 5
  metrics:
    - rmse
    - mae
    - r2

visualization:
  style: "minimalist"
  plots:
    - variogram
    - kriging_map
    - cross_validation
  colormap: "viridis"
  dpi: 300

output:
  save_predictions: true
  save_variance: true
  save_plots: true
  formats:
    - csv
    - png
""",

    "full": """# PetroSmith Config-Driven Analysis
# Full Template - All Options

project:
  name: "Comprehensive Analysis"
  description: "Full-featured geostatistical analysis"
  output_dir: "./results"
  author: "Your Name"
  tags:
    - geostats
    - kriging

data:
  input_file: "data.csv"
  x_column: "x"
  y_column: "y"
  z_column: "value"
  crs: "EPSG:4326"  # Coordinate reference system
  delimiter: ","
  skip_rows: 0

preprocessing:
  remove_outliers: true
  outlier_method: "iqr"  # Options: iqr, zscore, modified_zscore
  outlier_threshold: 3.0
  transform: "log"  # Options: log, boxcox, normal_score, sqrt
  handle_negatives: "shift"  # Options: clip, shift, remove
  interpolate_missing: false
  standardize: false

variogram:
  n_lags: 15
  max_lag: null
  lag_tolerance: 0.5
  models:
    - spherical
    - exponential
    - gaussian
    - linear
  auto_fit: true
  fit_method: "wls"  # Options: wls, ols, ml
  anisotropy: false
  directions:
    - 0
    - 45
    - 90
    - 135

kriging:
  method: "ordinary"  # Options: ordinary, simple, universal
  neighborhood:
    max_neighbors: 25
    min_neighbors: 3
    search_radius: null
    search_strategy: "circular"  # Options: circular, elliptical, octant
  grid:
    x_min: 0.0
    x_max: 100.0
    y_min: 0.0
    y_max: 100.0
    resolution: 1.0
    method: "regular"  # Options: regular, adaptive

validation:
  cross_validation: true
  n_folds: 5
  metrics:
    - rmse
    - mae
    - r2
    - mse
    - mape
  holdout_fraction: 0.0
  random_seed: 42

visualization:
  style: "minimalist"  # Options: minimalist, default, publication, presentation
  plots:
    - variogram
    - kriging_map
    - variance_map
    - cross_validation
    - histogram
    - qq_plot
    - scatter
  colormap: "viridis"
  dpi: 300
  figure_size:
    - 10
    - 8
  show_data_points: true
  contour_levels: 20

output:
  save_predictions: true
  save_variance: true
  save_plots: true
  save_model: true
  formats:
    - csv
    - npy
    - png
    - json
  compression: false
  save_config: true
""",

    "drilling": """# PetroSmith Drilling Analysis
# Drilling-Specific Template

project:
  name: "Drilling Analysis"
  description: "Wellbore and drilling parameter analysis"
  output_dir: "./drilling_results"

data:
  input_file: "drilling_data.csv"
  x_column: "easting"
  y_column: "northing"
  z_column: "pressure"  # Or mud_weight, rop, etc.
  crs: "EPSG:32610"

preprocessing:
  remove_outliers: true
  outlier_method: "iqr"
  outlier_threshold: 2.5
  transform: null

variogram:
  n_lags: 20
  models:
    - spherical
    - exponential
  auto_fit: true

kriging:
  method: "ordinary"
  neighborhood:
    max_neighbors: 30
    min_neighbors: 5
  grid:
    x_min: 500000
    x_max: 520000
    y_min: 4000000
    y_max: 4020000
    resolution: 100

validation:
  cross_validation: true
  n_folds: 5
  metrics:
    - rmse
    - mae

visualization:
  style: "publication"
  plots:
    - variogram
    - kriging_map
    - variance_map
    - cross_validation
  colormap: "jet"
  dpi: 300

output:
  save_predictions: true
  save_variance: true
  save_plots: true
  formats:
    - csv
    - png
    - geotiff

# Drilling-specific parameters
drilling:
  calculate_mud_weight: true
  pressure_gradient: 0.465  # psi/ft
  wellbore_stability: true
  drilling_parameters:
    max_safe_mud_weight: 16.0  # ppg
    min_safe_mud_weight: 9.0   # ppg
    kick_tolerance: 0.5        # ppg
""",

    "reservoir": """# PetroSmith Reservoir Analysis
# Reservoir-Specific Template

project:
  name: "Reservoir Property Analysis"
  description: "Porosity and saturation distribution"
  output_dir: "./reservoir_results"
  tags:
    - reservoir
    - volumetrics

data:
  input_file: "reservoir_data.csv"
  x_column: "x_coord"
  y_column: "y_coord"
  z_column: "porosity"  # Or permeability, saturation
  crs: "EPSG:32610"

preprocessing:
  remove_outliers: true
  outlier_method: "modified_zscore"
  outlier_threshold: 3.5
  transform: null
  standardize: false

variogram:
  n_lags: 15
  models:
    - spherical
    - exponential
    - gaussian
  auto_fit: true
  anisotropy: true
  directions:
    - 0
    - 90

kriging:
  method: "ordinary"
  neighborhood:
    max_neighbors: 25
    min_neighbors: 3
    search_strategy: "elliptical"
  grid:
    x_min: 0
    x_max: 5000
    y_min: 0
    y_max: 5000
    resolution: 50

validation:
  cross_validation: true
  n_folds: 10
  metrics:
    - rmse
    - mae
    - r2

visualization:
  style: "publication"
  plots:
    - variogram
    - kriging_map
    - variance_map
    - histogram
    - scatter
  colormap: "RdYlBu"
  dpi: 300
  show_data_points: true

output:
  save_predictions: true
  save_variance: true
  save_plots: true
  save_model: true
  formats:
    - csv
    - npy
    - png
    - geotiff
    - json

# Reservoir-specific parameters
reservoir:
  calculate_volumes: true
  porosity_data: "porosity"
  saturation_data: "sw"
  net_to_gross: 0.8
  formation_volume_factor: 1.2
"""
}


def generate_template(
    output_path: str = "config_template.yaml",
    template_type: Literal["basic", "full", "drilling", "reservoir"] = "basic"
) -> None:
    """
    Generate configuration template file
    
    Args:
        output_path: Where to save template
        template_type: Type of template to generate
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(TEMPLATES[template_type])


def list_available_templates() -> list[str]:
    """List available template types"""
    return list(TEMPLATES.keys())
