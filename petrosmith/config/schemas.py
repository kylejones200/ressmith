"""
Pydantic schemas for config-driven workflow
"""

from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from pathlib import Path


class ProjectConfig(BaseModel):
    """Project metadata configuration"""
    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    output_dir: str = Field("./results", description="Output directory for results")
    author: Optional[str] = Field(None, description="Analysis author")
    tags: List[str] = Field(default_factory=list, description="Project tags")


class DataConfig(BaseModel):
    """Data input configuration"""
    input_file: str = Field(..., description="Path to input data file")
    x_column: str = Field(..., description="Column name for X coordinates")
    y_column: str = Field(..., description="Column name for Y coordinates")
    z_column: str = Field(..., description="Column name for values")
    crs: Optional[str] = Field(None, description="Coordinate reference system (e.g., 'EPSG:4326')")
    delimiter: str = Field(",", description="CSV delimiter")
    skip_rows: int = Field(0, description="Number of rows to skip")
    
    @field_validator('input_file')
    @classmethod
    def validate_input_file(cls, v: str) -> str:
        """Input file path. Existence is enforced at pipeline load_data() so that
        config templates can be validated before the data file is created."""
        return v


class PreprocessingConfig(BaseModel):
    """Data preprocessing configuration"""
    remove_outliers: bool = Field(False, description="Remove outliers from data")
    outlier_method: Literal["iqr", "zscore", "modified_zscore"] = Field(
        "iqr", 
        description="Outlier detection method"
    )
    outlier_threshold: float = Field(3.0, description="Outlier threshold")
    transform: Optional[Literal["log", "boxcox", "normal_score", "sqrt"]] = Field(
        None, 
        description="Data transformation method"
    )
    handle_negatives: Literal["clip", "shift", "remove"] = Field(
        "shift", 
        description="How to handle negative values for log transform"
    )
    interpolate_missing: bool = Field(False, description="Interpolate missing values")
    standardize: bool = Field(False, description="Standardize data (mean=0, std=1)")


class VariogramConfig(BaseModel):
    """Variogram analysis configuration"""
    n_lags: int = Field(15, ge=5, le=50, description="Number of lag bins")
    max_lag: Optional[float] = Field(None, description="Maximum lag distance")
    lag_tolerance: float = Field(0.5, description="Lag tolerance as fraction of lag width")
    models: List[Literal["spherical", "exponential", "gaussian", "linear", "power"]] = Field(
        default_factory=lambda: ["spherical", "exponential", "gaussian"],
        description="Variogram models to test"
    )
    auto_fit: bool = Field(True, description="Automatically fit best model")
    fit_method: Literal["wls", "ols", "ml"] = Field(
        "wls", 
        description="Fitting method: weighted least squares, ordinary least squares, or maximum likelihood"
    )
    anisotropy: bool = Field(False, description="Consider anisotropic variogram")
    directions: List[float] = Field(
        default_factory=lambda: [0, 45, 90, 135],
        description="Directions for anisotropic analysis (degrees)"
    )


class NeighborhoodConfig(BaseModel):
    """Kriging neighborhood configuration"""
    max_neighbors: int = Field(25, ge=1, le=100, description="Maximum number of neighbors")
    min_neighbors: int = Field(3, ge=1, description="Minimum number of neighbors")
    search_radius: Optional[float] = Field(None, description="Search radius for neighbors")
    search_strategy: Literal["circular", "elliptical", "octant"] = Field(
        "circular",
        description="Neighbor search strategy"
    )


class GridConfig(BaseModel):
    """Grid configuration for interpolation"""
    x_min: float = Field(..., description="Minimum X coordinate")
    x_max: float = Field(..., description="Maximum X coordinate")
    y_min: float = Field(..., description="Minimum Y coordinate")
    y_max: float = Field(..., description="Maximum Y coordinate")
    resolution: float = Field(0.1, gt=0, description="Grid resolution")
    method: Literal["regular", "adaptive"] = Field("regular", description="Grid generation method")
    
    @field_validator('x_max')
    @classmethod
    def validate_x_range(cls, v: float, info) -> float:
        """Validate X range"""
        if 'x_min' in info.data and v <= info.data['x_min']:
            raise ValueError("x_max must be greater than x_min")
        return v
    
    @field_validator('y_max')
    @classmethod
    def validate_y_range(cls, v: float, info) -> float:
        """Validate Y range"""
        if 'y_min' in info.data and v <= info.data['y_min']:
            raise ValueError("y_max must be greater than y_min")
        return v


class KrigingConfig(BaseModel):
    """Kriging interpolation configuration"""
    method: Literal["ordinary", "simple", "universal"] = Field(
        "ordinary",
        description="Kriging method"
    )
    neighborhood: NeighborhoodConfig = Field(
        default_factory=NeighborhoodConfig,
        description="Neighborhood search parameters"
    )
    grid: GridConfig = Field(..., description="Interpolation grid")
    drift_terms: Optional[List[str]] = Field(
        None,
        description="Drift terms for universal kriging (e.g., ['x', 'y', 'x*y'])"
    )
    simple_kriging_mean: Optional[float] = Field(
        None,
        description="Known mean for simple kriging"
    )


class ValidationConfig(BaseModel):
    """Validation configuration"""
    cross_validation: bool = Field(True, description="Perform cross-validation")
    n_folds: int = Field(5, ge=2, le=20, description="Number of CV folds")
    metrics: List[Literal["rmse", "mae", "r2", "mse", "mape"]] = Field(
        default_factory=lambda: ["rmse", "mae", "r2"],
        description="Validation metrics to compute"
    )
    holdout_fraction: float = Field(
        0.0, 
        ge=0.0, 
        le=0.5, 
        description="Fraction of data to hold out for validation"
    )
    random_seed: Optional[int] = Field(42, description="Random seed for reproducibility")


class VisualizationConfig(BaseModel):
    """Visualization configuration"""
    style: Literal["minimalist", "default", "publication", "presentation"] = Field(
        "minimalist",
        description="Plot style"
    )
    plots: List[Literal[
        "variogram", 
        "kriging_map", 
        "variance_map",
        "cross_validation", 
        "histogram", 
        "qq_plot",
        "scatter",
        "semivariogram_cloud"
    ]] = Field(
        default_factory=lambda: ["variogram", "kriging_map", "cross_validation"],
        description="Plots to generate"
    )
    colormap: str = Field("viridis", description="Matplotlib colormap")
    dpi: int = Field(300, ge=72, le=600, description="Figure DPI")
    figure_size: tuple[float, float] = Field((10, 8), description="Figure size (width, height)")
    show_data_points: bool = Field(True, description="Show data points on maps")
    contour_levels: int = Field(20, description="Number of contour levels")


class OutputConfig(BaseModel):
    """Output configuration"""
    save_predictions: bool = Field(True, description="Save prediction grid")
    save_variance: bool = Field(True, description="Save variance grid")
    save_plots: bool = Field(True, description="Save plots")
    save_model: bool = Field(True, description="Save fitted model")
    formats: List[Literal["csv", "npy", "geotiff", "png", "pdf", "json"]] = Field(
        default_factory=lambda: ["csv", "png"],
        description="Output file formats"
    )
    compression: bool = Field(False, description="Compress output files")
    save_config: bool = Field(True, description="Save copy of config file")


class DrillingConfig(BaseModel):
    """Drilling-specific analysis configuration"""
    calculate_mud_weight: bool = Field(False, description="Calculate mud weight requirements")
    pressure_gradient: Optional[float] = Field(None, description="Formation pressure gradient (psi/ft)")
    wellbore_stability: bool = Field(False, description="Perform wellbore stability analysis")
    drilling_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional drilling parameters"
    )


class ReservoirConfig(BaseModel):
    """Reservoir engineering configuration"""
    calculate_volumes: bool = Field(False, description="Calculate reservoir volumes")
    porosity_data: Optional[str] = Field(None, description="Porosity data column")
    saturation_data: Optional[str] = Field(None, description="Saturation data column")
    net_to_gross: float = Field(1.0, ge=0.0, le=1.0, description="Net-to-gross ratio")
    formation_volume_factor: float = Field(1.0, description="Formation volume factor")


class AnalysisConfig(BaseModel):
    """Complete analysis configuration"""
    project: ProjectConfig = Field(..., description="Project metadata")
    data: DataConfig = Field(..., description="Data configuration")
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig,
        description="Preprocessing configuration"
    )
    variogram: VariogramConfig = Field(
        default_factory=VariogramConfig,
        description="Variogram configuration"
    )
    kriging: KrigingConfig = Field(..., description="Kriging configuration")
    validation: ValidationConfig = Field(
        default_factory=ValidationConfig,
        description="Validation configuration"
    )
    visualization: VisualizationConfig = Field(
        default_factory=VisualizationConfig,
        description="Visualization configuration"
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Output configuration"
    )
    drilling: Optional[DrillingConfig] = Field(
        None,
        description="Drilling-specific configuration"
    )
    reservoir: Optional[ReservoirConfig] = Field(
        None,
        description="Reservoir-specific configuration"
    )
    
    class Config:
        """Pydantic model configuration"""
        json_schema_extra = {
            "example": {
                "project": {
                    "name": "Example Analysis",
                    "output_dir": "./results"
                },
                "data": {
                    "input_file": "data.csv",
                    "x_column": "x",
                    "y_column": "y",
                    "z_column": "value"
                },
                "kriging": {
                    "method": "ordinary",
                    "grid": {
                        "x_min": 0,
                        "x_max": 100,
                        "y_min": 0,
                        "y_max": 100,
                        "resolution": 1.0
                    }
                }
            }
        }
