# PetroSmith Configuration Examples

This directory contains example configuration files demonstrating PetroSmith's config-driven workflow.

## Quick Start

1. **Generate a template:**
   ```bash
   petrosmith init --output my_config.yaml
   ```

2. **Run an example:**
   ```bash
   petrosmith run --config examples/configs/basic_example.yaml
   ```

3. **Validate a config:**
   ```bash
   petrosmith validate examples/configs/basic_example.yaml
   ```

## Available Examples

### 1. Basic Example (`basic_example.yaml`)
Simple geostatistical analysis with minimal configuration.

**Features:**
- Basic kriging interpolation
- Automatic variogram fitting
- Simple cross-validation
- Minimalist visualizations

**Usage:**
```bash
petrosmith run --config examples/configs/basic_example.yaml
```

### 2. Advanced Example (`advanced_example.yaml`)
Comprehensive analysis with all preprocessing options.

**Features:**
- Outlier detection and removal (IQR method)
- Log transformation
- Multiple variogram models
- Extensive cross-validation
- Publication-quality plots

**Usage:**
```bash
petrosmith run --config examples/configs/advanced_example.yaml
```

### 3. Drilling Example (`drilling_example.yaml`)
Drilling pressure and mud weight analysis.

**Features:**
- Spatial analysis of formation pressures
- Mud weight calculations
- Wellbore stability analysis
- UTM coordinate system support

**Usage:**
```bash
petrosmith run --config examples/configs/drilling_example.yaml
```

### 4. Reservoir Example (`reservoir_example.yaml`)
Reservoir property distribution analysis.

**Features:**
- Porosity mapping
- Anisotropic variogram
- Volumetric calculations
- Net-to-gross ratio analysis

**Usage:**
```bash
petrosmith run --config examples/configs/reservoir_example.yaml
```

## Sample Data Files

The examples use synthetic data for demonstration:

- `sample_data.csv` - Generic spatial data (50 points)
- `drilling_data.csv` - Formation pressure data with UTM coordinates (30 points)
- `reservoir_data.csv` - Porosity and saturation data (30 points)

## Configuration Structure

All config files follow this basic structure:

```yaml
project:           # Project metadata
  name: "..."
  output_dir: "..."

data:              # Input data
  input_file: "..."
  x_column: "..."
  y_column: "..."
  z_column: "..."

preprocessing:     # Data preprocessing
  remove_outliers: true/false
  transform: log/boxcox/null

variogram:         # Variogram modeling
  n_lags: 15
  models: [...]
  auto_fit: true

kriging:           # Interpolation
  method: ordinary/simple/universal
  neighborhood: {...}
  grid: {...}

validation:        # Quality control
  cross_validation: true
  n_folds: 5

visualization:     # Plots
  style: minimalist/publication
  plots: [...]

output:            # Results
  formats: [csv, png, ...]
```

## CLI Commands

### Run Analysis
```bash
petrosmith run --config my_config.yaml
```

### Dry Run (Validate Only)
```bash
petrosmith run --config my_config.yaml --dry-run
```

### Batch Processing
```bash
petrosmith batch config1.yaml config2.yaml config3.yaml
```

### Parameter Sweep
```bash
petrosmith sweep --config base.yaml \
  --param "kriging.neighborhood.max_neighbors=10,25,50" \
  --param "variogram.n_lags=10,15,20"
```

### Get Information
```bash
petrosmith info --config my_config.yaml
```

### List Templates
```bash
petrosmith templates
```

## Customization Guide

### Modify Grid Resolution
```yaml
kriging:
  grid:
    resolution: 1.0  # Smaller = finer grid (slower)
```

### Change Kriging Method
```yaml
kriging:
  method: "ordinary"  # Options: ordinary, simple, universal
```

### Adjust Outlier Detection
```yaml
preprocessing:
  remove_outliers: true
  outlier_method: "iqr"        # Options: iqr, zscore, modified_zscore
  outlier_threshold: 3.0       # Higher = less aggressive
```

### Select Visualization Style
```yaml
visualization:
  style: "minimalist"          # Options: minimalist, default, publication, presentation
  dpi: 300                     # Higher = better quality (larger files)
```

## Output Structure

After running an analysis, you'll find:

```
results/
├── config_used.yaml          # Copy of configuration
├── data_statistics.json      # Data summary
├── analysis_report.txt       # Text report
├── predictions.csv           # Interpolated values
├── variance.csv              # Prediction variance
└── plots/
    ├── variogram.png
    ├── kriging_map.png
    ├── variance_map.png
    └── cross_validation.png
```

## Tips and Best Practices

1. **Start with basic template:** Use `petrosmith init` to create a template
2. **Validate first:** Always run `petrosmith validate` before full analysis
3. **Use dry-run:** Test configuration with `--dry-run` flag
4. **Version control configs:** Keep config files in git for reproducibility
5. **Document changes:** Use project description field to track modifications
6. **Adjust grid resolution:** Balance between accuracy and computation time
7. **Check outliers:** Review outlier removal results before final analysis
8. **Use appropriate CRS:** Specify coordinate reference system for geographic data

## Troubleshooting

### "Config file not found"
- Check file path is correct
- Use absolute path if relative path doesn't work

### "Input file not found"
- Verify `data.input_file` path in config
- Paths are relative to where you run the command

### "Validation failed"
- Run `petrosmith validate config.yaml` to see detailed errors
- Check that all required fields are present
- Ensure grid bounds (x_min < x_max, y_min < y_max)

### Slow execution
- Reduce grid resolution
- Decrease number of cross-validation folds
- Reduce max_neighbors parameter

## Support

For more information:
- Run `petrosmith --help`
- Run `petrosmith <command> --help` for command-specific help
- Check main documentation: `README.md`
- View API docs: `docs/build/html/index.html`
