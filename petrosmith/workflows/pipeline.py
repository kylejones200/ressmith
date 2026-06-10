"""
Base pipeline class for config-driven workflows
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from petrosmith.config.schemas import AnalysisConfig
from petrosmith.geostats import (
    compute_empirical_variogram,
    fit_variogram_wls,
    ordinary_kriging,
    variogram_model,
)

logger = logging.getLogger(__name__)


class GeostatsPipeline:
    """
    Geostatistical analysis pipeline
    
    Implements kriging-based spatial interpolation workflow.
    """
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.data: Optional[Dict[str, np.ndarray]] = None
        self.results: Dict[str, Any] = {}
        self.output_dir = Path(config.project.output_dir)
        self.variogram_model = None
        self.kriging_result = None
    
    def execute(self) -> Dict[str, Any]:
        """Execute complete pipeline"""
        logger.info("="*60)
        logger.info(f"  {self.config.project.name}")
        logger.info("="*60)
        
        self.load_data()
        
        if self.config.preprocessing.remove_outliers or self.config.preprocessing.transform:
            self.preprocess()
        
        self.analyze()
        
        if self.config.validation.cross_validation:
            self.validate()
        
        if self.config.visualization.plots:
            self.visualize()
        
        self.save_results()
        
        logger.info("="*60)
        logger.info("  Analysis Complete")
        logger.info("="*60)
        
        return self.results
        
    def load_data(self) -> None:
        """Load data from CSV file"""
        logger.info("Step: Loading data")
        
        input_path = Path(self.config.data.input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Load CSV
        df = pd.read_csv(
            input_path,
            delimiter=self.config.data.delimiter,
            skiprows=self.config.data.skip_rows
        )
        
        # Extract columns
        self.data = {
            'x': df[self.config.data.x_column].values,
            'y': df[self.config.data.y_column].values,
            'z': df[self.config.data.z_column].values
        }
        
        # Store original data
        self.data['x_original'] = self.data['x'].copy()
        self.data['y_original'] = self.data['y'].copy()
        self.data['z_original'] = self.data['z'].copy()
        
        n_points = len(self.data['x'])
        logger.info(f"  Loaded {n_points:,} data points")
        
        # Store data statistics
        self.results['data_stats'] = {
            'n_points': n_points,
            'x_range': (float(self.data['x'].min()), float(self.data['x'].max())),
            'y_range': (float(self.data['y'].min()), float(self.data['y'].max())),
            'z_range': (float(self.data['z'].min()), float(self.data['z'].max())),
            'z_mean': float(self.data['z'].mean()),
            'z_std': float(self.data['z'].std())
        }
    
    def preprocess(self) -> None:
        """Preprocess data"""
        logger.info("Step: Preprocessing data")
        
        n_original = len(self.data['z'])
        
        if self.config.preprocessing.remove_outliers:
            self._remove_outliers()
        
        if self.config.preprocessing.transform:
            self._apply_transformation()
        
        n_final = len(self.data['z'])
        if n_final < n_original:
            logger.info(f"  Removed {n_original - n_final} outliers ({100*(1-n_final/n_original):.1f}%)")
    
    def _remove_outliers(self) -> None:
        """Remove outliers from data"""
        method = self.config.preprocessing.outlier_method
        threshold = self.config.preprocessing.outlier_threshold
        z = self.data['z']
        
        if method == "iqr":
            q1, q3 = np.percentile(z, [25, 75])
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            mask = (z >= lower) & (z <= upper)
        elif method == "zscore":
            z_scores = np.abs((z - z.mean()) / z.std())
            mask = z_scores < threshold
        elif method == "modified_zscore":
            median = np.median(z)
            mad = np.median(np.abs(z - median))
            modified_z_scores = 0.6745 * (z - median) / mad
            mask = np.abs(modified_z_scores) < threshold
        else:
            raise ValueError(f"Unknown outlier method: {method}")
        
        # Apply mask
        self.data = {k: v[mask] for k, v in self.data.items()}
    
    def _apply_transformation(self) -> None:
        """Apply data transformation"""
        transform = self.config.preprocessing.transform
        z = self.data['z']
        
        if transform == "log":
            # Handle negative/zero values
            if np.any(z <= 0):
                if self.config.preprocessing.handle_negatives == "shift":
                    shift = abs(z.min()) + 1e-6
                    z = z + shift
                    logger.info(f"  Shifted data by {shift:.6f} for log transform")
                elif self.config.preprocessing.handle_negatives == "clip":
                    z = np.clip(z, 1e-6, None)
                elif self.config.preprocessing.handle_negatives == "remove":
                    mask = z > 0
                    self.data = {k: v[mask] for k, v in self.data.items()}
                    z = self.data['z']
            
            self.data['z_transformed'] = np.log(z)
            logger.info("  Applied log transform")
            
        elif transform == "sqrt":
            if np.any(z < 0):
                if self.config.preprocessing.handle_negatives == "shift":
                    shift = abs(z.min())
                    z = z + shift
                    logger.info(f"  Shifted data by {shift:.6f} for sqrt transform")
            self.data['z_transformed'] = np.sqrt(z)
            logger.info("  Applied sqrt transform")
            
        elif transform == "boxcox":
            from scipy.stats import boxcox
            if np.any(z <= 0):
                shift = abs(z.min()) + 1e-6
                z = z + shift
            self.data['z_transformed'], self.lambda_param = boxcox(z)
            logger.info(f"  Applied Box-Cox transform (lambda={self.lambda_param:.3f})")
            
        elif transform == "normal_score":
            from scipy.stats import norm
            ranks = z.argsort().argsort()
            quantiles = (ranks + 1) / (len(z) + 1)
            self.data['z_transformed'] = norm.ppf(quantiles)
            logger.info("  Applied normal score transform")
    
    def analyze(self) -> None:
        """Perform geostatistical analysis: variogram + ordinary kriging."""
        logger.info("Step: Performing geostatistical analysis")
        x = self.data["x"]
        y = self.data["y"]
        z = self.data.get("z_transformed", self.data["z"])

        # Empirical variogram
        lags, gamma_emp, counts = compute_empirical_variogram(
            x, y, z,
            n_lags=self.config.variogram.n_lags,
            max_lag=self.config.variogram.max_lag,
            lag_tolerance=self.config.variogram.lag_tolerance,
        )
        self.results["variogram"] = {
            "lags": lags.tolist(),
            "gamma": gamma_emp.tolist(),
            "counts": counts.tolist(),
        }

        # Fit variogram model
        models_to_try = [m for m in self.config.variogram.models if m in ("spherical", "exponential", "gaussian", "linear", "power")]
        if not models_to_try:
            models_to_try = ["spherical"]
        best_wss = np.inf
        best_params = None
        best_model = models_to_try[0]
        for model in models_to_try:
            try:
                r, s, n = fit_variogram_wls(lags, gamma_emp, counts, model)
                pred = variogram_model(lags, model, r, s, n)
                wss = np.sum(np.sqrt(counts) * (gamma_emp - pred) ** 2)
                if wss < best_wss:
                    best_wss = wss
                    best_params = (r, s, n)
                    best_model = model
            except Exception as e:
                logger.debug("Variogram fit failed for %s: %s", model, e)
        if best_params is None:
            best_params = (np.max(lags) * 0.5, np.nanmax(gamma_emp), 0.0)
        range_, sill, nugget = best_params
        self.variogram_model = {"model": best_model, "range": range_, "sill": sill, "nugget": nugget}
        self.results["variogram_fit"] = self.variogram_model
        logger.info("  Fitted variogram: %s (range=%.2f, sill=%.2f, nugget=%.2f)", best_model, range_, sill, nugget)

        # Build grid
        g = self.config.kriging.grid
        xx = np.linspace(g.x_min, g.x_max, int((g.x_max - g.x_min) / g.resolution) + 1)
        yy = np.linspace(g.y_min, g.y_max, int((g.y_max - g.y_min) / g.resolution) + 1)
        xg, yg = np.meshgrid(xx, yy)
        x_grid = xg.flatten()
        y_grid = yg.flatten()

        # Ordinary kriging
        z_pred, z_var = ordinary_kriging(
            x, y, z,
            x_grid, y_grid,
            model=best_model,
            range_=range_,
            sill=sill,
            nugget=nugget,
            max_neighbors=self.config.kriging.neighborhood.max_neighbors,
            min_neighbors=self.config.kriging.neighborhood.min_neighbors,
        )
        self.kriging_result = {
            "x_grid": x_grid,
            "y_grid": y_grid,
            "z_pred": z_pred,
            "z_var": z_var,
            "shape": (len(yy), len(xx)),
        }
        self.results["kriging"] = {
            "x_min": float(g.x_min),
            "x_max": float(g.x_max),
            "y_min": float(g.y_min),
            "y_max": float(g.y_max),
            "resolution": float(g.resolution),
            "predictions": z_pred.tolist(),
            "variance": z_var.tolist(),
        }
        logger.info("  Kriging complete: %d grid points", len(x_grid))
    
    def validate(self) -> None:
        """Perform leave-one-out cross-validation with RMSE, MAE, R2."""
        logger.info("Step: Validating results")
        x = self.data["x"]
        y = self.data["y"]
        z = self.data.get("z_transformed", self.data["z"])
        vm = self.variogram_model or self.results.get("variogram_fit")
        if not vm:
            logger.warning("  No variogram model; skipping cross-validation")
            self.results["validation"] = {"status": "skipped", "reason": "no variogram"}
            return
        model = vm["model"]
        range_, sill, nugget = vm["range"], vm["sill"], vm["nugget"]
        n = len(z)
        pred_loocv = np.full(n, np.nan)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            x_leave = x[mask]
            y_leave = y[mask]
            z_leave = z[mask]
            pt_x, pt_y = np.array([x[i]]), np.array([y[i]])
            try:
                pred_loocv[i], _ = ordinary_kriging(
                    x_leave, y_leave, z_leave,
                    pt_x, pt_y,
                    model=model, range_=range_, sill=sill, nugget=nugget,
                    max_neighbors=self.config.kriging.neighborhood.max_neighbors,
                    min_neighbors=min(self.config.kriging.neighborhood.min_neighbors, n - 1),
                )
            except Exception:
                pred_loocv[i] = np.mean(z_leave)
        valid = ~np.isnan(pred_loocv)
        if np.sum(valid) == 0:
            self.results["validation"] = {"status": "failed", "reason": "no valid predictions"}
            return
        obs = z[valid]
        pred = pred_loocv[valid]
        rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))
        mae = float(np.mean(np.abs(obs - pred)))
        ss_res = np.sum((obs - pred) ** 2)
        ss_tot = np.sum((obs - np.mean(obs)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        metrics = {"rmse": rmse, "mae": mae, "r2": r2}
        if "mse" in self.config.validation.metrics:
            metrics["mse"] = float(np.mean((obs - pred) ** 2))
        if "mape" in self.config.validation.metrics:
            mape = np.mean(np.abs((obs - pred) / (np.where(obs != 0, obs, 1e-10)))) * 100
            metrics["mape"] = float(mape)
        self.results["validation"] = {
            "method": "leave_one_out",
            "n_folds": n,
            "metrics": metrics,
            "observed": obs.tolist(),
            "predicted": pred.tolist(),
        }
        logger.info("  Cross-validation: RMSE=%.4f, MAE=%.4f, R2=%.4f", rmse, mae, r2)
    
    def visualize(self) -> None:
        """Create visualizations: variogram, kriging map, cross-validation scatter."""
        import matplotlib.pyplot as plt

        from petrosmith.visualization import set_minimalist_style
        from petrosmith.visualization.styles import apply_clean_axes

        set_minimalist_style()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        figsize = self.config.visualization.figure_size
        dpi = self.config.visualization.dpi
        colormap = self.config.visualization.colormap

        for plot_type in self.config.visualization.plots:
            try:
                if plot_type == "variogram" and "variogram" in self.results:
                    fig, ax = plt.subplots(figsize=figsize)
                    lags = np.array(self.results["variogram"]["lags"])
                    gamma = np.array(self.results["variogram"]["gamma"])
                    ax.scatter(lags, gamma, s=30, c="black", label="Empirical")
                    vm = self.results.get("variogram_fit", {})
                    if vm:
                        h_line = np.linspace(0, np.max(lags) * 1.2, 100)
                        g_line = variogram_model(
                            h_line, vm["model"], vm["range"], vm["sill"], vm.get("nugget", 0)
                        )
                        ax.plot(h_line, g_line, "b-", lw=2, label=f"Fitted ({vm['model']})")
                    apply_clean_axes(ax, "Semivariogram")
                    ax.set_xlabel("Lag distance")
                    ax.set_ylabel("Semivariance")
                    ax.legend()
                    fig.tight_layout()
                    path = self.output_dir / "variogram.png"
                    fig.savefig(path, dpi=dpi, bbox_inches="tight")
                    plt.close(fig)
                    logger.info("  Generated variogram plot: %s", path.name)

                elif plot_type == "kriging_map" and self.kriging_result:
                    kr = self.kriging_result
                    shape = kr["shape"]
                    z_pred = kr["z_pred"].reshape(shape)
                    fig, ax = plt.subplots(figsize=figsize)
                    im = ax.imshow(
                        z_pred,
                        extent=[
                            self.config.kriging.grid.x_min,
                            self.config.kriging.grid.x_max,
                            self.config.kriging.grid.y_max,
                            self.config.kriging.grid.y_min,
                        ],
                        aspect="auto",
                        cmap=colormap,
                        origin="upper",
                    )
                    if self.config.visualization.show_data_points:
                        ax.scatter(
                            self.data["x"],
                            self.data["y"],
                            c="black",
                            s=20,
                            marker="o",
                            alpha=0.8,
                        )
                    plt.colorbar(im, ax=ax, label="Predicted value")
                    apply_clean_axes(ax, "Kriging prediction map")
                    ax.set_xlabel("X")
                    ax.set_ylabel("Y")
                    fig.tight_layout()
                    path = self.output_dir / "kriging_map.png"
                    fig.savefig(path, dpi=dpi, bbox_inches="tight")
                    plt.close(fig)
                    logger.info("  Generated kriging map: %s", path.name)

                elif plot_type == "variance_map" and self.kriging_result:
                    kr = self.kriging_result
                    z_var = kr["z_var"].reshape(kr["shape"])
                    fig, ax = plt.subplots(figsize=figsize)
                    im = ax.imshow(
                        z_var,
                        extent=[
                            self.config.kriging.grid.x_min,
                            self.config.kriging.grid.x_max,
                            self.config.kriging.grid.y_max,
                            self.config.kriging.grid.y_min,
                        ],
                        aspect="auto",
                        cmap=colormap,
                        origin="upper",
                    )
                    plt.colorbar(im, ax=ax, label="Kriging variance")
                    apply_clean_axes(ax, "Kriging variance map")
                    ax.set_xlabel("X")
                    ax.set_ylabel("Y")
                    fig.tight_layout()
                    path = self.output_dir / "variance_map.png"
                    fig.savefig(path, dpi=dpi, bbox_inches="tight")
                    plt.close(fig)
                    logger.info("  Generated variance map: %s", path.name)

                elif plot_type == "cross_validation" and "validation" in self.results:
                    val = self.results["validation"]
                    if "observed" in val and "predicted" in val:
                        obs = np.array(val["observed"])
                        pred = np.array(val["predicted"])
                        fig, ax = plt.subplots(figsize=figsize)
                        ax.scatter(obs, pred, s=25, c="black", alpha=0.7)
                        lims = [min(obs.min(), pred.min()), max(obs.max(), pred.max())]
                        ax.plot(lims, lims, "r--", lw=2, label="1:1")
                        apply_clean_axes(ax, "Cross-validation: Observed vs Predicted")
                        ax.set_xlabel("Observed")
                        ax.set_ylabel("Predicted")
                        ax.legend()
                        fig.tight_layout()
                        path = self.output_dir / "cross_validation.png"
                        fig.savefig(path, dpi=dpi, bbox_inches="tight")
                        plt.close(fig)
                        logger.info("  Generated cross-validation plot: %s", path.name)
            except Exception as e:
                logger.warning("  Failed to generate %s plot: %s", plot_type, e)
    
    def save_results(self) -> None:
        """Save results to disk"""
        logger.info("Step: Saving results")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration copy
        if self.config.output.save_config:
            from petrosmith.config.parser import save_config
            config_path = self.output_dir / "config_used.yaml"
            save_config(self.config, config_path)
            logger.info(f"  Saved config: {config_path.name}")
        
        # Save data statistics
        stats_path = self.output_dir / "data_statistics.json"
        with open(stats_path, "w") as f:
            json.dump(self.results.get("data_stats", {}), f, indent=2)
        logger.info("  Saved statistics: %s", stats_path.name)

        # Save predictions and variance if configured
        if self.config.output.save_predictions and self.kriging_result:
            kr = self.kriging_result
            pred_path = self.output_dir / "predictions.csv"
            pd.DataFrame({
                "x": kr["x_grid"],
                "y": kr["y_grid"],
                "predicted": kr["z_pred"],
                "variance": kr["z_var"],
            }).to_csv(pred_path, index=False)
            logger.info("  Saved predictions: %s", pred_path.name)
        if self.config.output.save_variance and self.kriging_result and "csv" in self.config.output.formats:
            # Already in predictions.csv; could save separate variance grid
            pass

        # Save summary report
        self._save_report()
        
        logger.info(f"  Results saved to: {self.output_dir}")
    
    def _save_report(self) -> None:
        """Generate and save text report"""
        report_path = self.output_dir / "analysis_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write(f"  {self.config.project.name}\n")
            f.write("="*60 + "\n\n")
            
            # Project info
            f.write("PROJECT INFORMATION\n")
            f.write("-"*60 + "\n")
            if self.config.project.description:
                f.write(f"Description: {self.config.project.description}\n")
            if self.config.project.author:
                f.write(f"Author: {self.config.project.author}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            
            # Data statistics
            if 'data_stats' in self.results:
                stats = self.results['data_stats']
                f.write("DATA STATISTICS\n")
                f.write("-"*60 + "\n")
                f.write(f"Number of points: {stats['n_points']:,}\n")
                f.write(f"X range: {stats['x_range'][0]:.2f} to {stats['x_range'][1]:.2f}\n")
                f.write(f"Y range: {stats['y_range'][0]:.2f} to {stats['y_range'][1]:.2f}\n")
                f.write(f"Value range: {stats['z_range'][0]:.3f} to {stats['z_range'][1]:.3f}\n")
                f.write(f"Mean value: {stats['z_mean']:.3f}\n")
                f.write(f"Std deviation: {stats['z_std']:.3f}\n\n")
            
            # Analysis summary
            if "variogram_fit" in self.results:
                vf = self.results["variogram_fit"]
                f.write("VARIOGRAM FIT\n")
                f.write("-" * 60 + "\n")
                f.write(f"Model: {vf.get('model', 'N/A')}\n")
                f.write(f"Range: {vf.get('range', 0):.4f}\n")
                f.write(f"Sill: {vf.get('sill', 0):.4f}\n")
                f.write(f"Nugget: {vf.get('nugget', 0):.4f}\n\n")
            if "validation" in self.results and "metrics" in self.results["validation"]:
                m = self.results["validation"]["metrics"]
                f.write("CROSS-VALIDATION METRICS\n")
                f.write("-" * 60 + "\n")
                for k, v in m.items():
                    f.write(f"  {k}: {v:.4f}\n")
                f.write("\n")

            # Configuration summary
            f.write("ANALYSIS CONFIGURATION\n")
            f.write("-" * 60 + "\n")
            f.write(f"Preprocessing: {'Yes' if self.config.preprocessing.remove_outliers or self.config.preprocessing.transform else 'No'}\n")
            f.write(f"Kriging method: {self.config.kriging.method}\n")
            f.write(f"Validation: {'Yes' if self.config.validation.cross_validation else 'No'}\n")
            f.write(f"Visualizations: {len(self.config.visualization.plots)}\n")
