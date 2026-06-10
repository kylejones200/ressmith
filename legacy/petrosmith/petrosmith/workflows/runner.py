"""
Config-driven workflow runner
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

from petrosmith.config.parser import load_config
from petrosmith.config.schemas import AnalysisConfig
from petrosmith.workflows.pipeline import GeostatsPipeline

logger = logging.getLogger(__name__)


class ConfigRunner:
    """
    Main runner for config-driven workflows
    
    Loads configuration and executes appropriate pipeline.
    """
    
    def __init__(self, config_path: str, overrides: Optional[Dict[str, Any]] = None):
        """
        Initialize runner
        
        Args:
            config_path: Path to configuration file
            overrides: Optional dictionary of config overrides
        """
        self.config_path = Path(config_path)
        self.config = load_config(config_path)
        
        # Apply overrides if provided
        if overrides:
            from petrosmith.config.parser import merge_configs
            self.config = merge_configs(self.config, overrides)
        
        # Select appropriate pipeline
        self.pipeline = self._create_pipeline()
    
    def _create_pipeline(self) -> GeostatsPipeline:
        """
        Create appropriate pipeline based on config
        
        Returns:
            Pipeline instance
        """
        # For now, we only have GeostatsPipeline
        # In future, could branch based on analysis type
        return GeostatsPipeline(self.config)
    
    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Execute complete workflow
        
        Args:
            verbose: Print detailed progress
            
        Returns:
            Dictionary of results
        """
        start_time = time.time()
        
        try:
            # Execute pipeline
            results = self.pipeline.execute()
            
            # Add execution metadata
            results['execution'] = {
                'config_file': str(self.config_path),
                'duration_seconds': time.time() - start_time,
                'status': 'success'
            }
            
            if verbose:
                logger.info(f"Execution time: {results['execution']['duration_seconds']:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during execution: {e}")
            raise
    
    def dry_run(self) -> None:
        """
        Validate configuration without running analysis
        """
        logger.info("="*60)
        logger.info(f"  DRY RUN: {self.config.project.name}")
        logger.info("="*60)
        
        logger.info("Configuration is valid")
        logger.info(f"Project: {self.config.project.name}")
        logger.info(f"Output: {self.config.project.output_dir}")
        logger.info(f"Data: {self.config.data.input_file}")
        logger.info(f"Method: {self.config.kriging.method} kriging")
        
        if self.config.preprocessing.remove_outliers:
            logger.info(f"Preprocessing: Remove outliers ({self.config.preprocessing.outlier_method})")
        if self.config.preprocessing.transform:
            logger.info(f"Transform: {self.config.preprocessing.transform}")
        
        logger.info(f"Validation: {'Yes' if self.config.validation.cross_validation else 'No'}")
        logger.info(f"Visualizations: {', '.join(self.config.visualization.plots)}")
        
        logger.info("="*60)


class BatchRunner:
    """
    Run multiple analyses from multiple config files
    """
    
    def __init__(self, config_paths: list[str]):
        """
        Initialize batch runner
        
        Args:
            config_paths: List of config file paths
        """
        self.config_paths = [Path(p) for p in config_paths]
        self.results = {}
    
    def run(self, continue_on_error: bool = True) -> Dict[str, Any]:
        """
        Run all analyses
        
        Args:
            continue_on_error: Continue to next config if one fails
            
        Returns:
            Dictionary mapping config paths to results
        """
        logger.info("="*60)
        logger.info(f"  BATCH ANALYSIS: {len(self.config_paths)} configurations")
        logger.info("="*60)
        
        for i, config_path in enumerate(self.config_paths, 1):
            logger.info(f"[{i}/{len(self.config_paths)}] Processing: {config_path.name}")
            
            try:
                runner = ConfigRunner(str(config_path))
                result = runner.run(verbose=False)
                self.results[str(config_path)] = {
                    'status': 'success',
                    'result': result
                }
                logger.info(f"Completed: {config_path.name}")
                
            except Exception as e:
                logger.error(f"Failed: {config_path.name}")
                logger.error(f"  Error: {e}")
                
                self.results[str(config_path)] = {
                    'status': 'failed',
                    'error': str(e)
                }
                
                if not continue_on_error:
                    raise
        
        # Summary
        n_success = sum(1 for r in self.results.values() if r['status'] == 'success')
        n_failed = len(self.results) - n_success
        
        logger.info("="*60)
        logger.info(f"  BATCH COMPLETE: {n_success} succeeded, {n_failed} failed")
        logger.info("="*60)
        
        return self.results


class ParameterSweep:
    """
    Run analysis with parameter variations
    """
    
    def __init__(self, base_config_path: str, param_grid: Dict[str, list]):
        """
        Initialize parameter sweep
        
        Args:
            base_config_path: Base configuration file
            param_grid: Dictionary mapping parameter paths to lists of values
                       e.g., {'kriging.neighborhood.max_neighbors': [10, 25, 50]}
        """
        self.base_config_path = Path(base_config_path)
        self.base_config = load_config(base_config_path)
        self.param_grid = param_grid
        self.results = []
    
    def run(self) -> list[Dict[str, Any]]:
        """
        Run parameter sweep
        
        Returns:
            List of results for each parameter combination
        """
        import itertools
        
        # Generate all parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        logger.info("="*60)
        logger.info(f"  PARAMETER SWEEP: {len(combinations)} combinations")
        logger.info("="*60)
        
        for i, values in enumerate(combinations, 1):
            # Create parameter dict
            params = dict(zip(param_names, values))
            
            logger.info(f"[{i}/{len(combinations)}] Testing: {params}")
            
            # Create override dict
            overrides = {}
            for param_path, value in params.items():
                self._set_nested_value(overrides, param_path, value)
            
            # Run analysis
            try:
                runner = ConfigRunner(str(self.base_config_path), overrides=overrides)
                result = runner.run(verbose=False)
                
                self.results.append({
                    'parameters': params,
                    'status': 'success',
                    'result': result
                })
                logger.info("Completed")
                
            except Exception as e:
                logger.error(f"Failed: {e}")
                self.results.append({
                    'parameters': params,
                    'status': 'failed',
                    'error': str(e)
                })
        
        logger.info("="*60)
        logger.info("  SWEEP COMPLETE")
        logger.info("="*60)
        
        return self.results
    
    def _set_nested_value(self, d: dict, path: str, value: Any) -> None:
        """
        Set value in nested dictionary using dot notation
        
        Args:
            d: Dictionary to modify
            path: Dot-separated path (e.g., 'kriging.neighborhood.max_neighbors')
            value: Value to set
        """
        keys = path.split('.')
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value
