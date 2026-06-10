"""
Configuration parser for YAML/JSON config files
"""

import yaml
import json
from pathlib import Path
from typing import Union
from pydantic import ValidationError

from petrosmith.config.schemas import AnalysisConfig


def load_config(config_path: Union[str, Path]) -> AnalysisConfig:
    """
    Load and validate configuration file
    
    Args:
        config_path: Path to YAML or JSON config file
        
    Returns:
        Validated AnalysisConfig object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config is invalid
        ValueError: If file format is unsupported
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load config based on file extension
    suffix = config_path.suffix.lower()
    
    if suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    elif suffix == '.json':
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    else:
        raise ValueError(
            f"Unsupported config format: {suffix}. "
            "Please use .yaml, .yml, or .json"
        )
    
    # Validate and parse config
    try:
        config = AnalysisConfig(**config_dict)
        return config
    except ValidationError as e:
        raise ValidationError(f"Config validation failed: {e}") from e


def save_config(config: AnalysisConfig, output_path: Union[str, Path], format: str = 'yaml') -> None:
    """
    Save configuration to file
    
    Args:
        config: AnalysisConfig object to save
        output_path: Path for output file
        format: Output format ('yaml' or 'json')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict
    config_dict = config.model_dump(exclude_none=True)
    
    if format == 'yaml':
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, indent=2)
    elif format == 'json':
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")


def validate_config_file(config_path: Union[str, Path]) -> tuple[bool, str]:
    """
    Validate config file without raising exceptions
    
    Args:
        config_path: Path to config file
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        load_config(config_path)
        return True, "Config file is valid"
    except FileNotFoundError as e:
        return False, f"File not found: {e}"
    except ValidationError as e:
        return False, f"Validation error:\n{e}"
    except Exception as e:
        return False, f"Error: {e}"


def merge_configs(base_config: AnalysisConfig, override_dict: dict) -> AnalysisConfig:
    """
    Merge override values into base config
    
    Args:
        base_config: Base configuration
        override_dict: Dictionary of values to override
        
    Returns:
        New AnalysisConfig with merged values
    """
    config_dict = base_config.model_dump()
    
    # Recursively merge override values
    def merge_dicts(base: dict, override: dict) -> None:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                merge_dicts(base[key], value)
            else:
                base[key] = value
    
    merge_dicts(config_dict, override_dict)
    return AnalysisConfig(**config_dict)
