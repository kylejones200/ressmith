"""
Command-line interface for PetroSmith config-driven workflows
"""

import click
import sys
import logging
from pathlib import Path
from typing import Optional

from petrosmith import __version__
from petrosmith.config.parser import validate_config_file, load_config
from petrosmith.config.templates import generate_template, list_available_templates
from petrosmith.workflows.runner import ConfigRunner, BatchRunner, ParameterSweep

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)


@click.group()
@click.version_option(version=__version__, prog_name='petrosmith')
def cli():
    """
    PetroSmith - Petroleum Engineering Analysis Framework
    
    Config-driven workflows for geostatistical analysis, drilling,
    and reservoir engineering.
    """
    pass


@cli.command()
@click.option(
    '--config', '-c',
    required=True,
    type=click.Path(exists=True),
    help='Path to configuration file (YAML or JSON)'
)
@click.option(
    '--output', '-o',
    type=click.Path(),
    help='Override output directory'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Validate config without running analysis'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Verbose output'
)
def run(config: str, output: Optional[str], dry_run: bool, verbose: bool):
    """
    Run analysis from configuration file
    
    Example:
        petrosmith run --config analysis.yaml
    """
    try:
        # Create runner
        runner = ConfigRunner(config)
        
        # Override output directory if provided
        if output:
            runner.config.project.output_dir = output
        
        if dry_run:
            runner.dry_run()
        else:
            results = runner.run(verbose=verbose)
            click.echo(f"\nAnalysis complete")
            click.echo(f"Results saved to: {runner.config.project.output_dir}")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option(
    '--output', '-o',
    default='config_template.yaml',
    type=click.Path(),
    help='Output path for template file'
)
@click.option(
    '--type', '-t',
    'template_type',
    type=click.Choice(['basic', 'full', 'drilling', 'reservoir'], case_sensitive=False),
    default='basic',
    help='Template type to generate'
)
def init(output: str, template_type: str):
    """
    Generate configuration template
    
    Examples:
        petrosmith init --output my_config.yaml
        petrosmith init --type drilling --output drilling_config.yaml
    """
    try:
        generate_template(output, template_type)
        click.echo(f"Template created: {output}")
        click.echo(f"\nNext steps:")
        click.echo(f"  1. Edit {output} with your data and parameters")
        click.echo(f"  2. Run: petrosmith run --config {output}")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument(
    'config_file',
    type=click.Path(exists=True)
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Show detailed validation errors'
)
def validate(config_file: str, verbose: bool):
    """
    Validate configuration file
    
    Example:
        petrosmith validate my_config.yaml
    """
    is_valid, message = validate_config_file(config_file)
    
    if is_valid:
        click.echo(message)
        
        # Show config summary
        if verbose:
            try:
                config = load_config(config_file)
                click.echo(f"\nConfiguration Summary:")
                click.echo(f"  Project: {config.project.name}")
                click.echo(f"  Data: {config.data.input_file}")
                click.echo(f"  Method: {config.kriging.method}")
                click.echo(f"  Output: {config.project.output_dir}")
            except Exception as e:
                logging.getLogger(__name__).debug("Could not load config for summary: %s", e)
    else:
        click.echo(message, err=True)
        sys.exit(1)


@cli.command()
@click.argument(
    'config_files',
    nargs=-1,
    type=click.Path(exists=True),
    required=True
)
@click.option(
    '--continue-on-error',
    is_flag=True,
    help='Continue to next config if one fails'
)
def batch(config_files: tuple, continue_on_error: bool):
    """
    Run multiple analyses from multiple config files
    
    Example:
        petrosmith batch config1.yaml config2.yaml config3.yaml
    """
    try:
        runner = BatchRunner(list(config_files))
        results = runner.run(continue_on_error=continue_on_error)
        
        # Summary
        n_success = sum(1 for r in results.values() if r['status'] == 'success')
        n_failed = len(results) - n_success
        
        if n_failed > 0:
            click.echo(f"\nWarning: {n_failed} analysis(es) failed", err=True)
            sys.exit(1)
        else:
            click.echo(f"\nAll {n_success} analyses completed successfully")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    '--config', '-c',
    required=True,
    type=click.Path(exists=True),
    help='Base configuration file'
)
@click.option(
    '--param', '-p',
    'params',
    multiple=True,
    help='Parameter to sweep: path=value1,value2,value3 (e.g., kriging.neighborhood.max_neighbors=10,25,50)'
)
def sweep(config: str, params: tuple):
    """
    Run parameter sweep analysis
    
    Example:
        petrosmith sweep --config base.yaml \\
            --param "kriging.neighborhood.max_neighbors=10,25,50" \\
            --param "variogram.n_lags=10,15,20"
    """
    if not params:
        click.echo("Error: No parameters specified for sweep", err=True)
        click.echo("Use --param option to specify parameters")
        sys.exit(1)
    
    try:
        # Parse parameter grid
        param_grid = {}
        for param in params:
            if '=' not in param:
                raise ValueError(f"Invalid parameter format: {param}")
            
            path, values_str = param.split('=', 1)
            values = [_parse_value(v.strip()) for v in values_str.split(',')]
            param_grid[path.strip()] = values
        
        # Run sweep
        runner = ParameterSweep(config, param_grid)
        results = runner.run()
        
        # Summary
        n_success = sum(1 for r in results if r['status'] == 'success')
        click.echo(f"\nParameter sweep complete: {n_success}/{len(results)} succeeded")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    '--config', '-c',
    type=click.Path(exists=True),
    help='Show info for specific config file'
)
def info(config: Optional[str]):
    """
    Show information about PetroSmith or a config file
    
    Examples:
        petrosmith info
        petrosmith info --config my_config.yaml
    """
    if config:
        # Show config info
        try:
            cfg = load_config(config)
            
            click.echo(f"\n{'='*60}")
            click.echo(f"  Configuration: {Path(config).name}")
            click.echo(f"{'='*60}\n")
            
            click.echo(f"Project:")
            click.echo(f"  Name: {cfg.project.name}")
            if cfg.project.description:
                click.echo(f"  Description: {cfg.project.description}")
            click.echo(f"  Output: {cfg.project.output_dir}")
            
            click.echo(f"\nData:")
            click.echo(f"  Input: {cfg.data.input_file}")
            click.echo(f"  Columns: {cfg.data.x_column}, {cfg.data.y_column}, {cfg.data.z_column}")
            
            click.echo(f"\nAnalysis:")
            click.echo(f"  Method: {cfg.kriging.method} kriging")
            click.echo(f"  Variogram models: {', '.join(cfg.variogram.models)}")
            click.echo(f"  Grid resolution: {cfg.kriging.grid.resolution}")
            
            click.echo(f"\nProcessing:")
            click.echo(f"  Remove outliers: {cfg.preprocessing.remove_outliers}")
            click.echo(f"  Transform: {cfg.preprocessing.transform or 'None'}")
            click.echo(f"  Cross-validation: {cfg.validation.cross_validation}")
            
            click.echo(f"\nOutputs:")
            click.echo(f"  Formats: {', '.join(cfg.output.formats)}")
            click.echo(f"  Plots: {', '.join(cfg.visualization.plots)}")
            
            click.echo(f"\n{'='*60}\n")
        
        except Exception as e:
            click.echo(f"ERROR: Error reading config: {e}", err=True)
            sys.exit(1)
    else:
        # Show general info
        click.echo(f"\n{'='*60}")
        click.echo(f"  PetroSmith v{__version__}")
        click.echo(f"{'='*60}\n")
        click.echo("Petroleum Engineering Analysis Framework")
        click.echo("\nFeatures:")
        click.echo("  - Config-driven workflows")
        click.echo("  - Geostatistical analysis (kriging)")
        click.echo("  - Drilling analysis")
        click.echo("  - Reservoir engineering")
        click.echo("  - Automated validation and visualization")
        
        click.echo("\nAvailable Commands:")
        click.echo("  run       - Run analysis from config")
        click.echo("  init      - Generate config template")
        click.echo("  validate  - Validate config file")
        click.echo("  batch     - Run multiple configs")
        click.echo("  sweep     - Parameter sweep")
        click.echo("  info      - Show information")
        
        click.echo("\nTemplate Types:")
        for template in list_available_templates():
            click.echo(f"  - {template}")
        
        click.echo("\nQuick Start:")
        click.echo("  1. petrosmith init --output my_config.yaml")
        click.echo("  2. Edit my_config.yaml")
        click.echo("  3. petrosmith run --config my_config.yaml")
        
        click.echo(f"\n{'='*60}\n")


@cli.command()
def templates():
    """
    List available configuration templates
    """
    click.echo("\nAvailable Templates:\n")
    
    templates_info = {
        'basic': 'Simple analysis with core features',
        'full': 'Analysis with all available options',
        'drilling': 'Drilling-specific analysis',
        'reservoir': 'Reservoir engineering analysis'
    }
    
    for template, description in templates_info.items():
        click.echo(f"  {template:12s} - {description}")
    
    click.echo("\nGenerate template:")
    click.echo("  petrosmith init --type <template> --output <file>")
    click.echo("\nExample:")
    click.echo("  petrosmith init --type drilling --output drilling.yaml\n")


def _parse_value(value_str: str):
    """Parse string value to appropriate type"""
    # Try to convert to number
    try:
        if '.' in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        # Return as string
        if value_str.lower() == 'true':
            return True
        elif value_str.lower() == 'false':
            return False
        elif value_str.lower() == 'null':
            return None
        return value_str


if __name__ == '__main__':
    cli()
