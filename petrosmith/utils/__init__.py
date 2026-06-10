"""
Logging configuration for PetroSmith.

Provides centralized logging setup for all modules.
"""

import logging
import sys
from typing import Optional

# Default log format
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration for PetroSmith.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs
        format_string: Optional custom format string
        
    Returns:
        Configured logger instance
        
    Example:
        >>> from petrosmith.utils.logging import setup_logging
        >>> logger = setup_logging(level="DEBUG", log_file="petrosmith.log")
    """
    logger = logging.getLogger("petrosmith")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        format_string or DEFAULT_FORMAT,
        datefmt=DATE_FORMAT
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Name of the logger (typically __name__)
        
    Returns:
        Logger instance
        
    Example:
        >>> from petrosmith.utils.logging import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting calculation")
    """
    return logging.getLogger(f"petrosmith.{name}")

# Create default logger
_default_logger = logging.getLogger("petrosmith")
if not _default_logger.handlers:
    _default_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT, datefmt=DATE_FORMAT))
    _default_logger.addHandler(handler)

__all__ = ["setup_logging", "get_logger"]
