"""Utility functions and constants for ResSmith.

This module provides common utilities, error messages, and helper functions
used across the codebase.
"""

from ressmith.utils.errors import (
    ERR_INSUFFICIENT_DATA,
    ERR_MODEL_NOT_FITTED,
    ERR_UNSUPPORTED_FORMAT,
    format_error,
)

__all__ = [
    "ERR_MODEL_NOT_FITTED",
    "ERR_INSUFFICIENT_DATA",
    "ERR_UNSUPPORTED_FORMAT",
    "format_error",
]

