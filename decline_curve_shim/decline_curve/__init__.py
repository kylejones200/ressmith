"""
Decline curve analysis - compatibility shim for ressmith.

This package re-exports the dca API from ressmith for backward compatibility.
Use 'ressmith' directly for new development.
"""

from ressmith import dca

__all__ = ["dca"]
__version__ = "0.3.0"
