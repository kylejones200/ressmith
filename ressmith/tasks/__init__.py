"""
Layer 3: Tasks

Tasks translate user intent into object creation, validation, and primitive calls.
No matplotlib imports. Optional backends only if isolated and guarded.
"""

from ressmith.tasks.core import (
    BatchTask,
    EconTask,
    FitDeclineTask,
    ForecastTask,
)

__all__ = [
    "FitDeclineTask",
    "ForecastTask",
    "EconTask",
    "BatchTask",
]

