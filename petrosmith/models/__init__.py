"""
Layer 1: Data/Domain Models

Pure data classes representing petroleum engineering domain entities.
No business logic, only data structures and validation.
"""

from petrosmith.models.well import Well
from petrosmith.models.reservoir import Reservoir
from petrosmith.models.fluid import FluidProperties, Mud
from petrosmith.models.drilling import DrillingParameters, Casing, DrillString
from petrosmith.models.completion import WellCompletion, Perforation, Tubing

__all__ = [
    "Well",
    "Reservoir",
    "FluidProperties",
    "Mud",
    "DrillingParameters",
    "Casing",
    "DrillString",
    "WellCompletion",
    "Perforation",
    "Tubing",
]
