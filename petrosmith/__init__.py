"""
Petroleum Engineering Library

A 4-layer architecture library for petroleum engineering calculations.
"""

import logging

__version__ = "0.3.0"
__author__ = "Smith Forge"

# Configure library logging - add NullHandler to prevent "No handler" warnings
# Users can enable logging by configuring the 'petrosmith' logger
logging.getLogger(__name__).addHandler(logging.NullHandler())

from petrosmith.models import (
    Well,
    Reservoir,
    FluidProperties,
    DrillingParameters,
    WellCompletion
)

from petrosmith.core import (
    DrillingCalculations,
    ReservoirCalculations,
    WellControlCalculations,
    FluidCalculations
)

from petrosmith.services import (
    WellService,
    DrillingService,
    ProductionService,
    ReservoirService
)

from petrosmith.api import (
    WellAPI,
    DrillingAPI,
    ProductionAPI,
    ReservoirAPI
)

__all__ = [
    # Models (Layer 1)
    "Well",
    "Reservoir",
    "FluidProperties",
    "DrillingParameters",
    "WellCompletion",
    
    # Core (Layer 2)
    "DrillingCalculations",
    "ReservoirCalculations",
    "WellControlCalculations",
    "FluidCalculations",
    
    # Services (Layer 3)
    "WellService",
    "DrillingService",
    "ProductionService",
    "ReservoirService",
    
    # API (Layer 4)
    "WellAPI",
    "DrillingAPI",
    "ProductionAPI",
    "ReservoirAPI",
]
