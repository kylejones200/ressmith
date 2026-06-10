"""
Reservoir domain model - represents reservoir properties.
"""

from typing import Optional
from pydantic import BaseModel, Field, validator


class Reservoir(BaseModel):
    """
    Represents a petroleum reservoir with geological and petrophysical properties.
    
    This is a pure data model with no business logic.
    """
    # Identification
    reservoir_id: str = Field(..., description="Unique reservoir identifier")
    reservoir_name: str = Field(..., description="Reservoir name")
    formation: str = Field(..., description="Geological formation name")
    
    # Depth and thickness
    top_depth: float = Field(..., gt=0, description="Top of reservoir in feet TVD")
    bottom_depth: float = Field(..., gt=0, description="Bottom of reservoir in feet TVD")
    net_pay: float = Field(..., gt=0, description="Net pay thickness in feet")
    
    # Petrophysical properties
    porosity: float = Field(..., ge=0, le=1, description="Porosity (fraction)")
    permeability: float = Field(..., gt=0, description="Permeability in millidarcies")
    water_saturation: float = Field(..., ge=0, le=1, description="Water saturation (fraction)")
    
    # Pressure and temperature
    initial_pressure: float = Field(..., gt=0, description="Initial reservoir pressure in psi")
    current_pressure: Optional[float] = Field(None, gt=0, description="Current reservoir pressure in psi")
    temperature: float = Field(..., gt=0, description="Reservoir temperature in degrees F")
    
    # Fluid properties
    oil_gravity: Optional[float] = Field(None, gt=0, description="Oil API gravity")
    gas_gravity: Optional[float] = Field(None, gt=0, description="Gas specific gravity (air=1)")
    
    # Reservoir type
    fluid_type: str = Field(
        "oil",
        description="Type of hydrocarbon: oil, gas, black_oil, gas_condensate",
    )
    drive_mechanism: str = Field(..., description="Primary drive mechanism")
    
    @validator("fluid_type")
    def validate_fluid_type(cls, v):
        allowed = ["oil", "gas", "black_oil", "gas_condensate"]
        if v.lower() not in allowed:
            raise ValueError(f"fluid_type must be one of {allowed}")
        return v.lower()
    
    @validator("bottom_depth")
    def validate_depth(cls, v, values):
        if "top_depth" in values and v <= values["top_depth"]:
            raise ValueError("Bottom depth must be greater than top depth")
        return v
    
    @validator("net_pay")
    def validate_net_pay(cls, v, values):
        if "top_depth" in values and "bottom_depth" in values:
            gross_thickness = values["bottom_depth"] - values["top_depth"]
            if v > gross_thickness:
                raise ValueError("Net pay cannot exceed gross thickness")
        return v
    
    @validator("drive_mechanism")
    def validate_drive_mechanism(cls, v):
        allowed_types = ["solution_gas", "gas_cap", "water_drive", "gravity_drainage", "combination"]
        if v.lower() not in allowed_types:
            raise ValueError(f"Drive mechanism must be one of {allowed_types}")
        return v.lower()
    
    @property
    def gross_thickness(self) -> float:
        """Calculate gross thickness."""
        return self.bottom_depth - self.top_depth
    
    @property
    def hydrocarbon_saturation(self) -> float:
        """Calculate hydrocarbon saturation."""
        return 1.0 - self.water_saturation
    
    class Config:
        json_schema_extra = {
            "example": {
                "reservoir_id": "RES-001",
                "reservoir_name": "Main Pay Zone",
                "formation": "Cretaceous Sandstone",
                "top_depth": 8500.0,
                "bottom_depth": 8600.0,
                "net_pay": 80.0,
                "porosity": 0.22,
                "permeability": 150.0,
                "water_saturation": 0.25,
                "initial_pressure": 3500.0,
                "temperature": 180.0,
                "oil_gravity": 35.0,
                "fluid_type": "oil",
                "drive_mechanism": "solution_gas"
            }
        }
