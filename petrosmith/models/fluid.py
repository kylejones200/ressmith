"""
Fluid properties domain models - represents fluid characteristics.
"""

from typing import Optional
from pydantic import BaseModel, Field, validator


class FluidProperties(BaseModel):
    """
    Represents petroleum fluid properties.
    
    This is a pure data model with no business logic.
    """
    # Identification
    fluid_id: str = Field(..., description="Unique fluid identifier")
    fluid_type: str = Field(..., description="Type of fluid: oil, gas, water, condensate")
    
    # Basic properties
    density: float = Field(..., gt=0, description="Fluid density in lb/gal or ppg")
    viscosity: float = Field(..., gt=0, description="Fluid viscosity in cp")
    temperature: float = Field(..., description="Temperature in degrees F")
    
    # Oil properties
    api_gravity: Optional[float] = Field(None, gt=0, description="API gravity for oil")
    bubble_point_pressure: Optional[float] = Field(None, gt=0, description="Bubble point pressure in psi")
    gas_oil_ratio: Optional[float] = Field(None, ge=0, description="Gas-oil ratio in scf/bbl")
    
    # Gas properties
    gas_gravity: Optional[float] = Field(None, gt=0, description="Gas specific gravity (air=1)")
    z_factor: Optional[float] = Field(None, gt=0, description="Gas compressibility factor")
    
    # Formation volume factors
    oil_fvf: Optional[float] = Field(None, gt=0, description="Oil formation volume factor")
    gas_fvf: Optional[float] = Field(None, gt=0, description="Gas formation volume factor")
    
    @validator("fluid_type")
    def validate_fluid_type(cls, v):
        allowed_types = ["oil", "gas", "water", "condensate", "brine"]
        if v.lower() not in allowed_types:
            raise ValueError(f"Fluid type must be one of {allowed_types}")
        return v.lower()
    
    class Config:
        json_schema_extra = {
            "example": {
                "fluid_id": "FLD-001",
                "fluid_type": "oil",
                "density": 7.2,
                "viscosity": 2.5,
                "temperature": 150.0,
                "api_gravity": 35.0,
                "bubble_point_pressure": 2000.0,
                "gas_oil_ratio": 500.0
            }
        }


class Mud(BaseModel):
    """
    Represents drilling mud properties.
    
    This is a pure data model with no business logic.
    """
    # Identification
    mud_id: str = Field(..., description="Unique mud identifier")
    mud_type: str = Field(..., description="Type of mud: water-based, oil-based, synthetic")
    
    # Physical properties
    density: float = Field(..., gt=0, le=25, description="Mud weight in ppg (pounds per gallon)")
    viscosity: float = Field(..., gt=0, description="Plastic viscosity in cp")
    yield_point: float = Field(..., ge=0, description="Yield point in lb/100ft²")
    
    # Rheological properties
    gel_strength_10sec: Optional[float] = Field(None, ge=0, description="10-second gel strength")
    gel_strength_10min: Optional[float] = Field(None, ge=0, description="10-minute gel strength")
    
    # Chemical properties
    ph: float = Field(..., ge=0, le=14, description="pH value")
    chlorides: Optional[float] = Field(None, ge=0, description="Chloride content in ppm")
    
    # Filtration properties
    api_filtrate: Optional[float] = Field(None, ge=0, description="API filtrate loss in ml/30min")
    filter_cake_thickness: Optional[float] = Field(None, ge=0, description="Filter cake thickness in 1/32 inch")
    
    @validator("mud_type")
    def validate_mud_type(cls, v):
        allowed_types = ["water-based", "oil-based", "synthetic", "pneumatic"]
        if v.lower() not in allowed_types:
            raise ValueError(f"Mud type must be one of {allowed_types}")
        return v.lower()
    
    class Config:
        json_schema_extra = {
            "example": {
                "mud_id": "MUD-001",
                "mud_type": "water-based",
                "density": 10.5,
                "viscosity": 35.0,
                "yield_point": 15.0,
                "ph": 9.5,
                "api_filtrate": 6.0
            }
        }
