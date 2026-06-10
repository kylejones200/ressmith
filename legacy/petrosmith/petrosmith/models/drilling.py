"""
Drilling domain models - represents drilling equipment and parameters.
"""

from typing import Optional, List
from pydantic import BaseModel, Field, validator


class DrillingParameters(BaseModel):
    """
    Represents drilling operational parameters.
    
    This is a pure data model with no business logic.
    """
    # Depth information
    measured_depth: float = Field(..., gt=0, description="Current measured depth in feet")
    true_vertical_depth: float = Field(..., gt=0, description="Current TVD in feet")
    
    # Drilling parameters
    weight_on_bit: float = Field(..., ge=0, description="Weight on bit in klbs")
    rotary_speed: float = Field(..., ge=0, description="Rotary speed in RPM")
    pump_rate: float = Field(..., ge=0, description="Pump rate in gpm")
    standpipe_pressure: float = Field(..., ge=0, description="Standpipe pressure in psi")
    
    # Performance
    rate_of_penetration: float = Field(..., ge=0, description="ROP in ft/hr")
    torque: Optional[float] = Field(None, ge=0, description="Torque in ft-lbs")
    drag: Optional[float] = Field(None, description="Drag in klbs")
    
    # Hydraulics
    annular_velocity: Optional[float] = Field(None, ge=0, description="Annular velocity in ft/min")
    equivalent_circulating_density: Optional[float] = Field(None, gt=0, description="ECD in ppg")
    
    class Config:
        json_schema_extra = {
            "example": {
                "measured_depth": 8500.0,
                "true_vertical_depth": 8350.0,
                "weight_on_bit": 35.0,
                "rotary_speed": 120.0,
                "pump_rate": 450.0,
                "standpipe_pressure": 2500.0,
                "rate_of_penetration": 45.0
            }
        }


class Casing(BaseModel):
    """
    Represents a casing string.
    
    This is a pure data model with no business logic.
    """
    casing_id: str = Field(..., description="Unique casing identifier")
    casing_type: str = Field(..., description="Type: conductor, surface, intermediate, production")
    
    # Dimensions
    outer_diameter: float = Field(..., gt=0, description="OD in inches")
    inner_diameter: float = Field(..., gt=0, description="ID in inches")
    weight: float = Field(..., gt=0, description="Weight in lb/ft")
    
    # Depth setting
    top_depth: float = Field(0.0, ge=0, description="Top depth in feet")
    bottom_depth: float = Field(..., gt=0, description="Bottom depth in feet (shoe depth)")
    
    # Material
    grade: str = Field(..., description="Steel grade (e.g., J-55, K-55, N-80)")
    connection_type: str = Field(..., description="Connection type")
    
    # Cement
    cement_top: Optional[float] = Field(None, ge=0, description="Top of cement in feet")
    
    @validator("inner_diameter")
    def validate_diameters(cls, v, values):
        if "outer_diameter" in values and v >= values["outer_diameter"]:
            raise ValueError("Inner diameter must be less than outer diameter")
        return v
    
    @validator("bottom_depth")
    def validate_depth(cls, v, values):
        if "top_depth" in values and v <= values["top_depth"]:
            raise ValueError("Bottom depth must be greater than top depth")
        return v
    
    @validator("casing_type")
    def validate_casing_type(cls, v):
        allowed_types = ["conductor", "surface", "intermediate", "production", "liner"]
        if v.lower() not in allowed_types:
            raise ValueError(f"Casing type must be one of {allowed_types}")
        return v.lower()
    
    @property
    def length(self) -> float:
        """Calculate casing length."""
        return self.bottom_depth - self.top_depth
    
    class Config:
        json_schema_extra = {
            "example": {
                "casing_id": "CSG-001",
                "casing_type": "surface",
                "outer_diameter": 13.375,
                "inner_diameter": 12.615,
                "weight": 54.5,
                "top_depth": 0.0,
                "bottom_depth": 3000.0,
                "grade": "K-55",
                "connection_type": "Buttress"
            }
        }


class DrillString(BaseModel):
    """
    Represents a drill string assembly.
    
    This is a pure data model with no business logic.
    """
    # Components
    bit_diameter: float = Field(..., gt=0, description="Bit diameter in inches")
    bit_type: str = Field(..., description="Bit type")
    
    # Drill collar
    drill_collar_od: float = Field(..., gt=0, description="Drill collar OD in inches")
    drill_collar_id: float = Field(..., gt=0, description="Drill collar ID in inches")
    drill_collar_length: float = Field(..., gt=0, description="Total DC length in feet")
    
    # Heavy weight drill pipe
    hwdp_length: Optional[float] = Field(None, ge=0, description="HWDP length in feet")
    
    # Drill pipe
    drill_pipe_od: float = Field(..., gt=0, description="Drill pipe OD in inches")
    drill_pipe_id: float = Field(..., gt=0, description="Drill pipe ID in inches")
    drill_pipe_length: float = Field(..., gt=0, description="Drill pipe length in feet")
    
    @validator("bit_diameter")
    def validate_bit_diameter(cls, v, values):
        """Bit diameter should be reasonable."""
        if v < 3.0 or v > 36.0:
            raise ValueError("Bit diameter must be between 3 and 36 inches")
        return v
    
    @property
    def total_length(self) -> float:
        """Calculate total drill string length."""
        hwdp = self.hwdp_length if self.hwdp_length else 0.0
        return self.drill_collar_length + hwdp + self.drill_pipe_length
    
    class Config:
        json_schema_extra = {
            "example": {
                "bit_diameter": 8.5,
                "bit_type": "PDC",
                "drill_collar_od": 6.5,
                "drill_collar_id": 2.5,
                "drill_collar_length": 600.0,
                "hwdp_length": 300.0,
                "drill_pipe_od": 5.0,
                "drill_pipe_id": 4.276,
                "drill_pipe_length": 9000.0
            }
        }
