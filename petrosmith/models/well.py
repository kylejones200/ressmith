"""
Well domain model - represents a petroleum well.
"""

from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, validator


class WellLocation(BaseModel):
    """Geographic location of a well."""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in degrees")
    elevation: float = Field(..., description="Surface elevation in feet")
    

class WellTrajectory(BaseModel):
    """Well trajectory data."""
    measured_depth: float = Field(..., gt=0, description="Measured depth in feet")
    true_vertical_depth: float = Field(..., gt=0, description="True vertical depth in feet")
    inclination: float = Field(..., ge=0, le=90, description="Inclination angle in degrees")
    azimuth: float = Field(..., ge=0, lt=360, description="Azimuth angle in degrees")


class Well(BaseModel):
    """
    Represents a petroleum well with all its properties.
    
    This is a pure data model with no business logic.
    """
    # Identification
    well_id: str = Field(..., description="Unique well identifier")
    well_name: str = Field(..., description="Well name/number")
    operator: str = Field(..., description="Operating company")
    field_name: Optional[str] = Field(None, description="Field name")
    
    # Location
    location: WellLocation
    
    # Well specifications
    total_depth: float = Field(..., gt=0, description="Total measured depth in feet")
    target_formation: str = Field(..., description="Target formation name")
    well_type: str = Field(..., description="Well type: vertical, directional, horizontal")
    
    # Status and dates
    status: str = Field("planned", description="Well status: planned, drilling, completed, producing, abandoned")
    spud_date: Optional[datetime] = Field(None, description="Drilling start date")
    completion_date: Optional[datetime] = Field(None, description="Well completion date")
    
    # Trajectory
    trajectory: List[WellTrajectory] = Field(default_factory=list, description="Well trajectory survey data")
    
    # Additional properties
    water_depth: Optional[float] = Field(None, ge=0, description="Water depth for subsea wells in feet")
    
    @validator("well_type")
    def validate_well_type(cls, v):
        allowed_types = ["vertical", "directional", "horizontal", "multilateral"]
        if v.lower() not in allowed_types:
            raise ValueError(f"Well type must be one of {allowed_types}")
        return v.lower()
    
    @validator("status")
    def validate_status(cls, v):
        allowed_statuses = ["planned", "drilling", "completed", "producing", "suspended", "abandoned"]
        if v.lower() not in allowed_statuses:
            raise ValueError(f"Status must be one of {allowed_statuses}")
        return v.lower()
    
    class Config:
        json_schema_extra = {
            "example": {
                "well_id": "W-001",
                "well_name": "Discovery-1",
                "operator": "Petro Corp",
                "field_name": "East Field",
                "location": {
                    "latitude": 29.5,
                    "longitude": -95.3,
                    "elevation": 150.0
                },
                "total_depth": 10000.0,
                "target_formation": "Cretaceous",
                "well_type": "vertical",
                "status": "drilling"
            }
        }
