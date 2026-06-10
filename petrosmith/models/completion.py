"""
Well completion domain models - represents completion equipment and design.
"""

from typing import Optional, List
from pydantic import BaseModel, Field, validator


class Perforation(BaseModel):
    """
    Represents a perforation interval.
    
    This is a pure data model with no business logic.
    """
    perforation_id: str = Field(..., description="Unique perforation identifier")
    
    # Interval
    top_depth: float = Field(..., gt=0, description="Top of perforated interval in feet")
    bottom_depth: float = Field(..., gt=0, description="Bottom of perforated interval in feet")
    
    # Perforation properties
    shot_density: float = Field(..., gt=0, description="Shots per foot")
    hole_diameter: float = Field(..., gt=0, description="Perforation hole diameter in inches")
    phasing: int = Field(..., gt=0, le=360, description="Phasing angle in degrees")
    
    # Gun specifications
    gun_type: str = Field(..., description="Perforating gun type")
    charge_type: str = Field(..., description="Charge type")
    
    @validator("bottom_depth")
    def validate_depth(cls, v, values):
        if "top_depth" in values and v <= values["top_depth"]:
            raise ValueError("Bottom depth must be greater than top depth")
        return v
    
    @property
    def interval_length(self) -> float:
        """Calculate perforation interval length."""
        return self.bottom_depth - self.top_depth
    
    @property
    def total_shots(self) -> int:
        """Calculate total number of shots."""
        return int(self.interval_length * self.shot_density)
    
    class Config:
        json_schema_extra = {
            "example": {
                "perforation_id": "PERF-001",
                "top_depth": 8500.0,
                "bottom_depth": 8550.0,
                "shot_density": 4.0,
                "hole_diameter": 0.43,
                "phasing": 60,
                "gun_type": "Through-tubing",
                "charge_type": "Deep penetrating"
            }
        }


class Tubing(BaseModel):
    """
    Represents production tubing.
    
    This is a pure data model with no business logic.
    """
    tubing_id: str = Field(..., description="Unique tubing identifier")
    
    # Dimensions
    outer_diameter: float = Field(..., gt=0, description="OD in inches")
    inner_diameter: float = Field(..., gt=0, description="ID in inches")
    weight: float = Field(..., gt=0, description="Weight in lb/ft")
    
    # Depth setting
    top_depth: float = Field(0.0, ge=0, description="Top depth in feet")
    bottom_depth: float = Field(..., gt=0, description="Bottom depth in feet")
    
    # Material
    grade: str = Field(..., description="Steel grade")
    connection_type: str = Field(..., description="Connection type")
    
    # Accessories
    has_packer: bool = Field(False, description="Has packer installed")
    packer_depth: Optional[float] = Field(None, ge=0, description="Packer depth in feet")
    
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
    
    @validator("packer_depth")
    def validate_packer_depth(cls, v, values):
        if v is not None:
            if "has_packer" not in values or not values["has_packer"]:
                raise ValueError("Packer depth specified but has_packer is False")
            if "bottom_depth" in values and v > values["bottom_depth"]:
                raise ValueError("Packer depth cannot exceed tubing bottom depth")
        return v
    
    @property
    def length(self) -> float:
        """Calculate tubing length."""
        return self.bottom_depth - self.top_depth
    
    class Config:
        json_schema_extra = {
            "example": {
                "tubing_id": "TBG-001",
                "outer_diameter": 2.875,
                "inner_diameter": 2.441,
                "weight": 6.5,
                "top_depth": 0.0,
                "bottom_depth": 8000.0,
                "grade": "J-55",
                "connection_type": "EUE",
                "has_packer": True,
                "packer_depth": 7900.0
            }
        }


class WellCompletion(BaseModel):
    """
    Represents a complete well completion design.
    
    This is a pure data model with no business logic.
    """
    completion_id: str = Field(..., description="Unique completion identifier")
    well_id: str = Field(..., description="Associated well ID")
    completion_type: str = Field(..., description="Completion type")
    
    # Components
    tubing: Tubing
    perforations: List[Perforation] = Field(default_factory=list, description="Perforation intervals")
    
    # Production specifications
    expected_rate_oil: Optional[float] = Field(None, ge=0, description="Expected oil rate in bbl/day")
    expected_rate_gas: Optional[float] = Field(None, ge=0, description="Expected gas rate in Mscf/day")
    expected_rate_water: Optional[float] = Field(None, ge=0, description="Expected water rate in bbl/day")
    
    # Artificial lift
    artificial_lift_type: Optional[str] = Field(None, description="Type of artificial lift")
    
    @validator("completion_type")
    def validate_completion_type(cls, v):
        allowed_types = ["openhole", "cased_hole", "slotted_liner", "gravel_pack", "frac_pack"]
        if v.lower() not in allowed_types:
            raise ValueError(f"Completion type must be one of {allowed_types}")
        return v.lower()
    
    @validator("artificial_lift_type")
    def validate_artificial_lift(cls, v):
        if v is not None:
            allowed_types = ["none", "esp", "gas_lift", "rod_pump", "pcp", "jet_pump"]
            if v.lower() not in allowed_types:
                raise ValueError(f"Artificial lift type must be one of {allowed_types}")
            return v.lower()
        return v
    
    @property
    def total_perforated_interval(self) -> float:
        """Calculate total perforated interval length."""
        return sum(perf.interval_length for perf in self.perforations)
    
    class Config:
        json_schema_extra = {
            "example": {
                "completion_id": "COMP-001",
                "well_id": "W-001",
                "completion_type": "cased_hole",
                "tubing": {
                    "tubing_id": "TBG-001",
                    "outer_diameter": 2.875,
                    "inner_diameter": 2.441,
                    "weight": 6.5,
                    "top_depth": 0.0,
                    "bottom_depth": 8000.0,
                    "grade": "J-55",
                    "connection_type": "EUE"
                },
                "expected_rate_oil": 500.0,
                "expected_rate_gas": 250.0,
                "artificial_lift_type": "esp"
            }
        }
