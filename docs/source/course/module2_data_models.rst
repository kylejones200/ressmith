Module 2: Data Models and Validation
====================================

.. meta::
   :description: Learn to build robust, validated data models for petroleum engineering using Pydantic
   :keywords: pydantic, data validation, type safety, domain models, petroleum engineering

**Learning Objectives**

After completing this module, you will be able to:

- Design type-safe data models using Pydantic
- Implement custom validators for engineering constraints
- Build domain models for wells, reservoirs, and production systems
- Serialize/deserialize data for databases and APIs
- Ensure data integrity throughout your applications
- Apply domain-driven design principles to petroleum engineering

**Time Commitment:** 4-6 hours

**Prerequisites:** Module 1 (Fundamentals)

----

Introduction: Why Data Validation Matters
------------------------------------------

The Problem with Unvalidated Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In petroleum engineering, **data quality directly impacts decisions worth millions of dollars**:

❌ **Without Validation:**

.. code-block:: python

   # Dangerous: No validation!
   well_depth = -5000  # Negative depth?
   porosity = 1.5      # >100% porosity?
   pressure = "high"   # String instead of number?
   
   # Calculate reserves
   ooip = 7758 * area * depth * porosity * So / Bo
   # Result: Garbage in, garbage out!

This leads to:

- Invalid calculations
- Runtime errors in production
- Incorrect business decisions
- Failed audits
- Lost time debugging

✅ **With Pydantic Validation:**

.. code-block:: python

   from pydantic import BaseModel, Field, validator
   
   class Well(BaseModel):
       well_id: str
       depth: float = Field(gt=0, description="Well depth in feet")
       porosity: float = Field(gt=0, le=1, description="Porosity fraction")
       pressure: float = Field(gt=0, description="Pressure in psi")
       
       @validator('depth')
       def validate_depth(cls, v):
           if v > 50000:
               raise ValueError('Depth exceeds maximum drillable depth')
           return v

Benefits:

- ✅ Automatic validation at data entry
- ✅ Clear error messages
- ✅ Type safety for IDE support
- ✅ Self-documenting code
- ✅ Serialization built-in
- ✅ Database-ready

----

Part 1: Pydantic Fundamentals
------------------------------

1.1 Basic Models
^^^^^^^^^^^^^^^^^

Let's start with a simple well model:

.. code-block:: python

   from pydantic import BaseModel
   from typing import Optional
   from datetime import datetime
   
   class Well(BaseModel):
       """Basic well data model"""
       
       well_id: str
       well_name: str
       operator: str
       field: str
       latitude: float
       longitude: float
       spud_date: datetime
       status: str  # "drilling", "producing", "shut-in", "abandoned"
       total_depth: Optional[float] = None
       
   # Create a well
   well = Well(
       well_id="API-123456",
       well_name="Smith #1",
       operator="XTO Energy",
       field="Permian Basin",
       latitude=31.8457,
       longitude=-102.3676,
       spud_date="2024-01-15",
       status="producing",
       total_depth=10500
   )
   
   print(well.well_name)  # Smith #1
   print(well.spud_date)  # datetime object

**Key Features:**

- Automatic type conversion (string → datetime)
- Optional fields with defaults
- Immutable by default (use ``model_copy()`` to modify)
- JSON serialization: ``well.model_dump_json()``

1.2 Field Validators
^^^^^^^^^^^^^^^^^^^^

Add constraints to ensure data quality:

.. code-block:: python

   from pydantic import BaseModel, Field
   
   class Reservoir(BaseModel):
       """Reservoir with field-level validation"""
       
       reservoir_id: str = Field(..., min_length=1, max_length=50)
       
       porosity: float = Field(
           ...,
           gt=0,           # Greater than 0
           le=1,           # Less than or equal to 1
           description="Porosity as fraction (0-1)"
       )
       
       permeability: float = Field(
           ...,
           gt=0,
           description="Permeability in millidarcies"
       )
       
       net_pay: float = Field(
           ...,
           gt=0,
           lt=10000,  # Reasonable maximum
           description="Net pay thickness in feet"
       )
       
       initial_pressure: float = Field(
           ...,
           gt=0,
           lt=30000,  # Maximum expected pressure
           description="Initial reservoir pressure in psi"
       )
       
       temperature: float = Field(
           ...,
           gt=32,   # Above freezing
           lt=500,  # Below extreme geothermal
           description="Reservoir temperature in °F"
       )

**Common Field Constraints:**

- ``gt``, ``ge``: Greater than (or equal)
- ``lt``, ``le``: Less than (or equal)
- ``min_length``, ``max_length``: String length
- ``regex``: Pattern matching
- ``description``: Documentation

1.3 Custom Validators
^^^^^^^^^^^^^^^^^^^^^^

Implement complex business logic:

.. code-block:: python

   from pydantic import BaseModel, validator, root_validator
   from typing import Literal
   
   class CompletionDesign(BaseModel):
       """Well completion with custom validation"""
       
       completion_type: Literal["openhole", "cased_perforated", "frac"]
       tubing_size: float  # inches
       packer_depth: float  # feet
       perforation_top: Optional[float] = None  # feet
       perforation_bottom: Optional[float] = None  # feet
       frac_stages: Optional[int] = None
       
       @validator('tubing_size')
       def validate_tubing(cls, v):
           """Ensure tubing size is standard"""
           standard_sizes = [2.375, 2.875, 3.5, 4.5, 5.5, 7.0]
           if v not in standard_sizes:
               raise ValueError(
                   f'Tubing size must be one of {standard_sizes}'
               )
           return v
       
       @validator('perforation_bottom')
       def validate_perf_interval(cls, v, values):
           """Ensure perforations make sense"""
           if v is not None and 'perforation_top' in values:
               if v <= values['perforation_top']:
                   raise ValueError(
                       'perforation_bottom must be greater than perforation_top'
                   )
           return v
       
       @root_validator
       def validate_completion_consistency(cls, values):
           """Check overall completion consistency"""
           comp_type = values.get('completion_type')
           
           # Cased/perforated must have perforation data
           if comp_type == 'cased_perforated':
               if not values.get('perforation_top'):
                   raise ValueError(
                       'Cased/perforated completion requires perforation data'
                   )
           
           # Frac completion must have stage count
           if comp_type == 'frac':
               if not values.get('frac_stages'):
                   raise ValueError(
                       'Frac completion requires frac_stages'
                   )
               if values.get('frac_stages') < 1:
                   raise ValueError(
                       'frac_stages must be at least 1'
                   )
           
           return values

**Validator Types:**

1. **Field validators** (``@validator``): Validate individual fields
2. **Root validators** (``@root_validator``): Validate entire model
3. **Pre-validators**: Run before type conversion
4. **Post-validators**: Run after type conversion (default)

----

Part 2: Domain Models for Petroleum Engineering
------------------------------------------------

2.1 Well Model
^^^^^^^^^^^^^^

Well data model:

.. code-block:: python

   from pydantic import BaseModel, Field, validator
   from typing import Optional, Literal, List
   from datetime import datetime
   from enum import Enum
   
   class WellStatus(str, Enum):
       """Well status enumeration"""
       DRILLING = "drilling"
       COMPLETING = "completing"
       PRODUCING = "producing"
       SHUT_IN = "shut_in"
       ABANDONED = "abandoned"
       PLUGGED = "plugged"
   
   class WellType(str, Enum):
       """Well type enumeration"""
       VERTICAL = "vertical"
       DIRECTIONAL = "directional"
       HORIZONTAL = "horizontal"
   
   class Well(BaseModel):
       """Complete well data model"""
       
       # Identification
       well_id: str = Field(..., description="Unique well identifier (API number)")
       well_name: str = Field(..., description="Well name")
       operator: str = Field(..., description="Operating company")
       field: str = Field(..., description="Field name")
       
       # Location
       latitude: float = Field(..., ge=-90, le=90)
       longitude: float = Field(..., ge=-180, le=180)
       surface_elevation: float = Field(..., description="Surface elevation in feet")
       
       # Well characteristics
       well_type: WellType
       status: WellStatus
       spud_date: datetime
       completion_date: Optional[datetime] = None
       
       # Depths
       measured_depth: float = Field(..., gt=0, description="Measured depth in feet")
       true_vertical_depth: float = Field(..., gt=0, description="TVD in feet")
       
       # Production data
       current_oil_rate: Optional[float] = Field(None, ge=0, description="Oil rate in STB/day")
       current_gas_rate: Optional[float] = Field(None, ge=0, description="Gas rate in Mscf/day")
       current_water_rate: Optional[float] = Field(None, ge=0, description="Water rate in STB/day")
       
       # Completion
       tubing_size: Optional[float] = Field(None, gt=0, description="Tubing OD in inches")
       casing_size: Optional[float] = Field(None, gt=0, description="Production casing in inches")
       
       @validator('true_vertical_depth')
       def tvd_less_than_md(cls, v, values):
           """TVD cannot exceed MD"""
           if 'measured_depth' in values and v > values['measured_depth']:
               raise ValueError('TVD cannot exceed measured depth')
           return v
       
       @validator('completion_date')
       def completion_after_spud(cls, v, values):
           """Completion must be after spud"""
           if v and 'spud_date' in values:
               if v < values['spud_date']:
                   raise ValueError('Completion date must be after spud date')
           return v
       
       def water_cut(self) -> Optional[float]:
           """Calculate water cut percentage"""
           if self.current_oil_rate and self.current_water_rate:
               total_liquid = self.current_oil_rate + self.current_water_rate
               return (self.current_water_rate / total_liquid) * 100
           return None
       
       def gor(self) -> Optional[float]:
           """Calculate producing GOR"""
           if self.current_gas_rate and self.current_oil_rate:
               return (self.current_gas_rate * 1000) / self.current_oil_rate
           return None
       
       class Config:
           """Model configuration"""
           use_enum_values = True  # Store as string values
           validate_assignment = True  # Validate on attribute assignment
           json_schema_extra = {
               "example": {
                   "well_id": "API-12-345-67890",
                   "well_name": "Smith #1H",
                   "operator": "XTO Energy",
                   "field": "Permian Basin",
                   "latitude": 31.8457,
                   "longitude": -102.3676,
                   "surface_elevation": 2850,
                   "well_type": "horizontal",
                   "status": "producing",
                   "spud_date": "2024-01-15",
                   "measured_depth": 15000,
                   "true_vertical_depth": 9000,
                   "current_oil_rate": 500,
                   "current_gas_rate": 750,
                   "current_water_rate": 100
               }
           }

2.2 Reservoir Model
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pydantic import BaseModel, Field, validator
   from typing import Literal, Optional
   
   class Reservoir(BaseModel):
       """Reservoir data model with engineering constraints"""
       
       # Identification
       reservoir_id: str
       reservoir_name: str
       formation: str
       field: str
       
       # Geometry
       area_acres: float = Field(..., gt=0, description="Reservoir area in acres")
       net_pay_ft: float = Field(..., gt=0, lt=1000, description="Net pay in feet")
       gross_pay_ft: Optional[float] = Field(None, gt=0, description="Gross pay in feet")
       
       # Rock properties
       porosity: float = Field(..., gt=0, le=0.5, description="Porosity fraction")
       permeability_md: float = Field(..., gt=0, lt=10000, description="Permeability in md")
       
       # Fluid properties
       oil_saturation: float = Field(..., gt=0, le=1, description="Oil saturation fraction")
       water_saturation: float = Field(..., gt=0, le=1, description="Water saturation fraction")
       gas_saturation: Optional[float] = Field(None, gt=0, le=1)
       
       # Pressure and temperature
       initial_pressure_psi: float = Field(..., gt=0, lt=30000)
       current_pressure_psi: Optional[float] = Field(None, gt=0)
       temperature_f: float = Field(..., gt=32, lt=500)
       
       # Fluid type
       fluid_type: Literal["oil", "gas", "gas_condensate"]
       
       # Drive mechanism
       drive_mechanism: Literal[
           "solution_gas",
           "gas_cap",
           "water_drive",
           "gravity_drainage",
           "combination"
       ]
       
       @validator('water_saturation', 'oil_saturation', 'gas_saturation')
       def saturation_valid(cls, v):
           """Saturations must be between 0 and 1"""
           if v is not None and (v < 0 or v > 1):
               raise ValueError('Saturation must be between 0 and 1')
           return v
       
       @validator('gas_saturation')
       def saturations_sum_to_one(cls, v, values):
           """Check that saturations sum to ~1.0"""
           if v is not None:
               So = values.get('oil_saturation', 0)
               Sw = values.get('water_saturation', 0)
               Sg = v
               total = So + Sw + Sg
               if abs(total - 1.0) > 0.01:  # Allow small tolerance
                   raise ValueError(
                       f'Saturations must sum to 1.0 (got {total:.3f})'
                   )
           return v
       
       @validator('current_pressure_psi')
       def current_less_than_initial(cls, v, values):
           """Current pressure should be less than initial"""
           if v and 'initial_pressure_psi' in values:
               if v > values['initial_pressure_psi']:
                   raise ValueError(
                       'Current pressure cannot exceed initial pressure'
                   )
           return v
       
       @validator('gross_pay_ft')
       def gross_greater_than_net(cls, v, values):
           """Gross pay must be >= net pay"""
           if v and 'net_pay_ft' in values:
               if v < values['net_pay_ft']:
                   raise ValueError(
                       'Gross pay must be greater than or equal to net pay'
                   )
           return v
       
       def net_to_gross_ratio(self) -> Optional[float]:
           """Calculate N/G ratio"""
           if self.gross_pay_ft:
               return self.net_pay_ft / self.gross_pay_ft
           return None
       
       def hydrocarbon_saturation(self) -> float:
           """Calculate total hydrocarbon saturation"""
           return 1.0 - self.water_saturation

2.3 Fluid Properties Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pydantic import BaseModel, Field, validator
   from typing import Optional
   
   class FluidProperties(BaseModel):
       """PVT fluid properties model"""
       
       # Oil properties
       oil_api_gravity: Optional[float] = Field(
           None, gt=5, lt=60, description="API gravity"
       )
       oil_fvf: Optional[float] = Field(
           None, gt=1.0, lt=3.0, description="Oil FVF (rb/STB)"
       )
       oil_viscosity_cp: Optional[float] = Field(
           None, gt=0.1, lt=10000, description="Oil viscosity (cp)"
       )
       
       # Gas properties
       gas_specific_gravity: Optional[float] = Field(
           None, gt=0.5, lt=1.5, description="Gas SG (air=1)"
       )
       gas_fvf: Optional[float] = Field(
           None, gt=0.001, lt=0.1, description="Gas FVF (rcf/scf)"
       )
       gas_viscosity_cp: Optional[float] = Field(
           None, gt=0.01, lt=0.1, description="Gas viscosity (cp)"
       )
       
       # Solution gas
       solution_gor_scf_stb: Optional[float] = Field(
           None, ge=0, lt=10000, description="Solution GOR (scf/STB)"
       )
       
       # Water properties
       water_fvf: Optional[float] = Field(
           None, gt=0.95, lt=1.2, description="Water FVF (rb/STB)"
       )
       water_viscosity_cp: Optional[float] = Field(
           None, gt=0.1, lt=5.0, description="Water viscosity (cp)"
       )
       water_salinity_ppm: Optional[float] = Field(
           None, ge=0, lt=300000, description="Water salinity (ppm)"
       )
       
       # Pressure/Temperature
       bubble_point_psi: Optional[float] = Field(
           None, gt=0, lt=10000, description="Bubble point pressure (psi)"
       )
       reference_pressure_psi: float = Field(
           ..., gt=0, description="Reference pressure for PVT"
       )
       reference_temperature_f: float = Field(
           ..., gt=32, lt=400, description="Reference temperature"
       )
       
       @validator('oil_api_gravity')
       def validate_api_gravity(cls, v):
           """Validate API gravity range"""
           if v:
               if v < 10:
                   print("Warning: Very heavy crude oil")
               elif v > 45:
                   print("Warning: Very light crude oil/condensate")
           return v

----

Part 3: Advanced Validation Patterns
-------------------------------------

3.1 Cross-Field Validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Validate relationships between fields:

.. code-block:: python

   from pydantic import BaseModel, root_validator
   
   class WellTest(BaseModel):
       """Well test data with cross-field validation"""
       
       test_date: datetime
       test_duration_hours: float = Field(gt=0, lt=720)  # Max 30 days
       
       # Rates
       oil_rate_stb_day: float = Field(ge=0)
       gas_rate_mscf_day: float = Field(ge=0)
       water_rate_stb_day: float = Field(ge=0)
       
       # Pressures
       flowing_tubing_pressure_psi: float = Field(gt=0)
       flowing_casing_pressure_psi: float = Field(gt=0)
       static_pressure_psi: Optional[float] = Field(None, gt=0)
       
       # Choke
       choke_size_64ths: float = Field(gt=0, le=128)
       
       @root_validator
       def validate_test_data(cls, values):
           """Validate test data consistency"""
           
           # At least one fluid must be produced
           oil = values.get('oil_rate_stb_day', 0)
           gas = values.get('gas_rate_mscf_day', 0)
           water = values.get('water_rate_stb_day', 0)
           
           if oil + gas + water == 0:
               raise ValueError('At least one production rate must be > 0')
           
           # Static pressure should exceed flowing pressure
           static_p = values.get('static_pressure_psi')
           flowing_p = values.get('flowing_tubing_pressure_psi')
           
           if static_p and flowing_p:
               if static_p < flowing_p:
                   raise ValueError(
                       'Static pressure should exceed flowing pressure'
                   )
           
           # Calculate water cut
           if oil + water > 0:
               water_cut = (water / (oil + water)) * 100
               if water_cut > 98:
                   print(f"Warning: Very high water cut ({water_cut:.1f}%)")
           
           return values

3.2 Conditional Validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Different rules for different scenarios:

.. code-block:: python

   from pydantic import BaseModel, validator
   from typing import Literal, Optional
   
   class ArtificialLiftDesign(BaseModel):
       """Artificial lift with conditional validation"""
       
       lift_type: Literal["ESP", "gas_lift", "rod_pump", "PCP", "none"]
       
       # ESP-specific
       esp_stages: Optional[int] = Field(None, gt=0, lt=1000)
       esp_frequency_hz: Optional[float] = Field(None, gt=0, le=120)
       esp_setting_depth_ft: Optional[float] = Field(None, gt=0)
       
       # Gas lift-specific
       gas_injection_rate_mscf_day: Optional[float] = Field(None, gt=0)
       injection_pressure_psi: Optional[float] = Field(None, gt=0)
       number_of_valves: Optional[int] = Field(None, gt=0, le=10)
       
       # Rod pump-specific
       pump_size_inches: Optional[float] = Field(None, gt=0, le=5)
       stroke_length_inches: Optional[float] = Field(None, gt=0, le=300)
       strokes_per_minute: Optional[float] = Field(None, gt=0, le=30)
       
       @root_validator
       def validate_lift_parameters(cls, values):
           """Ensure required parameters for each lift type"""
           lift_type = values.get('lift_type')
           
           if lift_type == 'ESP':
               if not values.get('esp_stages'):
                   raise ValueError('ESP design requires esp_stages')
               if not values.get('esp_setting_depth_ft'):
                   raise ValueError('ESP design requires esp_setting_depth_ft')
           
           elif lift_type == 'gas_lift':
               if not values.get('gas_injection_rate_mscf_day'):
                   raise ValueError('Gas lift requires gas_injection_rate_mscf_day')
               if not values.get('number_of_valves'):
                   raise ValueError('Gas lift requires number_of_valves')
           
           elif lift_type == 'rod_pump':
               required = ['pump_size_inches', 'stroke_length_inches', 'strokes_per_minute']
               missing = [f for f in required if not values.get(f)]
               if missing:
                   raise ValueError(f'Rod pump missing required fields: {missing}')
           
           return values

----

Part 4: Serialization and Database Integration
-----------------------------------------------

4.1 JSON Serialization
^^^^^^^^^^^^^^^^^^^^^^

Convert models to/from JSON:

.. code-block:: python

   # Model to JSON
   well = Well(
       well_id="API-123",
       well_name="Test #1",
       # ... other fields
   )
   
   # Serialize to JSON string
   json_str = well.model_dump_json(indent=2)
   
   # Serialize to dict
   well_dict = well.model_dump()
   
   # Exclude certain fields
   public_data = well.model_dump(exclude={'current_oil_rate', 'current_gas_rate'})
   
   # Include only certain fields
   summary = well.model_dump(include={'well_id', 'well_name', 'status'})

.. code-block:: python

   # JSON to model
   json_data = '''
   {
       "well_id": "API-456",
       "well_name": "Test #2",
       "operator": "XTO",
       "field": "Permian",
       "latitude": 31.5,
       "longitude": -102.0,
       "surface_elevation": 2800,
       "well_type": "horizontal",
       "status": "producing",
       "spud_date": "2024-01-20",
       "measured_depth": 14000,
       "true_vertical_depth": 8500
   }
   '''
   
   # Parse JSON
   well = Well.model_validate_json(json_data)
   
   # Or from dict
   well_dict = json.loads(json_data)
   well = Well(**well_dict)

4.2 Database Integration
^^^^^^^^^^^^^^^^^^^^^^^^^

Use with SQLAlchemy:

.. code-block:: python

   from sqlalchemy import Column, String, Float, DateTime, Integer
   from sqlalchemy.ext.declarative import declarative_base
   from sqlalchemy.orm import Session
   from pydantic import BaseModel
   
   Base = declarative_base()
   
   # SQLAlchemy ORM model
   class WellDB(Base):
       __tablename__ = 'wells'
       
       well_id = Column(String, primary_key=True)
       well_name = Column(String, nullable=False)
       operator = Column(String)
       field = Column(String)
       latitude = Column(Float)
       longitude = Column(Float)
       status = Column(String)
       measured_depth = Column(Float)
       true_vertical_depth = Column(Float)
       current_oil_rate = Column(Float)
       current_gas_rate = Column(Float)
       spud_date = Column(DateTime)
   
   # Pydantic model for API
   class WellAPI(BaseModel):
       well_id: str
       well_name: str
       operator: str
       field: str
       latitude: float
       longitude: float
       status: str
       measured_depth: float
       true_vertical_depth: float
       current_oil_rate: Optional[float] = None
       current_gas_rate: Optional[float] = None
       
       class Config:
           from_attributes = True  # Enable ORM mode
   
   # Usage
   def get_well(db: Session, well_id: str) -> WellAPI:
       """Get well from database and return as Pydantic model"""
       well_db = db.query(WellDB).filter(WellDB.well_id == well_id).first()
       return WellAPI.model_validate(well_db)
   
   def create_well(db: Session, well: WellAPI) -> WellDB:
       """Create well in database from Pydantic model"""
       well_db = WellDB(**well.model_dump())
       db.add(well_db)
       db.commit()
       return well_db

4.3 API Integration
^^^^^^^^^^^^^^^^^^^

Build type-safe APIs:

.. code-block:: python

   from fastapi import FastAPI, HTTPException
   from pydantic import BaseModel
   from typing import List
   
   app = FastAPI()
   
   # Request model
   class WellCreate(BaseModel):
       well_name: str
       operator: str
       field: str
       latitude: float = Field(ge=-90, le=90)
       longitude: float = Field(ge=-180, le=180)
       # ... other fields
   
   # Response model
   class WellResponse(BaseModel):
       well_id: str
       well_name: str
       status: str
       created_at: datetime
   
   # API endpoint with automatic validation
   @app.post("/wells/", response_model=WellResponse)
   def create_well(well: WellCreate):
       """
       Create a new well.
       
       Pydantic automatically validates the request body!
       """
       # Generate ID
       well_id = f"API-{hash(well.well_name) % 100000:06d}"
       
       # Create well (validation already done by Pydantic)
       # ... save to database ...
       
       return WellResponse(
           well_id=well_id,
           well_name=well.well_name,
           status="drilling",
           created_at=datetime.now()
       )
   
   @app.get("/wells/", response_model=List[WellResponse])
   def list_wells(skip: int = 0, limit: int = 100):
       """List wells with pagination"""
       # ... query database ...
       return wells

----

Part 5: Real-World Application
-------------------------------

5.1 Complete Field Data Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Building a complete field management system:

.. code-block:: python

   from pydantic import BaseModel, Field
   from typing import List, Dict, Optional
   from datetime import datetime
   
   class Field(BaseModel):
       """Complete field data model"""
       
       # Field identification
       field_id: str
       field_name: str
       basin: str
       country: str
       operator: str
       
       # Reservoirs in the field
       reservoirs: List[Reservoir] = []
       
       # Wells in the field
       wells: List[Well] = []
       
       # Production summary
       daily_oil_production_stb: Optional[float] = None
       daily_gas_production_mscf: Optional[float] = None
       daily_water_production_stb: Optional[float] = None
       
       # Economic
       oil_price_per_bbl: Optional[float] = Field(None, gt=0)
       gas_price_per_mcf: Optional[float] = Field(None, gt=0)
       
       # Metadata
       created_at: datetime = Field(default_factory=datetime.now)
       updated_at: datetime = Field(default_factory=datetime.now)
       
       def total_production(self) -> Dict[str, float]:
           """Calculate total field production"""
           return {
               'oil_stb_day': sum(
                   w.current_oil_rate or 0 for w in self.wells
               ),
               'gas_mscf_day': sum(
                   w.current_gas_rate or 0 for w in self.wells
               ),
               'water_stb_day': sum(
                   w.current_water_rate or 0 for w in self.wells
               )
           }
       
       def active_wells(self) -> List[Well]:
           """Get all producing wells"""
           return [w for w in self.wells if w.status == WellStatus.PRODUCING]
       
       def field_water_cut(self) -> Optional[float]:
           """Calculate field-wide water cut"""
           prod = self.total_production()
           oil = prod['oil_stb_day']
           water = prod['water_stb_day']
           if oil + water > 0:
               return (water / (oil + water)) * 100
           return None
       
       def daily_revenue(self) -> Optional[float]:
           """Calculate daily field revenue"""
           if self.oil_price_per_bbl and self.gas_price_per_mcf:
               prod = self.total_production()
               oil_revenue = prod['oil_stb_day'] * self.oil_price_per_bbl
               gas_revenue = prod['gas_mscf_day'] * self.gas_price_per_mcf
               return oil_revenue + gas_revenue
           return None

5.2 Data Import/Export
^^^^^^^^^^^^^^^^^^^^^^^

Load field data from various sources:

.. code-block:: python

   import pandas as pd
   import json
   
   def load_wells_from_csv(filename: str) -> List[Well]:
       """Load wells from CSV file with validation"""
       df = pd.read_csv(filename)
       
       wells = []
       errors = []
       
       for idx, row in df.iterrows():
           try:
               well = Well(**row.to_dict())
               wells.append(well)
           except Exception as e:
               errors.append({
                   'row': idx,
                   'well_name': row.get('well_name'),
                   'error': str(e)
               })
       
       if errors:
           print(f"Failed to import {len(errors)} wells:")
           for err in errors:
               print(f"  Row {err['row']}: {err['error']}")
       
       print(f"Successfully imported {len(wells)} wells")
       return wells
   
   def export_field_to_json(field: Field, filename: str):
       """Export complete field data to JSON"""
       with open(filename, 'w') as f:
           f.write(field.model_dump_json(indent=2))
   
   def load_field_from_json(filename: str) -> Field:
       """Load field data from JSON with validation"""
       with open(filename, 'r') as f:
           data = json.load(f)
       return Field(**data)

----

Practice Exercises
------------------

Exercise 2.1: Build a Production Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a validated model for monthly production data:

**Requirements:**

1. Track oil, gas, and water production
2. Ensure all rates are non-negative
3. Calculate water cut and GOR
4. Validate that production date is not in the future
5. Store operating hours for the month (0-744)

.. code-block:: python

   from pydantic import BaseModel, Field
   from datetime import datetime

   class MonthlyProduction(BaseModel):
       well_id: str
       production_month: datetime
       operating_hours: float = Field(ge=0, le=744)
       oil_produced_stb: float = Field(ge=0)
       gas_produced_mscf: float = Field(ge=0)
       water_produced_stb: float = Field(ge=0)

       def water_cut(self) -> float:
           total = self.oil_produced_stb + self.water_produced_stb
           return (self.water_produced_stb / total * 100) if total > 0 else 0.0

       def gor(self) -> float:
           return (self.gas_produced_mscf * 1000 / self.oil_produced_stb) if self.oil_produced_stb > 0 else 0.0

**Solution:**

.. code-block:: python

   from pydantic import BaseModel, Field, validator
   from datetime import datetime
   
   class MonthlyProduction(BaseModel):
       well_id: str
       production_month: datetime
       operating_hours: float = Field(ge=0, le=744)
       
       oil_produced_stb: float = Field(ge=0)
       gas_produced_mscf: float = Field(ge=0)
       water_produced_stb: float = Field(ge=0)
       
       @validator('production_month')
       def not_future(cls, v):
           if v > datetime.now():
               raise ValueError('Production month cannot be in future')
           return v
       
       def water_cut(self) -> float:
           total_liquid = self.oil_produced_stb + self.water_produced_stb
           if total_liquid > 0:
               return (self.water_produced_stb / total_liquid) * 100
           return 0.0
       
       def gor(self) -> float:
           if self.oil_produced_stb > 0:
               return (self.gas_produced_mscf * 1000) / self.oil_produced_stb
           return 0.0

Exercise 2.2: Reservoir with Material Balance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Build a model that tracks reservoir depletion:

**Requirements:**

1. Initial and current pressure
2. Initial and produced volumes
3. Validate current pressure < initial pressure
4. Calculate recovery factor
5. Estimate remaining reserves

.. code-block:: python

   from pydantic import BaseModel, Field, model_validator

   class ReservoirDepletion(BaseModel):
       initial_pressure: float = Field(gt=0)
       current_pressure: float = Field(gt=0)
       initial_oil_in_place: float = Field(ge=0)
       cumulative_production: float = Field(ge=0)

       @model_validator(mode='after')
       def pressure_decline(self):
           if self.current_pressure >= self.initial_pressure:
               raise ValueError('current_pressure must be less than initial_pressure')
           return self

       @property
       def recovery_factor(self) -> float:
           return self.cumulative_production / self.initial_oil_in_place if self.initial_oil_in_place > 0 else 0.0

       @property
       def remaining_reserves(self) -> float:
           return self.initial_oil_in_place - self.cumulative_production

Exercise 2.3: Well Completion Validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a completion model with complex validation:

1. Different validation rules for openhole vs. cased completions
2. Ensure perforation intervals are within pay zone
3. Validate tubing/casing size compatibility
4. Check that packer depth makes sense

----

Summary and Key Takeaways
--------------------------

**What You Learned:**

✅ Building type-safe data models with Pydantic  
✅ Implementing field-level and model-level validation  
✅ Creating domain models for petroleum engineering  
✅ Serializing data for databases and APIs  
✅ Applying validation to real-world engineering problems  

**Best Practices:**

1. **Always validate at data entry** - Catch errors early
2. **Use enums for categorical data** - Prevent typos
3. **Provide clear error messages** - Help users fix issues
4. **Document with Field descriptions** - Self-documenting code
5. **Test edge cases** - Validate your validators!

**Next Steps:**

- Module 3: Testing and validation strategies
- Module 4: Apply these models to reservoir engineering
- Build your own models for your specific assets

----

Additional Resources
--------------------

**Pydantic Documentation:**

- Official docs: https://docs.pydantic.dev
- Field types: https://docs.pydantic.dev/latest/concepts/fields/
- Validators: https://docs.pydantic.dev/latest/concepts/validators/

**Related Modules:**

- :doc:`module1_fundamentals` - Python basics
- :doc:`module3_validation` - Testing strategies
- :doc:`module4_reservoir_engineering` - Apply to reservoir models

----

**Module Complete!** ✅

You now have the skills to build robust, validated data models for petroleum engineering applications.

**Time to complete:** 4-6 hours

**Next:** :doc:`module3_validation`
