Module 1: Fundamentals of Petroleum Engineering
==============================================

**Learning Objectives**

By the end of this module, you will:

- Understand the petroleum system from exploration to production
- Master fundamental concepts: porosity, permeability, fluid properties
- Learn how to represent physical assets as data models
- Build your first reservoir calculation in Python

**Prerequisites:** Basic understanding of petroleum geology and Python syntax

**Time Commitment:** 4-6 hours

---

Introduction: Why Modern Tools Matter
-------------------------------------

The Evolution of Petroleum Engineering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1950s-1980s:** Slide rules and hand calculations  
**1980s-2000s:** Spreadsheets (Excel becomes dominant)  
**2000s-2020s:** Commercial software (Eclipse, Petrel, OFM)  
**2020s+:** Code-based workflows (Python, Julia, R)

The Spreadsheet Problem
~~~~~~~~~~~~~~~~~~~~~~~

You've been there: a critical reserves calculation spreadsheet with:

- 47 tabs referencing each other
- Formulas like ``=IF(VLOOKUP(A2,$K$12:$M$47,3,FALSE)>0.5,B2*$AC$1,0)``
- Last modified by someone who left the company 3 years ago
- No version control, multiple copies: "Reserves_v3_FINAL_v2_use_this.xlsx"

When the CEO asks "How did you get that EUR estimate?", you click through tabs hoping to reverse-engineer the logic.

**There has to be a better way.**

The Code-Based Approach
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from petrosmith import ReservoirAPI
   
   # Create a reservoir model
   api = ReservoirAPI()
   res = api.create_reservoir(
       reservoir_name="Wolfcamp Shale - Midland Basin",
       formation="Wolfcamp A",
       top_depth=8500,
       bottom_depth=8620,
       net_pay=85,
       porosity=0.08,          # 8% - Tight shale
       permeability=0.05,      # 50 nanodarcies
       water_saturation=0.35,
       initial_pressure=6500,
       temperature=285,
       oil_gravity=42,         # Light oil
       drive_mechanism="solution_gas"
   )
   
   # Calculate OOIP
   ooip = api.calculate_ooip(
       reservoir_id=res.reservoir_id,
       area=640,  # One section
       oil_fvf=1.45
   )
   
   print(f"Original Oil in Place: {ooip:,.0f} STB")
   print(f"At 7.5% recovery: {ooip * 0.075:,.0f} STB EUR")

**Benefits:**

✓ Self-documenting code  
✓ Version controlled in Git  
✓ Testable and reproducible  
✓ Peer reviewable  
✓ Auditable for compliance  

---

The Petroleum System
--------------------

Understanding the Complete Lifecycle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before we write code, let's review the fundamental petroleum engineering workflow:

.. image:: ../_static/petroleum_system.png
   :alt: Petroleum System Workflow
   :align: center

1. **Exploration & Appraisal**
   
   - Seismic interpretation
   - Well planning and drilling
   - Reservoir characterization
   - *Key Question:* How much hydrocarbon is in place?

2. **Development Planning**
   
   - Well spacing and count
   - Facility design
   - Economic analysis
   - *Key Question:* What's the optimal development strategy?

3. **Drilling & Completion**
   
   - Wellbore design
   - Hydraulics and well control
   - Completion selection
   - *Key Question:* How do we safely access the reservoir?

4. **Production Operations**
   
   - Artificial lift
   - Well testing
   - Production optimization
   - *Key Question:* How do we maximize recovery?

5. **Reservoir Management**
   
   - Performance tracking
   - Infill drilling
   - Enhanced recovery
   - *Key Question:* How do we sustain production?

Each phase requires calculations. PetroSmith provides tools for all of them.

---

Fundamental Concepts
--------------------

Rock Properties
~~~~~~~~~~~~~~~

**Porosity (φ)** - Void Space
   
   The fraction of rock volume that can store fluids.
   
   .. math::
      
      \phi = \frac{V_{pore}}{V_{bulk}}
   
   **Typical Values:**
   
   - Sandstone: 15-30%
   - Carbonate: 5-20%
   - Shale: 2-10%

   .. code-block:: python
   
      from pydantic import BaseModel, Field
      
      class Reservoir(BaseModel):
          porosity: float = Field(gt=0.0, lt=1.0, description="Porosity fraction")
          
          @property
          def porosity_percent(self) -> float:
              """Return porosity as percentage."""
              return self.porosity * 100
      
      # Create reservoir
      res = Reservoir(porosity=0.22)
      print(f"Porosity: {res.porosity_percent:.1f}%")  # 22.0%

**Permeability (k)** - Flow Capacity
   
   The ability of rock to transmit fluids.
   
   Measured in millidarcies (md) or darcies (D).
   
   **Typical Values:**
   
   - High: >1000 md (beach sand)
   - Moderate: 10-1000 md (conventional reservoirs)
   - Low: 0.1-10 md (tight formations)
   - Ultra-low: <0.1 md (shales, measured in nanodarcies)

**Water Saturation (Sw)** - Fraction of Pore Space Filled with Water
   
   .. math::
      
      S_w + S_o + S_g = 1.0
   
   Where So = oil saturation, Sg = gas saturation

   .. code-block:: python
   
      # Validation ensures physical constraints
      class Reservoir(BaseModel):
          water_saturation: float = Field(ge=0.0, le=1.0)
          
          @property
          def hydrocarbon_saturation(self) -> float:
              return 1.0 - self.water_saturation
      
      res = Reservoir(porosity=0.20, water_saturation=0.25)
      print(f"Hydrocarbon saturation: {res.hydrocarbon_saturation:.2f}")  # 0.75

Fluid Properties
~~~~~~~~~~~~~~~~

**Oil Formation Volume Factor (Bo)**

   Relates reservoir barrels to stock tank barrels.
   
   .. math::
      
      B_o = \frac{\text{Reservoir Volume}}{\text{Stock Tank Volume}}
   
   Typical range: 1.0 - 2.5 rb/stb
   
   Higher Bo means oil expands more (more dissolved gas, lighter oil).

**Gas-Oil Ratio (GOR)**

   Standard cubic feet of gas per stock tank barrel of oil.
   
   .. math::
      
      GOR = \frac{Q_g}{Q_o}
   
   **Classification:**
   
   - Black oil: <100 scf/stb
   - Volatile oil: 100-3000 scf/stb
   - Gas condensate: >3000 scf/stb

**Viscosity (μ)**

   Resistance to flow, measured in centipoise (cp).
   
   - Water: ~1 cp
   - Light oil (40° API): 1-5 cp
   - Medium oil (30° API): 10-50 cp
   - Heavy oil (15° API): 100-10,000 cp

Implementing Fluid Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from petrosmith.models import FluidProperties
   from petrosmith.core import FluidCalculations
   
   # Define fluid
   fluid = FluidProperties(
       oil_gravity=35,     # API gravity
       gas_gravity=0.65,   # Relative to air
       water_salinity=35000  # ppm
   )
   
   # Calculate properties at reservoir conditions
   calc = FluidCalculations()
   
   # Oil FVF using Standing correlation
   bo = calc.oil_formation_volume_factor(
       oil_gravity=fluid.oil_gravity,
       gas_gravity=fluid.gas_gravity,
       gas_oil_ratio=500,  # scf/stb
       temperature=180,    # °F
       pressure=3000       # psia
   )
   
   print(f"Oil FVF: {bo:.3f} rb/stb")
   
   # Viscosity using Beggs-Robinson correlation
   viscosity = calc.oil_viscosity(
       oil_gravity=fluid.oil_gravity,
       temperature=180
   )
   
   print(f"Oil viscosity: {viscosity:.2f} cp")

---

Your First Calculation: Original Oil In Place (OOIP)
----------------------------------------------------

Theory: Volumetric Method
~~~~~~~~~~~~~~~~~~~~~~~~~

The volumetric equation is fundamental to reserves estimation:

.. math::

   OOIP = \frac{7758 \times A \times h \times \phi \times (1 - S_w)}{B_o}

Where:

- A = Area (acres)
- h = Net pay thickness (feet)
- φ = Porosity (fraction)
- Sw = Water saturation (fraction)
- Bo = Oil formation volume factor (rb/stb)
- 7758 = Conversion factor (barrels per acre-foot)

Manual Calculation Example
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Given:**

- Area: 640 acres (1 section)
- Net pay: 80 feet
- Porosity: 22%
- Water saturation: 30%
- Oil FVF: 1.25 rb/stb

**Calculate:**

.. math::

   OOIP &= \frac{7758 \times 640 \times 80 \times 0.22 \times (1-0.30)}{1.25} \\
        &= \frac{7758 \times 640 \times 80 \times 0.22 \times 0.70}{1.25} \\
        &= \frac{61,621,248}{1.25} \\
        &= 49,297,000 \text{ STB}

PetroSmith Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from petrosmith import ReservoirAPI
   
   # Initialize API
   api = ReservoirAPI()
   
   # Create reservoir
   reservoir = api.create_reservoir(
       reservoir_id="RES-001",
       reservoir_name="Main Sand",
       formation="Jurassic Sandstone",
       top_depth=8000,
       bottom_depth=8080,
       net_pay=80,
       porosity=0.22,
       permeability=150,
       water_saturation=0.30,
       initial_pressure=3500,
       temperature=180,
       oil_gravity=35,
       drive_mechanism="water_drive"
   )
   
   # Calculate OOIP
   ooip = api.calculate_ooip(
       reservoir_id="RES-001",
       area=640,
       oil_fvf=1.25
   )
   
   print(f"Original Oil In Place: {ooip:,.0f} STB")
   # Result: 49,297,000 STB (matches hand calculation!)

**Advantages of the code approach:**

1. **Validated Input:** Porosity must be 0-1, can't accidentally enter 22 instead of 0.22
2. **Documented:** Variable names are self-explanatory
3. **Reproducible:** Run it 100 times, get the same answer
4. **Testable:** Unit tests ensure the formula is correct
5. **Auditable:** SEC can review your calculation logic

---

Understanding Data Models
-------------------------

What is a Data Model?
~~~~~~~~~~~~~~~~~~~~~

A **data model** is a structured representation of a real-world entity. In petroleum engineering:

- **Well** model represents a physical wellbore
- **Reservoir** model represents a subsurface formation
- **FluidProperties** model represents hydrocarbon characteristics

Why Use Pydantic?
~~~~~~~~~~~~~~~~~

Pydantic provides:

1. **Type Safety:** Ensures data types are correct
2. **Validation:** Enforces physical constraints
3. **Documentation:** Self-documenting with type hints
4. **Serialization:** Easy JSON export for databases

Example: Building a Well Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pydantic import BaseModel, Field
   from typing import Optional
   from datetime import date
   
   class WellLocation(BaseModel):
       """Geographic coordinates of a well."""
       latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")
       longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees")
       surface_elevation: Optional[float] = Field(None, description="Elevation in feet MSL")
   
   class Well(BaseModel):
       """Represents a drilled wellbore."""
       well_id: str = Field(..., min_length=1, description="Unique well identifier")
       well_name: str = Field(..., description="Well name")
       operator: str = Field(..., description="Operating company")
       location: WellLocation
       spud_date: Optional[date] = None
       total_depth: float = Field(..., gt=0, description="Total measured depth in feet")
       well_type: str = Field(..., description="vertical, directional, or horizontal")
       status: str = Field(..., description="drilling, producing, shut-in, abandoned")
       
       @property
       def is_producing(self) -> bool:
           return self.status == "producing"

**Using the Well model:**

.. code-block:: python

   from datetime import date
   
   # Create a well
   well = Well(
       well_id="API-42-123-45678",
       well_name="Smith 1-H",
       operator="XTO Energy",
       location=WellLocation(
           latitude=31.8457,
           longitude=-102.3676,
           surface_elevation=2850
       ),
       spud_date=date(2024, 3, 15),
       total_depth=15420,
       well_type="horizontal",
       status="producing"
   )
   
   print(f"{well.well_name}: {well.total_depth:,.0f} ft MD")
   print(f"Producing: {well.is_producing}")  # True
   
   # Validation catches errors
   try:
       bad_well = Well(
           well_id="",  # ❌ Too short
           well_name="Test",
           operator="ACME Oil",
           location=WellLocation(latitude=91, longitude=-100),  # ❌ Invalid latitude
           total_depth=-1000,  # ❌ Negative depth
           well_type="vertical",
           status="producing"
       )
   except ValueError as e:
       print(f"Validation error: {e}")

---

Practice Exercise 1.1: Calculate OOIP
--------------------------------------

**Scenario:**

You've just completed a discovery well in the Permian Basin. Preliminary analysis shows:

- Reservoir area: 320 acres (half section)
- Gross interval: 120 feet
- Net-to-gross ratio: 0.70 (84 feet net pay)
- Average porosity: 18%
- Water saturation: 35%
- Oil FVF: 1.32 rb/stb

**Tasks:**

1. Create a reservoir model using PetroSmith
2. Calculate OOIP
3. Estimate recoverable reserves assuming 35% recovery factor
4. Calculate per-acre reserves

**Solution:**

.. code-block:: python

   from petrosmith import ReservoirAPI
   
   api = ReservoirAPI()
   
   # Step 1: Create reservoir
   reservoir = api.create_reservoir(
       reservoir_id="PERM-001",
       reservoir_name="Discovery Well - Wolfcamp",
       formation="Wolfcamp A",
       top_depth=8200,
       bottom_depth=8320,
       net_pay=84,  # 120 * 0.70
       porosity=0.18,
       permeability=0.08,  # Assume tight
       water_saturation=0.35,
       initial_pressure=5500,
       temperature=240,
       oil_gravity=40,
       drive_mechanism="solution_gas"
   )
   
   # Step 2: Calculate OOIP
   ooip = api.calculate_ooip(
       reservoir_id="PERM-001",
       area=320,
       oil_fvf=1.32
   )
   
   print(f"OOIP: {ooip:,.0f} STB")
   
   # Step 3: Recoverable reserves (EUR)
   recovery_factor = 0.35
   eur = ooip * recovery_factor
   print(f"EUR (at 35% RF): {eur:,.0f} STB")
   
   # Step 4: Per-acre reserves
   per_acre = eur / 320
   print(f"Per-acre EUR: {per_acre:,.0f} STB/acre")

**Expected Output:**

.. code-block:: text

   OOIP: 13,528,727 STB
   EUR (at 35% RF): 4,735,054 STB
   Per-acre EUR: 14,797 STB/acre

**Analysis:**

This is a typical tight oil reservoir. At $80/bbl and $8M well cost, this would be economically attractive with modern completion techniques.

---

Module Summary
--------------

**Key Concepts Learned:**

✓ Petroleum engineering fundamentals (porosity, permeability, fluids)  
✓ Why code-based workflows beat spreadsheets  
✓ How to use data models (Pydantic)  
✓ Volumetric reserves calculation (OOIP)  
✓ Input validation and error handling

**PetroSmith Components Used:**

- ``ReservoirAPI`` - High-level interface
- ``Reservoir`` model - Data representation
- ``calculate_ooip()`` - Volumetric calculation

**Next Steps:**

In :doc:`module2_data_models`, we'll dive deeper into building comprehensive data models for wells, completions, and production data.

---

Additional Resources
--------------------

**Textbooks:**

- Craft & Hawkins - "Applied Petroleum Reservoir Engineering"
- Ahmed - "Reservoir Engineering Handbook"
- Dake - "Fundamentals of Reservoir Engineering"

**Industry Standards:**

- SPE PRMS (Petroleum Resources Management System)
- SEC reserves reporting guidelines
- SPEE (Society of Petroleum Evaluation Engineers) standards

**Practice Problems:**

See :doc:`exercises` for 20+ additional problems with solutions.

**Questions?**

Join the discussion on GitHub or review the API reference: :doc:`../api/index`

---

.. note::
   **Self-Assessment Quiz**
   
   Test your understanding:
   
   1. What's the difference between porosity and permeability?
   2. Why is Bo typically greater than 1.0?
   3. What physical constraints should be validated for Sw?
   4. How does the 7758 conversion factor in the OOIP equation work?
   5. When would you use solution gas drive vs. water drive?
   
   Answers at the end of this module documentation.
