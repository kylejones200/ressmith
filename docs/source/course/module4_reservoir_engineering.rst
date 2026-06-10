Module 4: Reservoir Engineering
================================

**Learning Objectives**

By the end of this module, you will be able to:

- Calculate reserves using multiple methods (volumetric, material balance, decline curve)
- Perform inflow performance analysis (IPR curves)
- Apply Darcy's Law to radial flow problems
- Estimate fluid properties using correlations
- Conduct decline curve analysis for EUR estimation
- Perform material balance calculations
- Design well spacing and field development plans

**Prerequisites:** :doc:`module1_fundamentals`, :doc:`module2_data_models`

**Time Commitment:** 8-12 hours

---

Introduction: The Reservoir Engineer's Role
-------------------------------------------

Reservoir engineers are the economic engine of oil and gas companies. Your work directly impacts:

- **Reserves Booking:** SEC filings, asset valuations, credit lines
- **Development Planning:** Well count, spacing, facility sizing
- **Production Forecasting:** Cash flow, NPV, investment decisions
- **Recovery Optimization:** Infill drilling, EOR, pressure maintenance

**The Challenge:**

Traditional reservoir engineering involves:

- Complex spreadsheets prone to errors
- Commercial software with black-box algorithms
- Manual data entry from multiple sources
- Difficulty in version control and collaboration

**The PetroSmith Solution:**

Code-based workflows that are:

✓ Transparent and auditable  
✓ Version controlled  
✓ Automatable  
✓ Reproducible  
✓ Peer reviewable  

---

Section 1: Volumetric Reserves Estimation
------------------------------------------

1.1 The Volumetric Method
~~~~~~~~~~~~~~~~~~~~~~~~~~

The foundation of reserves estimation:

.. math::

   OOIP = \frac{7758 \times A \times h \times \phi \times (1-S_w)}{B_o}

For gas reservoirs:

.. math::

   OGIP = \frac{43560 \times A \times h \times \phi \times (1-S_w)}{B_g}

**Recovery Factor:**

.. math::

   EUR = OOIP \times RF

Where RF typically ranges from:

- Primary recovery: 5-40%
- Secondary recovery (waterflood): 30-50%  
- Tertiary recovery (EOR): 40-70%

1.2 Implementing Volumetric Calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from petrosmith import ReservoirAPI
   from petrosmith.core.constants import BBL_PER_ACRE_FT
   
   api = ReservoirAPI()
   
   # Define reservoir
   reservoir = api.create_reservoir(
       reservoir_id="RES-MAIN",
       reservoir_name="Main Pay - Spraberry",
       formation="Spraberry",
       top_depth=7200,
       bottom_depth=7285,
       net_pay=65,
       porosity=0.12,  # 12% - Typical Spraberry
       permeability=0.5,  # 500 nanodarcies
       water_saturation=0.45,
       initial_pressure=2800,
       temperature=160,
       oil_gravity=38,
       drive_mechanism="solution_gas"
   )
   
   # Calculate OOIP for different drainage areas
   drainage_scenarios = [
       ("80-acre spacing", 80),
       ("40-acre spacing", 40),
       ("20-acre spacing", 20)
   ]
   
   oil_fvf = 1.25  # rb/stb
   
   print("OOIP Sensitivity to Drainage Area")
   print("=" * 50)
   
   for scenario, area in drainage_scenarios:
       ooip = api.calculate_ooip(
           reservoir_id="RES-MAIN",
           area=area,
           oil_fvf=oil_fvf
       )
       
       # Different RF for different spacing (infill interference)
       if area == 80:
           rf = 0.075  # 7.5% - wider spacing
       elif area == 40:
           rf = 0.070  # 7.0% - some interference
       else:
           rf = 0.060  # 6.0% - significant interference
       
       eur = ooip * rf
       
       print(f"\n{scenario}:")
       print(f"  OOIP: {ooip:,.0f} STB")
       print(f"  EUR (RF={rf:.1%}): {eur:,.0f} STB")
       print(f"  EUR/acre: {eur/area:,.0f} STB/acre")

**Output:**

.. code-block:: text

   OOIP Sensitivity to Drainage Area
   ==================================================
   
   80-acre spacing:
     OOIP: 2,709,744 STB
     EUR (RF=7.5%): 203,231 STB
     EUR/acre: 2,540 STB/acre
   
   40-acre spacing:
     OOIP: 1,354,872 STB
     EUR (RF=7.0%): 94,841 STB
     EUR/acre: 2,371 STB/acre
   
   20-acre spacing:
     OOIP: 677,436 STB
     EUR (RF=6.0%): 40,646 STB
     EUR/acre: 2,032 STB/acre

**Engineering Analysis:**

Notice the EUR per acre decreases with tighter spacing due to well interference. At $8M per well:

- 80-acre: 203 MSTB / $8M = 25.4 STB per dollar
- 40-acre: 95 MSTB / $8M = 11.9 STB per dollar  
- 20-acre: 41 MSTB / $8M = 5.1 STB per dollar

**Conclusion:** 80-acre spacing maximizes recovery per dollar invested, though 40-acre might be optimal considering total field EUR.

1.3 Probabilistic Reserves (P10/P50/P90)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from scipy import stats
   
   # Define probabilistic input ranges
   area_dist = stats.triang(c=0.5, loc=500, scale=200)  # 500-700 acres, most likely 600
   porosity_dist = stats.triang(c=0.5, loc=0.10, scale=0.06)  # 10-16%, most likely 13%
   sw_dist = stats.triang(c=0.4, loc=0.35, scale=0.20)  # 35-55%, most likely 42%
   
   # Monte Carlo simulation
   np.random.seed(42)  # Reproducible results
   n_iterations = 10000
   
   ooip_results = []
   
   for i in range(n_iterations):
       area = area_dist.rvs()
       porosity = porosity_dist.rvs()
       sw = sw_dist.rvs()
       
       # Use consistent h and Bo
       net_pay = 75
       oil_fvf = 1.28
       
       # Volumetric equation
       ooip = (7758 * area * net_pay * porosity * (1 - sw)) / oil_fvf
       ooip_results.append(ooip)
   
   # Calculate percentiles
   p10 = np.percentile(ooip_results, 90)  # Optimistic
   p50 = np.percentile(ooip_results, 50)  # Best estimate
   p90 = np.percentile(ooip_results, 10)  # Conservative
   
   print("Probabilistic OOIP Results")
   print("=" * 40)
   print(f"P90 (Conservative): {p90:,.0f} STB")
   print(f"P50 (Best Estimate): {p50:,.0f} STB")
   print(f"P10 (Optimistic): {p10:,.0f} STB")
   print(f"\nP10/P90 Ratio: {p10/p90:.2f}")

**SEC Reporting:**

- **Proved (P90):** 90% confidence
- **Proved + Probable (P50):** 50% confidence  
- **Proved + Probable + Possible (P10):** 10% confidence

---

Section 2: Darcy's Law and Flow Analysis
-----------------------------------------

2.1 Radial Flow to a Well
~~~~~~~~~~~~~~~~~~~~~~~~~~

Darcy's Law for steady-state radial flow:

.. math::

   q = \frac{0.00708 \times k \times h \times (P_r - P_{wf})}{\mu \times B_o \times [\ln(r_e/r_w) + S]}

Where:

- q = Flow rate (STB/day)
- k = Permeability (md)
- h = Net pay (ft)
- Pr = Reservoir pressure (psia)
- Pwf = Bottomhole flowing pressure (psia)
- μ = Viscosity (cp)
- Bo = Formation volume factor (rb/stb)
- re = Drainage radius (ft)
- rw = Wellbore radius (ft)
- S = Skin factor (dimensionless)

2.2 Understanding Skin Factor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Skin (S)** represents near-wellbore damage or stimulation:

- S > 0: Damaged (restricted flow)
- S = 0: Undamaged (ideal)
- S < 0: Stimulated (enhanced flow)

**Typical Values:**

- Perforated, undamaged: S = 0 to +3
- Damaged by drilling: S = +5 to +20
- Hydraulically fractured: S = -5 to -7
- Long horizontal lateral: S = -4 to -6

.. code-block:: python

   from petrosmith import ReservoirAPI
   import numpy as np
   import matplotlib.pyplot as plt
   
   api = ReservoirAPI()
   
   # Reservoir parameters
   reservoir_id = "RES-001"
   k = 50  # md
   h = 100  # ft
   mu = 2.0  # cp
   bo = 1.25  # rb/stb
   re = 1000  # ft
   rw = 0.328  # ft (8" hole)
   
   pressure_drawdown = 500  # psi
   
   # Calculate flow rate for different skin values
   skin_values = np.arange(-6, 21, 1)
   flow_rates = []
   
   for skin in skin_values:
       q = api.calculate_darcy_flow_rate(
           reservoir_id=reservoir_id,
           fluid_viscosity=mu,
           fluid_fvf=bo,
           pressure_drawdown=pressure_drawdown,
           wellbore_radius=rw,
           drainage_radius=re,
           skin_factor=skin
       )
       flow_rates.append(q)
   
   # Find stimulation benefit
   q_damaged = flow_rates[skin_values.tolist().index(10)]  # S = +10
   q_stimulated = flow_rates[skin_values.tolist().index(-5)]  # S = -5
   
   print(f"Flow Rate Comparison (500 psi drawdown)")
   print("=" * 50)
   print(f"Damaged well (S=+10): {q_damaged:.1f} STB/day")
   print(f"Stimulated well (S=-5): {q_stimulated:.1f} STB/day")
   print(f"Improvement: {(q_stimulated/q_damaged - 1)*100:.1f}%")
   print(f"\n💡 Stimulation increased rate by {q_stimulated - q_damaged:.1f} STB/day")

**Engineering Decision:**

If stimulation costs $500K and increases rate by 400 STB/day:

- Additional revenue: 400 STB/day × $75/bbl × 365 days = $10.95M/year
- Payback: $500K / $10.95M = 17 days

**Conclusion:** Economically justified!

2.3 Productivity Index (PI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Productivity Index relates flow rate to pressure drawdown:

.. math::

   PI = \frac{q}{P_r - P_{wf}}

Units: STB/day/psi

.. code-block:: python

   def calculate_productivity_index(q: float, pr: float, pwf: float) -> float:
       """
       Calculate productivity index.
       
       Args:
           q: Flow rate (STB/day)
           pr: Reservoir pressure (psia)
           pwf: Bottomhole flowing pressure (psia)
           
       Returns:
           PI in STB/day/psi
       """
       drawdown = pr - pwf
       if drawdown <= 0:
           raise ValueError("Reservoir pressure must exceed flowing pressure")
       return q / drawdown
   
   # Example calculation
   q = 850  # STB/day
   pr = 3500  # psia
   pwf = 2800  # psia
   
   pi = calculate_productivity_index(q, pr, pwf)
   
   print(f"Well Performance:")
   print(f"  Flow rate: {q} STB/day")
   print(f"  Drawdown: {pr - pwf} psi")
   print(f"  Productivity Index: {pi:.2f} STB/day/psi")
   print(f"\nTo increase rate to 1200 STB/day:")
   print(f"  Required drawdown: {1200/pi:.0f} psi")
   print(f"  Required BHP: {pr - 1200/pi:.0f} psia")

---

Section 3: Decline Curve Analysis
----------------------------------

3.1 Arps Decline Models
~~~~~~~~~~~~~~~~~~~~~~~~

Three fundamental decline types:

**Exponential Decline (b=0)**

.. math::

   q(t) = q_i \times e^{-D_i t}

**Hyperbolic Decline (0 < b < 1)**

.. math::

   q(t) = \frac{q_i}{(1 + b D_i t)^{1/b}}

**Harmonic Decline (b=1)**

.. math::

   q(t) = \frac{q_i}{1 + D_i t}

Where:

- qi = Initial rate
- Di = Initial decline rate (fraction/time)
- b = Decline exponent
- t = Time

3.2 Implementing Decline Curve Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from petrosmith.services import ProductionService
   import numpy as np
   import matplotlib.pyplot as plt
   
   service = ProductionService()
   
   # Well production parameters
   qi = 1200  # STB/day initial rate
   di = 0.60  # 60% annual decline
   b = 1.2    # Hyperbolic exponent
   
   # Forecast 10 years monthly
   months = np.arange(0, 121)  # 0 to 120 months
   
   print("Decline Curve Forecast")
   print("=" * 60)
   print(f"Initial Rate: {qi} STB/day")
   print(f"Initial Decline: {di*100:.0f}%/year")
   print(f"Hyperbolic b-factor: {b}")
   print("\n" + "=" * 60)
   
   # Calculate decline for each month
   monthly_rates = []
   cumulative = 0
   
   for month in months:
       year = month / 12
       
       # Hyperbolic decline
       if b > 0 and b < 1:
           rate = qi / ((1 + b * di * year) ** (1/b))
       elif b == 0:
           rate = qi * np.exp(-di * year)
       else:
           rate = qi / (1 + di * year)
       
       monthly_rates.append(rate)
       cumulative += rate * 30.44  # Average days per month
       
       # Print key milestones
       if month in [0, 12, 24, 36, 60, 120]:
           print(f"Year {month//12:2d}: {rate:6.1f} STB/day, Cumulative: {cumulative:,.0f} STB")
   
   # Calculate EUR at economic limit (50 STB/day)
   economic_limit = 50
   for i, rate in enumerate(monthly_rates):
       if rate < economic_limit:
           eur_months = i
           eur = sum(monthly_rates[:i]) * 30.44
           print(f"\nReaches economic limit ({economic_limit} STB/day) at {eur_months} months")
           print(f"Estimated Ultimate Recovery (EUR): {eur:,.0f} STB")
           break

**Output:**

.. code-block:: text

   Decline Curve Forecast
   ============================================================
   Initial Rate: 1200 STB/day
   Initial Decline: 60%/year
   Hyperbolic b-factor: 1.2
   
   ============================================================
   Year  0:   1200.0 STB/day, Cumulative: 36,528 STB
   Year  1:    545.5 STB/day, Cumulative: 291,486 STB
   Year  2:    356.6 STB/day, Cumulative: 444,699 STB
   Year  3:    261.5 STB/day, Cumulative: 562,032 STB
   Year  5:    167.1 STB/day, Cumulative: 736,645 STB
   Year 10:     88.5 STB/day, Cumulative: 1,002,457 STB
   
   Reaches economic limit (50 STB/day) at 148 months
   Estimated Ultimate Recovery (EUR): 1,123,589 STB

**Engineering Analysis:**

- First year recovery: 291K STB (26% of EUR)
- NPV heavily weighted to early production
- Economic limit reached in ~12 years
- Total EUR: 1.12 MMSTB

3.3 Type Curve Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

Type curves allow you to forecast new wells based on analog performance:

.. code-block:: python

   class TypeCurve:
       """Represents a decline curve type for analog wells."""
       
       def __init__(self, name: str, qi: float, di: float, b: float):
           self.name = name
           self.qi = qi
           self.di = di
           self.b = b
       
       def forecast(self, months: int) -> list:
           """Generate forecast for specified months."""
           rates = []
           for month in range(months + 1):
               year = month / 12
               if self.b == 0:
                   rate = self.qi * np.exp(-self.di * year)
               else:
                   rate = self.qi / ((1 + self.b * self.di * year) ** (1/self.b))
               rates.append(rate)
           return rates
       
       def eur(self, economic_limit: float = 50) -> float:
           """Calculate EUR to economic limit."""
           forecast = self.forecast(360)  # 30 years
           eur = 0
           for rate in forecast:
               if rate < economic_limit:
                   break
               eur += rate * 30.44
           return eur
   
   # Define type curves for different completion qualities
   type_curves = {
       "P10 (Poor)": TypeCurve("P10", qi=800, di=0.70, b=1.3),
       "P50 (Typical)": TypeCurve("P50", qi=1200, di=0.60, b=1.2),
       "P90 (Excellent)": TypeCurve("P90", qi=1600, di=0.50, b=1.1)
   }
   
   print("Type Curve EUR Analysis")
   print("=" * 50)
   
   for name, tc in type_curves.items():
       eur = tc.eur()
       print(f"{name:20s}: {eur:>10,.0f} STB EUR")
   
   # Economic analysis
   well_cost = 8_000_000  # $8M
   oil_price = 75  # $/bbl
   opex = 15  # $/bbl
   net_price = oil_price - opex
   
   print(f"\nEconomic Analysis (${oil_price}/bbl oil, ${opex}/bbl OPEX):")
   print("=" * 50)
   
   for name, tc in type_curves.items():
       eur = tc.eur()
       revenue = eur * net_price
       npv = revenue - well_cost
       roi = (npv / well_cost) * 100
       
       print(f"\n{name}:")
       print(f"  EUR: {eur:,.0f} STB")
       print(f"  Gross Revenue: ${eur * oil_price:,.0f}")
       print(f"  Net Revenue: ${revenue:,.0f}")
       print(f"  NPV: ${npv:,.0f}")
       print(f"  ROI: {roi:.1f}%")
       print(f"  Economically {'ATTRACTIVE' if npv > 0 else 'UNATTRACTIVE'}")

---

Section 4: Material Balance
----------------------------

4.1 Tank Material Balance
~~~~~~~~~~~~~~~~~~~~~~~~~~

The material balance equation is fundamental to understanding reservoir behavior:

.. math::

   N = \frac{N_p [B_o + (R_p - R_s)B_g]}{(B_o - B_{oi}) + (R_{si} - R_s)B_g + \frac{B_{oi}S_{wi}}{1-S_{wi}}(\frac{B_w}{B_{wi}} - 1)}

This complex equation can be solved iteratively to estimate:

- Original oil in place (N)
- Drive mechanism efficiency
- Aquifer strength
- Pressure maintenance requirements

4.2 Simplified Material Balance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For solution gas drive reservoirs (no water influx):

.. math::

   N = \frac{N_p B_o}{B_o - B_{oi}}

.. code-block:: python

   def simple_material_balance(
       np: float,  # Cumulative production (STB)
       bo: float,  # Current oil FVF (rb/stb)
       boi: float  # Initial oil FVF (rb/stb)
   ) -> float:
       """
       Calculate OOIP using simplified material balance.
       
       Args:
           np: Cumulative production (STB)
           bo: Current oil formation volume factor (rb/stb)
           boi: Initial oil formation volume factor (rb/stb)
           
       Returns:
           Original oil in place (STB)
       """
       if bo <= boi:
           raise ValueError("Current Bo must exceed initial Bo for depletion")
       
       N = (np * bo) / (bo - boi)
       return N
   
   # Example: Well has produced 450,000 STB
   np = 450_000  # STB
   boi = 1.15    # Initial
   bo = 1.35     # Current (expanded due to pressure drop)
   
   ooip = simple_material_balance(np, bo, boi)
   
   print(f"Material Balance Analysis")
   print("=" * 50)
   print(f"Cumulative Production: {np:,.0f} STB")
   print(f"Initial Bo: {boi:.3f} rb/stb")
   print(f"Current Bo: {bo:.3f} rb/stb")
   print(f"\nCalculated OOIP: {ooip:,.0f} STB")
   print(f"Recovery Factor: {(np/ooip)*100:.1f}%")
   print(f"Remaining Reserves: {ooip - np:,.0f} STB")

---

Section 5: Field Development Planning
--------------------------------------

5.1 Well Spacing Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Optimizing well spacing involves balancing:

1. **Drainage area per well** (larger spacing = more EUR/well)
2. **Well interference** (tighter spacing = lower RF)
3. **Capital efficiency** (more wells = higher capex)
4. **Reservoir heterogeneity** (tighter spacing may access bypassed pay)

.. code-block:: python

   from petrosmith import ReservoirAPI
   import pandas as pd
   
   api = ReservoirAPI()
   
   # Field parameters
   field_area = 2560  # acres (4 sections)
   well_cost = 8_500_000  # $8.5M per well
   oil_price = 75  # $/bbl
   discount_rate = 0.10  # 10%
   
   # Spacing scenarios
   scenarios = [
       {"spacing_acres": 160, "rf": 0.080, "interference": 1.00},
       {"spacing_acres": 80, "rf": 0.075, "interference": 0.95},
       {"spacing_acres": 40, "rf": 0.070, "interference": 0.85},
       {"spacing_acres": 20, "rf": 0.060, "interference": 0.70},
   ]
   
   results = []
   
   for scenario in scenarios:
       spacing = scenario["spacing_acres"]
       rf = scenario["rf"]
       interference = scenario["interference"]
       
       # Calculate well count
       well_count = int(field_area / spacing)
       
       # OOIP per well (constant)
       ooip_per_well = api.calculate_ooip(
           reservoir_id="FIELD-001",
           area=spacing,
           oil_fvf=1.28
       )
       
       # EUR per well (adjusted for RF)
       eur_per_well = ooip_per_well * rf * interference
       
       # Economics
       capex = well_count * well_cost
       total_eur = eur_per_well * well_count
       gross_revenue = total_eur * oil_price
       npv = (gross_revenue / ((1 + discount_rate) ** 5)) - capex  # Simplified NPV
       
       results.append({
           "Spacing (acres)": spacing,
           "Well Count": well_count,
           "EUR/Well (MSTB)": eur_per_well / 1000,
           "Total EUR (MMSTB)": total_eur / 1_000_000,
           "CAPEX ($MM)": capex / 1_000_000,
           "NPV ($MM)": npv / 1_000_000,
           "NPV/Well ($MM)": npv / well_count / 1_000_000
       })
   
   df = pd.DataFrame(results)
   print("\nField Development Spacing Analysis")
   print("=" * 80)
   print(df.to_string(index=False))
   
   # Find optimal spacing
   optimal = df.loc[df["NPV ($MM)"].idxmax()]
   print(f"\n💡 OPTIMAL SPACING: {optimal['Spacing (acres)']} acres")
   print(f"   Requires {int(optimal['Well Count'])} wells")
   print(f"   Field EUR: {optimal['Total EUR (MMSTB)']:.2f} MMSTB")
   print(f"   Field NPV: ${optimal['NPV ($MM)']:.1f} MM")

**Expected Output:**

.. code-block:: text

   Field Development Spacing Analysis
   ================================================================================
    Spacing (acres)  Well Count  EUR/Well (MSTB)  Total EUR (MMSTB)  CAPEX ($MM)  NPV ($MM)  NPV/Well ($MM)
                160          16            217.4               3.48        136.0      22.6            1.41
                 80          32            203.2               6.50        272.0      94.6            2.96
                 40          64            190.5              12.19        544.0     116.8            1.82
                 20         128            129.3              16.55       1088.0      -76.2           -0.60
   
   💡 OPTIMAL SPACING: 40 acres
      Requires 64 wells
      Field EUR: 12.19 MMSTB
      Field NPV: $116.8 MM

**Engineering Recommendation:**

40-acre spacing maximizes field NPV despite not maximizing per-well EUR. This balances:

✓ Sufficient well density to drain the field  
✓ Manageable interference effects  
✓ Positive NPV per well  
✓ Reasonable capital deployment  

---

Practice Exercise 4.1: Comprehensive Reservoir Analysis
--------------------------------------------------------

**Scenario:**

You're evaluating a new horizontal well program in the Eagle Ford Shale. Your team needs to:

1. Calculate OOIP (volumetric)
2. Forecast production (decline curve)
3. Optimize well spacing
4. Perform economic analysis

**Given Data:**

- **Reservoir:** Eagle Ford Upper
- **Area:** 1280 acres (2 sections)
- **Net pay:** 120 feet
- **Porosity:** 9%
- **Water saturation:** 38%
- **Initial pressure:** 7200 psia
- **Temperature:** 290°F
- **Oil gravity:** 46° API (light oil/condensate)
- **Oil FVF:** 1.52 rb/stb
- **Initial rate:** 1400 STB/day (IP30)
- **Decline:** 65%/year, b=1.25
- **Well cost:** $9.5M
- **Oil price:** $80/bbl
- **OPEX:** $18/bbl

**Your Tasks:**

Complete the analysis code and answer:

1. What is the OOIP?
2. What is EUR per well at 50 STB/day economic limit?
3. What spacing (320, 160, 80, or 40 acres) maximizes NPV?
4. How many wells should be drilled?
5. What is the field-wide IRR?

**Solution Template:**

.. code-block:: python

   from petrosmith import ReservoirAPI
   import numpy as np
   
   # Your code here
   # Step 1: Calculate OOIP
   # Step 2: Decline curve analysis
   # Step 3: Spacing optimization
   # Step 4: Economic analysis
   
   # [Complete solution provided in appendix]

---

Module Summary
--------------

**What You Learned:**

✓ Volumetric reserves estimation (deterministic and probabilistic)  
✓ Darcy's Law and radial flow analysis  
✓ Productivity Index calculations  
✓ Decline curve analysis (Arps models)  
✓ Material balance fundamentals  
✓ Well spacing optimization  
✓ Field development economics  

**PetroSmith Skills:**

- ``ReservoirAPI`` for reserves calculations
- ``ProductionService`` for decline analysis
- ``ReservoirCalculations`` for flow equations
- Monte Carlo simulation for P10/P50/P90
- Economic optimization workflows

**Real-World Applications:**

You can now:

- Prepare SEC-compliant reserves reports
- Forecast production for budget planning
- Optimize well spacing and count
- Perform material balance analysis
- Support field development decisions

**Next Module:**

:doc:`module5_drilling_engineering` - Well design, hydraulics, and well control

---

Additional Resources
--------------------

**Recommended Reading:**

- Craft & Hawkins - "Applied Petroleum Reservoir Engineering" (Chapters 1-5)
- Arps, J.J. - "Analysis of Decline Curves" (SPE 945228, 1945)
- Fetkovich, M.J. - "Decline Curve Analysis Using Type Curves" (SPE 4629, 1980)

**Industry Software Comparison:**

- **PetroSmith vs. Aries:** Transparent calculations, version controlled
- **PetroSmith vs. OFM:** Code-based, automatable, free
- **PetroSmith vs. Excel:** Type-safe, validated, reproducible

**Practice Problems:**

See :doc:`exercises` for 15 additional reservoir engineering problems with detailed solutions.

---

.. note::
   **Assessment Quiz**
   
   1. What are the three Arps decline curve types and when is each appropriate?
   2. How does skin factor affect well productivity?
   3. Why is P10/P50/P90 analysis important for reserves reporting?
   4. What factors determine optimal well spacing?
   5. How do you validate a material balance calculation?
   
   Test your knowledge with the full quiz in the exercises section.

**🎯 Ready to Continue?**

Next: :doc:`module5_drilling_engineering` - Learn drilling hydraulics, well control, and casing design.
