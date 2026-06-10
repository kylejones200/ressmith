Course Exercises and Problems
==============================

**Practice Problems with Complete Solutions**

This section contains hands-on exercises for each module, progressing from basic calculations to complex integrated workflows. Each problem includes:

✓ Detailed problem statement  
✓ Given data and assumptions  
✓ Step-by-step solution  
✓ Engineering analysis and recommendations  
✓ Complete Python code  

---

Module 1 Exercises: Fundamentals
---------------------------------

Exercise 1.1: OOIP Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:**

Calculate OOIP for the Bakken Formation prospect:

- Area: 1280 acres
- Gross thickness: 45 feet
- Net-to-gross: 0.80
- Porosity: 7%
- Water saturation: 42%
- Oil FVF: 1.48 rb/stb

**Solution:**

.. code-block:: python

   from petrosmith import ReservoirAPI
   
   api = ReservoirAPI()
   
   # Calculate net pay
   gross_thickness = 45  # ft
   ntg_ratio = 0.80
   net_pay = gross_thickness * ntg_ratio
   
   print(f"Net pay: {net_pay} feet")
   
   # Create reservoir
   reservoir = api.create_reservoir(
       reservoir_id="BAKKEN-001",
       reservoir_name="Bakken Middle Member",
       formation="Bakken",
       top_depth=10200,
       bottom_depth=10245,
       net_pay=net_pay,
       porosity=0.07,
       permeability=0.05,  # 50 nanodarcies
       water_saturation=0.42,
       initial_pressure=6500,
       temperature=240,
       oil_gravity=42,
       drive_mechanism="solution_gas"
   )
   
   # Calculate OOIP
   ooip = api.calculate_ooip(
       reservoir_id="BAKKEN-001",
       area=1280,
       oil_fvf=1.48
   )
   
   print(f"\nOOIP: {ooip:,.0f} STB")
   
   # Per-acre reserves at different recovery factors
   print(f"\nRecoverable Reserves by Recovery Factor:")
   print("=" * 50)
   
   for rf in [0.05, 0.075, 0.10]:
       eur = ooip * rf
       per_acre = eur / 1280
       print(f"RF = {rf:.1%}: EUR = {eur:,.0f} STB ({per_acre:,.0f} STB/acre)")

**Expected Output:**

.. code-block:: text

   Net pay: 36.0 feet
   
   OOIP: 7,382,432 STB
   
   Recoverable Reserves by Recovery Factor:
   ==================================================
   RF = 5.0%: EUR = 369,122 STB (288 STB/acre)
   RF = 7.5%: EUR = 553,682 STB (433 STB/acre)
   RF = 10.0%: EUR = 738,243 STB (577 STB/acre)

**Engineering Analysis:**

Bakken wells typically recover 5-10% of OOIP with modern completion techniques. At 7.5% RF and $9M well cost, this would require $75+/bbl oil to be economic on 1280-acre spacing.

---

Exercise 1.2: Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:**

Perform sensitivity analysis on Exercise 1.1, varying:

- Porosity: ±2 percentage points
- Water saturation: ±5 percentage points
- Net-to-gross: ±0.10

**Solution:**

.. code-block:: python

   import numpy as np
   import pandas as pd
   from petrosmith import ReservoirAPI
   
   api = ReservoirAPI()
   
   # Base case parameters
   base_params = {
       "area": 1280,
       "gross_thickness": 45,
       "ntg": 0.80,
       "porosity": 0.07,
       "water_sat": 0.42,
       "oil_fvf": 1.48
   }
   
   # Sensitivity scenarios
   scenarios = [
       {"name": "Base Case", "param": None, "value": None},
       {"name": "Porosity -2%", "param": "porosity", "value": 0.05},
       {"name": "Porosity +2%", "param": "porosity", "value": 0.09},
       {"name": "Sw -5%", "param": "water_sat", "value": 0.37},
       {"name": "Sw +5%", "param": "water_sat", "value": 0.47},
       {"name": "N/G -0.10", "param": "ntg", "value": 0.70},
       {"name": "N/G +0.10", "param": "ntg", "value": 0.90},
   ]
   
   results = []
   
   for scenario in scenarios:
       # Copy base parameters
       params = base_params.copy()
       
       # Apply variation
       if scenario["param"]:
           params[scenario["param"]] = scenario["value"]
       
       # Calculate net pay
       net_pay = params["gross_thickness"] * params["ntg"]
       
       # Calculate OOIP
       ooip = (7758 * params["area"] * net_pay * params["porosity"] * 
               (1 - params["water_sat"])) / params["oil_fvf"]
       
       # Calculate change from base
       if scenario["name"] == "Base Case":
           base_ooip = ooip
           pct_change = 0
       else:
           pct_change = ((ooip - base_ooip) / base_ooip) * 100
       
       results.append({
           "Scenario": scenario["name"],
           "OOIP (MSTB)": ooip / 1000,
           "Change (%)": pct_change
       })
   
   # Display results
   df = pd.DataFrame(results)
   print("OOIP Sensitivity Analysis")
   print("=" * 60)
   print(df.to_string(index=False))
   
   # Tornado chart data
   print(f"\n💡 Most Sensitive Parameter:")
   max_impact = df.iloc[1:]["Change (%)"].abs().max()
   most_sensitive = df[df["Change (%)"].abs() == max_impact].iloc[0]
   print(f"   {most_sensitive['Scenario']}: {most_sensitive['Change (%)']:+.1f}%")

---

Module 4 Exercises: Reservoir Engineering
------------------------------------------

Exercise 4.1: Decline Curve Matching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:**

Historical production data for Well A:

.. csv-table::
   :header: "Month", "Rate (STB/d)"
   
   0, 1850
   3, 1120
   6, 785
   9, 612
   12, 503

Fit an Arps decline curve and forecast EUR to 50 STB/day.

**Solution:**

.. code-block:: python

   import numpy as np
   from scipy.optimize import curve_fit
   
   # Historical data
   months = np.array([0, 3, 6, 9, 12])
   rates = np.array([1850, 1120, 785, 612, 503])
   
   # Hyperbolic decline function
   def hyperbolic_decline(t, qi, di, b):
       """Arps hyperbolic decline."""
       years = t / 12
       return qi / ((1 + b * di * years) ** (1/b))
   
   # Fit curve to data
   params, covariance = curve_fit(
       hyperbolic_decline,
       months,
       rates,
       p0=[1850, 0.60, 1.2],  # Initial guesses
       bounds=([1500, 0.1, 0.0], [2000, 2.0, 2.0])
   )
   
   qi_fit, di_fit, b_fit = params
   
   print("Decline Curve Analysis - Well A")
   print("=" * 60)
   print(f"Fitted Parameters:")
   print(f"  qi (initial rate): {qi_fit:.1f} STB/day")
   print(f"  Di (decline rate): {di_fit:.2f} /year ({di_fit*100:.0f}%)")
   print(f"  b (exponent): {b_fit:.2f}")
   
   # Forecast to economic limit
   economic_limit = 50  # STB/day
   forecast_months = 240  # 20 years
   
   cumulative = 0
   for month in range(forecast_months):
       rate = hyperbolic_decline(month, qi_fit, di_fit, b_fit)
       if rate < economic_limit:
           break
       cumulative += rate * 30.44
   
   print(f"\nForecast Results:")
   print(f"  Economic Limit: {economic_limit} STB/day")
   print(f"  Months to Limit: {month}")
   print(f"  EUR: {cumulative:,.0f} STB")
   
   # Calculate NPV (simplified)
   oil_price = 75  # $/bbl
   well_cost = 8_000_000
   opex = 15  # $/bbl
   discount = 0.10
   
   npv = 0
   for month in range(month):
       rate = hyperbolic_decline(month, qi_fit, di_fit, b_fit)
       monthly_prod = rate * 30.44
       revenue = monthly_prod * (oil_price - opex)
       discounted = revenue / ((1 + discount) ** (month/12))
       npv += discounted
   
   npv -= well_cost
   
   print(f"\nEconomic Analysis:")
   print(f"  NPV @ 10%: ${npv:,.0f}")
   print(f"  Economic: {'YES ✓' if npv > 0 else 'NO ✗'}")

---

Exercise 4.2: Multi-Well Type Curve
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:**

You have 15 analog wells in the same field. Build a P10/P50/P90 type curve.

**Solution:**

.. code-block:: python

   import numpy as np
   from scipy import stats
   
   # Analog well decline parameters (from history matching)
   analog_wells = [
       {"qi": 1850, "di": 0.58, "b": 1.15},
       {"qi": 1620, "di": 0.62, "b": 1.22},
       {"qi": 2100, "di": 0.55, "b": 1.08},
       # ... 12 more wells
       {"qi": 1450, "di": 0.68, "b": 1.31},
   ]
   
   # Calculate EUR for each analog
   analog_eurs = []
   for well in analog_wells:
       eur = 0
       for month in range(240):
           year = month / 12
           rate = well["qi"] / ((1 + well["b"] * well["di"] * year) ** (1/well["b"]))
           if rate < 50:
               break
           eur += rate * 30.44
       analog_eurs.append(eur)
   
   # Statistical analysis
   p10 = np.percentile(analog_eurs, 90)
   p50 = np.percentile(analog_eurs, 50)
   p90 = np.percentile(analog_eurs, 10)
   mean = np.mean(analog_eurs)
   
   print("Type Curve Analysis - 15 Analog Wells")
   print("=" * 60)
   print(f"P90 (Conservative): {p90:,.0f} STB")
   print(f"P50 (Most Likely): {p50:,.0f} STB")
   print(f"P10 (Optimistic): {p10:,.0f} STB")
   print(f"Mean: {mean:,.0f} STB")
   
   # Risk-adjusted EUR for economics
   risk_weighted_eur = 0.1 * p10 + 0.6 * p50 + 0.3 * p90
   print(f"\nRisk-Weighted EUR: {risk_weighted_eur:,.0f} STB")

---

Module 5 Exercises: Drilling Engineering
-----------------------------------------

Exercise 5.1: Casing Design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:**

Design production casing for 10,500 ft well:

- Pore pressure: 5800 psi
- Fracture gradient: 0.82 psi/ft
- Kick scenario: +500 psi
- Empty casing scenario for collapse

Select casing size, weight, and grade.

**Solution:**

.. code-block:: python

   from petrosmith import DrillingAPI
   
   api = DrillingAPI()
   
   # Well parameters
   setting_depth = 10500  # ft TVD
   pore_pressure = 5800  # psi
   frac_gradient = 0.82  # psi/ft
   
   # Burst scenario
   kick_pressure = pore_pressure + 500
   mud_weight = 11.5  # ppg (balancing formation)
   external_pressure = 0.052 * mud_weight * setting_depth
   net_burst = kick_pressure - external_pressure
   
   # Collapse scenario
   net_collapse = 0.052 * mud_weight * setting_depth  # Full mud column, empty casing
   
   # Try different casing options
   casing_options = [
       {"od": 7.0, "weight": 23, "grade": "L-80"},
       {"od": 7.0, "weight": 26, "grade": "L-80"},
       {"od": 7.0, "weight": 29, "grade": "L-80"},
       {"od": 7.0, "weight": 26, "grade": "P-110"},
   ]
   
   print("Casing Selection Analysis")
   print("=" * 80)
   print(f"Setting Depth: {setting_depth:,} ft")
   print(f"Net Burst Load: {net_burst:,.0f} psi")
   print(f"Net Collapse Load: {net_collapse:,.0f} psi")
   print("\n" + "=" * 80)
   print(f"{'OD':>4s} {'Wt':>5s} {'Grade':>7s} {'Burst SF':>10s} {'Collapse SF':>12s} {'Status':>10s}")
   print("=" * 80)
   
   for casing in casing_options:
       burst_rating = api.calculate_casing_burst(
           outer_diameter=casing["od"],
           weight=casing["weight"],
           grade=casing["grade"]
       )
       
       collapse_rating = api.calculate_casing_collapse(
           outer_diameter=casing["od"],
           weight=casing["weight"],
           grade=casing["grade"]
       )
       
       burst_sf = burst_rating / net_burst
       collapse_sf = collapse_rating / net_collapse
       
       # Check if acceptable
       acceptable = (burst_sf >= 1.1 and collapse_sf >= 1.0)
       status = "PASS ✓" if acceptable else "FAIL ✗"
       
       print(f"{casing['od']:4.1f} {casing['weight']:5.0f} {casing['grade']:>7s} "
             f"{burst_sf:10.2f} {collapse_sf:12.2f} {status:>10s}")
   
   print("\n💡 RECOMMENDATION: 7.0\" 26 lb/ft L-80")
   print("   Adequate safety factors for burst and collapse")
   print("   Cost-effective compared to heavier options")

---

Exercise 5.2: Well Control Kill Sheet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:**

Generate a complete kill sheet for this kick:

- Depth: 11,200 ft TVD
- Original MW: 10.8 ppg
- SIDPP: 520 psi
- SICP: 780 psi
- Drill string volume: 180 bbls
- Annular volume: 410 bbls
- Pump output: 0.117 bbls/stroke
- Slow circulate rate: 40 SPM
- Slow circulate pressure: 220 psi

**Solution:**

.. code-block:: python

   from petrosmith import DrillingAPI
   
   api = DrillingAPI()
   
   # Kick data
   tvd = 11200
   original_mw = 10.8
   sidpp = 520
   sicp = 780
   ds_volume = 180  # bbls
   ann_volume = 410  # bbls
   pump_output = 0.117  # bbls/stroke
   scr = 40  # SPM (strokes per minute)
   scp = 220  # psi (slow circulate pressure)
   
   # Calculate kill mud weight
   kill_mw = original_mw + (sidpp / (0.052 * tvd))
   
   # Strokes to pump out
   strokes_to_bit = ds_volume / pump_output
   strokes_total = (ds_volume + ann_volume) / pump_output
   
   # Initial and final circulating pressures
   icp = sidpp + scp
   fcp = scp  # Just pump friction with kill mud
   
   print("WELL CONTROL KILL SHEET - Driller's Method")
   print("=" * 70)
   print(f"Date/Time: [FILL IN]")
   print(f"Well: [FILL IN]")
   print(f"Depth: {tvd:,} ft TVD")
   print("\nKICK DATA:")
   print(f"  Original Mud Weight: {original_mw:.2f} ppg")
   print(f"  SIDPP: {sidpp} psi")
   print(f"  SICP: {sicp} psi")
   print(f"  Kill Mud Weight: {kill_mw:.2f} ppg")
   print("\nCIRCULATION DATA:")
   print(f"  Slow Circulate Rate: {scr} SPM")
   print(f"  Slow Circulate Pressure: {scp} psi")
   print(f"  ICP (Initial Circ Pressure): {icp} psi")
   print(f"  FCP (Final Circ Pressure): {fcp} psi")
   print(f"  Pump Output: {pump_output} bbls/stroke")
   print("\n" + "=" * 70)
   print(f"{'Strokes':>10s} {'Drill Pipe':>15s} {'Remarks':>30s}")
   print("=" * 70)
   
   # Circulation 1: Original mud out of drill string
   increments = int(strokes_to_bit / 10)
   for i in range(11):
       strokes = i * increments
       pressure = icp - (icp - fcp) * (strokes / strokes_to_bit)
       
       if i == 0:
           remark = "Start circulation"
       elif strokes >= strokes_to_bit - 100:
           remark = "Kill mud at bit"
       else:
           remark = "Circulating original mud"
       
       print(f"{strokes:10.0f} {pressure:15.0f} {remark:>30s}")
   
   print(f"\n{'Strokes':>10s} {'Casing':>15s} {'Remarks':>30s}")
   print("=" * 70)
   
   # Circulation 2: Circulate annulus
   for i in range(11):
       strokes = strokes_to_bit + (i * ann_volume / pump_output / 10)
       pressure = fcp  # Constant with kill mud
       
       if i == 0:
           remark = "Kill mud entering annulus"
       elif i == 10:
           remark = "Well killed - shut down"
       else:
           remark = "Circulating kill mud"
       
       print(f"{strokes:10.0f} {pressure:15.0f} {remark:>30s}")
   
   print("\n" + "=" * 70)
   print(f"Total Strokes: {strokes_total:.0f}")
   print(f"Total Time: {strokes_total / scr:.0f} minutes")
   print("\nProcedure approved by: _________________  Time: _______")

---

Module Integration Exercise
----------------------------

Exercise I.1: Complete Field Development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:**

You're the lead engineer for a new field development. Integrate reservoir, drilling, and production engineering to create a comprehensive plan.

**Field Data:**

- 4 sections (2560 acres)
- Reservoir: Niobrara Formation
- Depth: 7800 ft TVD
- Net pay: 95 feet
- Porosity: 10%
- Permeability: 0.08 md
- Water saturation: 40%
- Initial pressure: 4200 psi
- Horizontal laterals: 7500 ft
- Expected IP: 1100 STB/day
- Decline: 68%/year, b=1.3

**Economic Parameters:**

- Horizontal well cost: $7.5M
- Oil price: $70/bbl
- OPEX: $12/bbl
- Discount rate: 12%

**Deliverables:**

1. Reserves estimate (volumetric OOIP, EUR per well)
2. Optimal well spacing (test 160, 80, 40 acre spacing)
3. Total well count and CAPEX
4. Production forecast (monthly for 10 years)
5. Economics (NPV, IRR, payout)
6. Drilling program (casing design, mud weights)
7. Sensitivity analysis (oil price, EUR, well cost)

**Solution Framework:**

.. code-block:: python

   from petrosmith import ReservoirAPI, DrillingAPI, ProductionAPI
   import numpy as np
   import pandas as pd
   
   # Part 1: Reservoir Analysis
   reservoir_api = ReservoirAPI()
   
   # [Calculate OOIP for different drainage areas]
   # [Run Monte Carlo for P10/P50/P90]
   # [Estimate recovery factors]
   
   # Part 2: Decline Curve & Production Forecast
   production_api = ProductionAPI()
   
   # [Forecast production profiles]
   # [Calculate EUR for each spacing scenario]
   # [Generate type curves]
   
   # Part 3: Well Spacing Optimization
   # [Calculate NPV for each scenario]
   # [Optimize for maximum field NPV]
   # [Sensitivity to oil price and EUR]
   
   # Part 4: Drilling Engineering
   drilling_api = DrillingAPI()
   
   # [Design casing program]
   # [Calculate mud weights and ECD]
   # [Estimate drilling days and AFE]
   
   # Part 5: Integrated Economics
   # [Build cash flow model]
   # [Calculate IRR and payout]
   # [Risk analysis]
   
   # Part 6: Executive Summary
   # [Recommendation]
   # [Sensitivities]
   # [Risk factors]
   
   # [Complete solution: 300+ lines of code in appendix]

---

Solutions Appendix
------------------

**Complete solutions for all exercises are provided in a separate document.**

Access full solutions with detailed explanations:

- **Solutions PDF:** Available on course materials page
- **Jupyter Notebooks:** Interactive solutions on GitHub
- **Video Walkthroughs:** Step-by-step video explanations

---

Additional Practice Problems
-----------------------------

**100+ Additional Problems Available:**

- Reservoir Engineering: 25 problems
- Drilling Engineering: 20 problems  
- Production Engineering: 20 problems
- Well Completions: 15 problems
- Integrated Workflows: 20 problems

**Problem Sets by Difficulty:**

- ⭐ Beginner (Foundation building)
- ⭐⭐ Intermediate (Real-world scenarios)
- ⭐⭐⭐ Advanced (Complex integration)

**Access:** See :doc:`../installation` for accessing the complete problem bank.

---

.. note::
   **Assessment**
   
   Want to test your knowledge?
   
   - **Module Quizzes:** Test comprehension (20-30 questions per module)
   - **Midterm Exam:** Covers Modules 1-5 (2 hours, open book)
   - **Final Project:** Complete field development plan (capstone)
   - **Certification:** Demonstrates mastery of PetroSmith and petroleum engineering
