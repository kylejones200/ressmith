Reservoir Engineering Guide
===========================

This guide helps reservoir engineers leverage PetroSmith for daily workflows - from reserves estimation to production forecasting. Written for engineers who need answers, not just code.

The Reservoir Engineer's Challenge
----------------------------------

As a reservoir engineer, you're constantly asked:

- "What are our proven reserves for the SEC filing?"
- "Can we justify drilling another development well?"
- "What's the EUR (estimated ultimate recovery) for this well?"
- "Should we implement waterflood or CO2 injection?"

These questions require solid calculations, not guesses. But building reliable spreadsheets takes time, and errors can be costly. PetroSmith gives you engineering-grade tools to answer these questions quickly and confidently.

Core Workflows
-------------

1. Volumetric Reserves Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**The Engineering Problem:**

You need to estimate original oil in place (OOIP) for a newly discovered reservoir. You have:

- 3D seismic interpretation (area, structure)
- Core analysis (porosity, permeability)
- Log analysis (water saturation)
- PVT data (formation volume factor)

**Traditional Approach:**

Build Excel spreadsheet, manually input data, hope formulas are right, spend hours checking.

**PetroSmith Approach:**

.. code-block:: python

   from petrosmith.api import ReservoirAPI
   
   api = ReservoirAPI()
   
   # Input data from geological model
   reservoir_data = {
       'reservoir_id': 'ALPHA-MAIN',
       'reservoir_name': 'Alpha Main Sand',
       'formation': 'Cretaceous',
       'top_depth': 8500,
       'bottom_depth': 8650,
       'net_pay': 120,  # From log analysis
       'fluid_type': 'oil',
       'porosity': 0.24,  # From core
       'permeability': 180,  # md
       'water_saturation': 0.22,  # From logs
       'initial_pressure': 3800,  # From RFT
       'temperature': 195  # °F
   }
   
   reservoir = api.create_reservoir(**reservoir_data)
   
   # Calculate OOIP for development area
   reserves = api.calculate_reserves(
       reservoir_id='ALPHA-MAIN',
       area=1280,  # acres (2 sections)
       net_pay=120,
       water_saturation=0.22,
       formation_volume_factor=1.25  # From PVT
   )
   
   print(f"📊 Reserves Estimate for {reservoir_data['reservoir_name']}")
   print(f"="*60)
   print(f"Gross Area: 1,280 acres")
   print(f"Net Pay: {reservoir_data['net_pay']} ft")
   print(f"Porosity: {reservoir_data['porosity']:.1%}")
   print(f"Sw: {reservoir_data['water_saturation']:.1%}")
   print(f"\nRESULTS:")
   print(f"OOIP: {reserves['original_oil_in_place']:,.0f} STB")
   print(f"Recovery Factor: {reserves['recovery_factor']:.1%}")
   print(f"Recoverable: {reserves['recoverable_reserves']:,.0f} STB")
   print(f"Drive Mechanism: {reserves['drive_mechanism']}")

**Why This Matters:**

- **Speed:** 5 minutes vs. 2 hours in Excel
- **Accuracy:** No formula errors, validated industry correlations
- **Reproducibility:** Same inputs always give same outputs
- **Audit Trail:** Code is the documentation

**Engineering Insight:**

The recovery factor is estimated based on drive mechanism. For better accuracy:

.. code-block:: python

   # If you have production history, calculate actual recovery
   actual_recovery = cumulative_production / ooip
   
   # Update reservoir with observed data
   if actual_recovery > reserves['recovery_factor']:
       print("⚠️ Reservoir performing better than expected")
       print("Consider: Better sweep, stronger aquifer support")

2. Well Deliverability Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**The Engineering Problem:**

New well just completed. What's the productivity index? Will it meet production targets?

**Engineering Context:**

- Productivity Index (PI) tells you how much oil flows per psi of drawdown
- Low PI might indicate formation damage (skin)
- Need to decide: Do we stimulate? Do we acidize? Or is this expected?

**PetroSmith Solution:**

.. code-block:: python

   from petrosmith.api import ReservoirAPI
   
   api = ReservoirAPI()
   
   # Well and reservoir data
   well_id = 'ALPHA-3H'
   reservoir_id = 'ALPHA-MAIN'
   
   # Deliverability analysis
   analysis = api.analyze_well_deliverability(
       reservoir_id=reservoir_id,
       well_id=well_id,
       drainage_radius=1000,  # ft (calculated from spacing)
       wellbore_radius=0.328,  # ft (8.5" hole)
       fluid_viscosity=2.1,  # cp from PVT
       fluid_fvf=1.25,  # rb/STB
       skin_factor=5.0  # From pressure buildup test
   )
   
   print(f"Well Deliverability Analysis: {well_id}")
   print(f"="*60)
   print(f"Current Pressure: {analysis['current_pressure']:.0f} psi")
   print(f"Productivity Index: {analysis['productivity_index']:.2f} STB/day/psi")
   print(f"Skin Factor: {analysis['skin_factor']:.1f}")
   print(f"Max Theoretical Rate: {analysis['max_theoretical_rate']:.0f} STB/day")
   print(f"Drainage Area: {analysis['drainage_area_acres']:.0f} acres")
   
   # Engineering decision
   if analysis['skin_factor'] > 3:
       print("\n💡 RECOMMENDATION:")
       print("High skin factor indicates formation damage")
       print("Action: Consider acid stimulation")
       print(f"Potential rate increase: {analysis['max_theoretical_rate']*0.3:.0f} STB/day")

**Interpreting Results:**

- **PI > 1.0**: Good well, decent productivity
- **PI < 0.5**: Poor well, need stimulation or completion changes
- **Skin > 5**: Likely formation damage, candidate for acid
- **Skin < 0**: Fractured or naturally enhanced permeability

**IPR Curve for Well Planning:**

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Plot IPR curve
   ipr_data = analysis['ipr_curve']
   
   pressures = [point['bottomhole_pressure'] for point in ipr_data]
   rates = [point['flow_rate'] for point in ipr_data]
   
   plt.figure(figsize=(10, 6))
   plt.plot(rates, pressures, 'b-', linewidth=2, label='IPR')
   plt.axhline(y=analysis['current_pressure'], color='r', 
               linestyle='--', label='Current Pressure')
   plt.xlabel('Flow Rate (STB/day)', fontsize=12)
   plt.ylabel('Bottomhole Pressure (psi)', fontsize=12)
   plt.title(f'Inflow Performance Relationship - {well_id}', fontsize=14)
   plt.grid(True, alpha=0.3)
   plt.legend()
   plt.show()

3. Material Balance & Pressure Maintenance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**The Engineering Problem:**

Reservoir has been producing for 2 years. Current pressure is declining faster than expected. Management asks: "Should we start water injection?"

**Engineering Analysis:**

.. code-block:: python

   from petrosmith.api import ReservoirAPI
   
   api = ReservoirAPI()
   
   # Production history
   cumulative_production = 2_500_000  # STB after 2 years
   original_in_place = 65_000_000  # STB from volumetrics
   
   # Material balance analysis
   mb_analysis = api.perform_material_balance(
       reservoir_id='ALPHA-MAIN',
       cumulative_production=cumulative_production,
       original_in_place=original_in_place
   )
   
   print("Material Balance Analysis")
   print("="*60)
   print(f"Initial Pressure: {mb_analysis['initial_pressure']:.0f} psi")
   print(f"Current Pressure: {mb_analysis['current_pressure']:.0f} psi")
   print(f"Pressure Decline: {mb_analysis['pressure_decline']:.0f} psi")
   print(f"\nRecovery to Date: {mb_analysis['recovery_percent']:.2f}%")
   print(f"Remaining Reserves: {mb_analysis['remaining_reserves']:,.0f} STB")
   
   # Engineering decision logic
   pressure_decline_rate = mb_analysis['pressure_decline'] / 2  # per year
   
   if pressure_decline_rate > 200:
       print("\n⚠️ PRESSURE MAINTENANCE REQUIRED")
       print("\nOptions:")
       print("1. Water injection - Most common, good sweep efficiency")
       print("2. Gas injection - If available, maintains pressure")
       print("3. Infill drilling - Only if pressure adequate")
       print("\nRecommendation: Initiate waterflood pilot")
       
       # Waterflood economics
       water_injection_rate = 5000  # bbl/day
       voidage_replacement = water_injection_rate / (cumulative_production/730)
       print(f"\nProposed injection: {water_injection_rate:,.0f} bbl/day")
       print(f"Voidage replacement: {voidage_replacement:.1%}")

**Business Impact:**

Without pressure maintenance:
   - Production declines to uneconomic rate in 3-4 years
   - Recover only 15-20% of OOIP

With waterflood:
   - Extend field life by 10-15 years
   - Increase recovery to 35-45% of OOIP
   - Incremental value: $50-100 million

4. Production Forecasting & EUR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**The Engineering Problem:**

Board meeting next week. Need EUR (Estimated Ultimate Recovery) for budget planning and asset valuation.

**Engineering Workflow:**

.. code-block:: python

   from petrosmith.api import ProductionAPI, ReservoirAPI
   
   prod_api = ProductionAPI()
   res_api = ReservoirAPI()
   
   # Load production history (simplified)
   well_id = 'ALPHA-2'
   
   # Add 2 years of monthly production data
   historical_production = [
       # (date, oil_rate, gas_rate, water_rate)
       ('2024-01-01', 450, 225, 25),
       ('2024-02-01', 425, 218, 28),
       ('2024-03-01', 408, 210, 32),
       # ... more data points
       ('2025-12-01', 275, 142, 55),
   ]
   
   for date, oil, gas, water in historical_production:
       prod_api.add_production_data(well_id, date, oil, gas, water)
   
   # Generate 10-year forecast
   forecast = prod_api.forecast_production(well_id, forecast_years=10)
   
   print(f"EUR Analysis - {well_id}")
   print("="*60)
   print(f"Current Rate: {forecast['current_rate']:.0f} STB/day")
   print(f"Decline Rate: {forecast['decline_rate_annual']:.1%} per year")
   print(f"Economic Limit: {forecast['economic_limit']:.0f} STB/day")
   print(f"\n10-Year Forecast:")
   
   total_eur = 0
   for year in forecast['forecast']:
       if year['year'] > 0:  # Skip year 0 (current)
           print(f"  Year {year['year']}: {year['rate']:>6.0f} STB/day  "
                 f"Cumulative: {year['cumulative']:>10,.0f} STB")
           total_eur = year['cumulative']
   
   print(f"\nEstimated Ultimate Recovery: {total_eur:,.0f} STB")
   
   # Economic analysis
   oil_price = 75  # $/bbl
   opex = 25  # $/bbl
   revenue = total_eur * (oil_price - opex)
   print(f"\nEconomic Value (@${oil_price}/bbl, ${opex}/bbl OPEX):")
   print(f"Gross Revenue: ${revenue:,.0f}")

**Engineering Notes:**

Decline curve analysis assumes:
   - Exponential decline (works for most solution gas drive reservoirs)
   - No major operational changes
   - No stimulation or workovers

For more accurate forecasts:
   - Use hyperbolic decline if reservoir is gas-drive
   - Update forecast quarterly with actual data
   - Consider type curves if field-wide behavior is consistent

5. Field Development Planning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**The Engineering Problem:**

Have 2,000 acres proven. How many wells? What spacing? What's the optimal development plan?

**Comprehensive Analysis:**

.. code-block:: python

   from petrosmith.api import ReservoirAPI
   
   api = ReservoirAPI()
   
   # Field parameters
   field_area = 2000  # acres
   ooip_per_acre = 85000  # STB/acre (from volumetrics)
   
   # Evaluate different spacing scenarios
   spacing_options = [40, 80, 160]  # acres per well
   
   print("Field Development Scenarios")
   print("="*80)
   
   for spacing in spacing_options:
       num_wells = int(field_area / spacing)
       
       # Get reservoir performance estimate
       performance = api.estimate_reservoir_performance(
           reservoir_id='ALPHA-MAIN',
           number_of_wells=num_wells,
           well_spacing_acres=spacing
       )
       
       # Economics
       well_cost = 3_500_000  # $3.5MM per well
       total_capex = num_wells * well_cost
       
       initial_rate = performance['production_estimates']['total_initial_rate']
       plateau_years = performance['production_estimates']['plateau_duration_years']
       
       # Simple NPV calc (should use full DCF model)
       avg_rate = initial_rate * 0.7  # Decline factor
       total_production = avg_rate * 365 * plateau_years
       revenue = total_production * 50  # $50/bbl net
       npv = revenue - total_capex
       
       print(f"\nScenario: {spacing} acre spacing ({num_wells} wells)")
       print(f"  CAPEX: ${total_capex:,.0f}")
       print(f"  Initial Rate: {initial_rate:,.0f} STB/day")
       print(f"  Plateau Duration: {plateau_years:.1f} years")
       print(f"  Est. Production: {total_production:,.0f} STB")
       print(f"  NPV (simplified): ${npv:,.0f}")
       print(f"  NPV per well: ${npv/num_wells:,.0f}")

**Engineering Decision:**

Results show:
- 40-acre spacing: High CAPEX, marginal NPV per well
- 80-acre spacing: Optimal balance, best NPV per well
- 160-acre spacing: Lower CAPEX but poor sweep, leaves oil behind

**Recommendation: 80-acre spacing** (25 wells)
- Maximizes NPV while maintaining good drainage
- Allows phased development (drill 10 wells initially, evaluate, then drill remaining)

Best Practices for Reservoir Engineers
--------------------------------------

1. **Always Validate Against Type Curves**

   .. code-block:: python

      # Compare your calculations to field analogs
      calculated_rf = reserves['recovery_factor']
      field_average_rf = 0.35  # From similar fields
      
      if abs(calculated_rf - field_average_rf) > 0.10:
          print("⚠️ Recovery factor differs significantly from analogs")
          print("Review: Drive mechanism, rock quality, fluid properties")

2. **Document Assumptions**

   .. code-block:: python

      assumptions = {
          'porosity_source': 'Core analysis, 23 samples',
          'permeability_source': 'Core and log correlation',
          'water_saturation_method': 'Archie equation',
          'fvf_source': 'Standing correlation, validated with lab PVT',
          'recovery_factor_basis': 'Analog field performance'
      }
      
      # Save with results
      results['assumptions'] = assumptions

3. **Run Sensitivity Cases**

   .. code-block:: python

      # Monte Carlo style sensitivity
      import numpy as np
      
      results = []
      for i in range(1000):
           # Vary key parameters within ranges
           porosity = np.random.normal(0.22, 0.02)  # ± 2 porosity units
           sw = np.random.normal(0.25, 0.05)  # ± 5% water sat
           
           ooip = api.calculate_ooip(
               area=640,
               net_pay=80,
               porosity=max(0.05, min(0.40, porosity)),
               oil_saturation=1-sw,
               formation_volume_factor=1.2
           )
           results.append(ooip)
      
      # Statistical analysis
      p10 = np.percentile(results, 90)  # Optimistic
      p50 = np.percentile(results, 50)  # Most likely
      p90 = np.percentile(results, 10)  # Conservative
      
      print(f"OOIP Probability Distribution:")
      print(f"  P90 (Conservative): {p90:,.0f} STB")
      print(f"  P50 (Most Likely):  {p50:,.0f} STB")
      print(f"  P10 (Optimistic):   {p10:,.0f} STB")

4. **Integrate with Corporate Database**

   .. code-block:: python

      import psycopg2  # or your database library
      
      # Get data from corporate database
      conn = psycopg2.connect("dbname=corporate user=engineer")
      cursor = conn.execute("SELECT * FROM wells WHERE field='Alpha'")
      
      for well_data in cursor.fetchall():
          # Run PetroSmith analysis
          analysis = api.analyze_well_deliverability(**well_data)
          
          # Store results back to database
          cursor.execute("""
              UPDATE wells 
              SET productivity_index = %s, 
                  last_analysis_date = NOW()
              WHERE well_id = %s
          """, (analysis['productivity_index'], well_data['well_id']))
      
      conn.commit()

Common Pitfalls to Avoid
-----------------------

❌ **Mixing Units**
   PetroSmith uses field units consistently. Don't mix metric and imperial.

❌ **Ignoring Reservoir Heterogeneity**
   Volumetric calculations assume uniform properties. Real reservoirs aren't uniform.

❌ **Over-Reliance on Type Curves**
   Your reservoir is unique. Validate correlations with actual data.

✅ **Do This Instead:**
   - Use consistent units throughout
   - Apply correction factors for heterogeneity
   - Calibrate models with production history

Next Steps
----------

Master these workflows:

- :doc:`../workflows/reserves_estimation` - Detailed reserves calculation workflow
- :doc:`../workflows/production_forecasting` - Complete forecasting methodology
- :doc:`../api/calculations` - All reservoir calculation functions

Need help? Check the :doc:`../examples` for complete code examples.
