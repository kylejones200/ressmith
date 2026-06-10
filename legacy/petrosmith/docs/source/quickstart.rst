Quick Start Guide
=================

Get productive with PetroSmith in 15 minutes.

Your First Calculation
----------------------

Let's calculate reserves for a typical oil reservoir:

.. code-block:: python

   from petrosmith.api import ReservoirAPI
   
   # Initialize the API
   api = ReservoirAPI()
   
   # Create reservoir (Alpha Sand example)
   reservoir = api.create_reservoir(
       reservoir_id="ALPHA-001",
       reservoir_name="Alpha Sand",
       formation="Cretaceous Sandstone",
       top_depth=8500,        # feet
       bottom_depth=8600,     # feet
       net_pay=80,            # feet
       fluid_type="oil",
       porosity=0.22,         # 22%
       permeability=150,      # millidarcies
       water_saturation=0.25, # 25%
       initial_pressure=3500, # psi
       temperature=180        # °F
   )
   
   # Calculate reserves
   reserves = api.calculate_reserves(
       reservoir_id="ALPHA-001",
       area=640,              # acres (1 section)
       net_pay=80,
       water_saturation=0.25,
       formation_volume_factor=1.2
   )
   
   # Display results
   print(f"Original Oil in Place: {reserves['original_oil_in_place']:,.0f} STB")
   print(f"Recovery Factor: {reserves['recovery_factor']:.1%}")
   print(f"Recoverable Reserves: {reserves['recoverable_reserves']:,.0f} STB")

**Output:**

.. code-block:: text

   Original Oil in Place: 54,616,320 STB
   Recovery Factor: 50.0%
   Recoverable Reserves: 27,308,160 STB

Understanding the Results
~~~~~~~~~~~~~~~~~~~~~~~~

- **OOIP** (Original Oil in Place): Total oil in the reservoir, calculated volumetrically
- **Recovery Factor**: Percentage of OOIP that can be economically recovered
- **Recoverable Reserves**: Oil that will actually be produced

Common Workflows
---------------

Workflow 1: Drilling Hydraulics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calculate equivalent circulating density (ECD) to avoid formation fracture:

.. code-block:: python

   from petrosmith.api import DrillingAPI
   
   api = DrillingAPI()
   
   # Drilling at 8,000 ft with 10.5 ppg mud
   ecd = api.calculate_ecd(
       mud_weight=10.5,              # ppg
       annular_pressure_loss=300,    # psi
       tvd=8000                      # feet
   )
   
   print(f"ECD: {ecd:.2f} ppg")
   
   # Check against fracture gradient
   fracture_gradient = 12.5  # ppg equivalent
   if ecd > fracture_gradient:
       print("⚠️ WARNING: ECD exceeds fracture gradient!")
       print("Action: Reduce pump rate or mud viscosity")
   else:
       print("✓ Safe to continue drilling")

Workflow 2: Production Forecasting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Forecast production for next 5 years:

.. code-block:: python

   from petrosmith.api import ProductionAPI
   
   api = ProductionAPI()
   
   # Add historical production data
   historical_data = [
       ("2025-01-01", 500, 250, 50),   # date, oil, gas, water
       ("2025-02-01", 480, 240, 55),
       ("2025-03-01", 465, 232, 58),
       # ... more data points
   ]
   
   for date, oil, gas, water in historical_data:
       api.add_production_data("WELL-001", date, oil, gas, water)
   
   # Generate forecast
   forecast = api.forecast_production("WELL-001", forecast_years=5)
   
   print(f"Current rate: {forecast['current_rate']:.0f} STB/day")
   print(f"Decline rate: {forecast['decline_rate_annual']:.1%} per year")
   print("\nForecast:")
   for year in forecast['forecast']:
       print(f"  Year {year['year']}: {year['rate']:.0f} STB/day")

Workflow 3: Well Control
~~~~~~~~~~~~~~~~~~~~~~~~

Analyze a kick and calculate kill mud weight:

.. code-block:: python

   from petrosmith.api import DrillingAPI
   
   api = DrillingAPI()
   
   # Kick detected: 15 bbl pit gain
   kick_analysis = api.analyze_kick(
       well_id="WELL-001",
       pit_gain=15.0,    # barrels
       drcp=500,         # drill pipe pressure
       dcpp=600          # casing pressure
   )
   
   print(f"Kick severity: {kick_analysis['kick_analysis']['severity']}")
   print(f"Formation pressure: {kick_analysis['kick_analysis']['formation_pressure']:.0f} psi")
   print(f"\nKill Parameters:")
   print(f"  Current mud weight: {kick_analysis['kill_parameters']['current_mud_weight']:.2f} ppg")
   print(f"  Kill mud weight: {kick_analysis['kill_parameters']['kill_mud_weight']:.2f} ppg")
   print(f"  ICP: {kick_analysis['kill_parameters']['initial_circulating_pressure']:.0f} psi")
   print(f"  FCP: {kick_analysis['kill_parameters']['final_circulating_pressure']:.0f} psi")

Best Practices
-------------

1. **Always Validate Inputs**

   The library validates inputs, but double-check critical values:

   .. code-block:: python

      # Good practice: Verify reasonable ranges
      porosity = 0.22
      assert 0.05 <= porosity <= 0.40, "Porosity outside normal range"

2. **Use Descriptive IDs**

   .. code-block:: python

      # Bad
      reservoir_id="R1"
      
      # Good
      reservoir_id="ALPHA-SAND-SECTION-15"

3. **Store Results for Audit Trail**

   .. code-block:: python

      import json
      from datetime import datetime
      
      results = api.calculate_reserves(...)
      results['calculation_date'] = datetime.now().isoformat()
      results['engineer'] = "John Smith"
      
      with open('reserves_2026.json', 'w') as f:
          json.dump(results, f, indent=2)

4. **Batch Processing for Multiple Wells**

   .. code-block:: python

      # Efficient batch processing
      wells = load_well_data()  # From database or CSV
      
      results = []
      for well in wells:
          result = api.analyze_well_performance(well.id)
          results.append(result)
      
      # Export to Excel or database
      save_results(results)

Common Patterns
--------------

Pattern: Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from petrosmith.api import ReservoirAPI
   
   api = ReservoirAPI()
   
   # Base case
   base_reserves = api.calculate_ooip(
       area=640,
       net_pay=80,
       porosity=0.22,
       oil_saturation=0.75,
       formation_volume_factor=1.2
   )
   
   # Sensitivity: +/- 10% on porosity
   sensitivities = []
   for porosity_mult in [0.9, 1.0, 1.1]:
       reserves = api.calculate_ooip(
           area=640,
           net_pay=80,
           porosity=0.22 * porosity_mult,
           oil_saturation=0.75,
           formation_volume_factor=1.2
       )
       sensitivities.append({
           'case': f'Porosity {porosity_mult*100:.0f}%',
           'reserves': reserves
       })
   
   # Analyze swing
   low = sensitivities[0]['reserves']
   high = sensitivities[2]['reserves']
   print(f"Porosity sensitivity: {(high-low)/base_reserves:.1%} swing")

Pattern: Economic Screening
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from petrosmith.api import ProductionAPI
   
   api = ProductionAPI()
   
   # Screen 100 wells for ESP candidates
   oil_price = 75  # $/bbl
   esp_cost = 250_000  # $
   
   candidates = []
   for well in all_wells:
       optimization = api.optimize_artificial_lift(
           well_id=well.id,
           reservoir_pressure=well.pressure,
           depth=well.depth,
           desired_rate=well.target_rate,
           fluid_density=well.fluid_density
       )
       
       # Calculate economics
       incremental_production = optimization['recommended_system'] == 'esp'
       if incremental_production:
           daily_revenue = well.incremental_rate * oil_price
           payback_months = esp_cost / (daily_revenue * 30)
           
           if payback_months < 12:  # Less than 1 year payback
               candidates.append({
                   'well': well.id,
                   'payback': payback_months,
                   'npv': calculate_npv(well, optimization)
               })
   
   # Rank by economics
   candidates.sort(key=lambda x: x['payback'])
   print(f"Found {len(candidates)} economic ESP candidates")

Next Steps
----------

Now that you've mastered the basics:

1. **Explore User Guides** - Detailed workflows for your discipline
   
   - :doc:`guides/reservoir_engineering` - Reserves, forecasting, material balance
   - :doc:`guides/drilling_operations` - Hydraulics, casing, well control
   - :doc:`guides/production_optimization` - Artificial lift, nodal analysis

2. **Check Real Examples** - :doc:`examples` - Complete workflows

3. **Read API Docs** - :doc:`api/models` - All available functions

4. **Join Community** - Share experiences with other petroleum engineers

Tips & Tricks
------------

**Interactive Development**

Use Jupyter notebooks for interactive analysis:

.. code-block:: python

   # Cell 1: Setup
   from petrosmith.api import *
   import matplotlib.pyplot as plt
   
   # Cell 2: Calculate
   api = ReservoirAPI()
   # ... your calculations
   
   # Cell 3: Visualize
   plt.plot(forecast['pressures'], forecast['rates'])
   plt.xlabel('Pressure (psi)')
   plt.ylabel('Flow Rate (STB/day)')
   plt.title('IPR Curve')
   plt.show()

**Integration with Existing Tools**

.. code-block:: python

   import pandas as pd
   
   # Read from Excel
   wells_df = pd.read_excel('wells.xlsx')
   
   # Process with PetroSmith
   from petrosmith.api import ProductionAPI
   api = ProductionAPI()
   
   results = []
   for _, well in wells_df.iterrows():
       result = api.analyze_well_performance(well['well_id'])
       results.append(result)
   
   # Write back to Excel
   results_df = pd.DataFrame(results)
   results_df.to_excel('analysis_results.xlsx', index=False)

Questions?
---------

- Check the :doc:`glossary` for petroleum engineering terms
- See :doc:`examples` for more complete workflows
- Review :doc:`api/calculations` for all available methods
