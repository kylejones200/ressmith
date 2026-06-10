Module 6: Production Engineering
=================================

.. meta::
   :description: Comprehensive production engineering course covering artificial lift, nodal analysis, and well testing
   :keywords: production engineering, artificial lift, ESP, gas lift, nodal analysis, well testing

**Learning Objectives**

After completing this module, you will be able to:

- Build inflow performance relationships (IPR) using Vogel and Fetkovich methods
- Calculate vertical lift performance (VLP) for tubing flow
- Perform nodal analysis to optimize well performance
- Design electric submersible pump (ESP) systems
- Design and optimize gas lift installations
- Design sucker rod pumping systems
- Conduct and interpret well tests
- Optimize production operations for maximum value

**Time Commitment:** 8-10 hours

**Prerequisites:** Modules 1-4

----

Introduction: The Production Engineer's Role
--------------------------------------------

Production engineering bridges **reservoir** and **surface facilities**, ensuring:

- **Maximum economic recovery** from reservoirs
- **Optimized artificial lift** selection and design
- **Efficient flow** from reservoir to separator
- **Well integrity** and longevity
- **Safe operations** within equipment limits

**Key Questions Production Engineers Answer:**

1. What rate can this well produce?
2. Which artificial lift method is optimal?
3. Where is the bottleneck in the system?
4. How do we increase production economically?
5. When should we workover this well?

----

Part 1: Inflow Performance Relationships (IPR)
-----------------------------------------------

1.1 Introduction to IPR
^^^^^^^^^^^^^^^^^^^^^^^^

The **IPR curve** shows the relationship between **bottomhole flowing pressure (Pwf)** and **production rate (q)**.

**Physical Basis:**

Flow from reservoir into wellbore follows:

.. math::

   q = J \cdot (P_r - P_{wf})

Where:
- :math:`q` = flow rate (STB/day)
- :math:`J` = productivity index (STB/day/psi)
- :math:`P_r` = average reservoir pressure (psi)
- :math:`P_{wf}` = bottomhole flowing pressure (psi)

**This is valid for single-phase flow above bubble point.**

**Below bubble point:** Flow becomes non-linear due to:
- Free gas in reservoir
- Relative permeability effects
- Increased flow resistance

1.2 Vogel's Method (Two-Phase IPR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For **undersaturated reservoirs** (Pr < Pb), Vogel developed an empirical correlation:

.. math::

   \frac{q}{q_{max}} = 1 - 0.2 \left(\frac{P_{wf}}{P_r}\right) - 0.8 \left(\frac{P_{wf}}{P_r}\right)^2

**Key Points:**

- :math:`q_{max}` = maximum rate at Pwf = 0 (theoretical)
- Non-linear relationship
- Accounts for two-phase flow effects
- Most widely used for oil wells below bubble point

**PetroSmith Implementation:**

.. code-block:: python

   from petrosmith.core import ProductionCalculations
   import matplotlib.pyplot as plt
   
   # Create Vogel IPR
   reservoir_pressure = 2500  # psi
   max_rate = 800  # STB/day (from test data)
   
   # Generate IPR curve
   pressures = []
   rates = []
   
   for pwf in range(0, int(reservoir_pressure) + 100, 100):
       if pwf <= reservoir_pressure:
           q = ProductionCalculations.calculate_vogel_ipr(
               reservoir_pressure=reservoir_pressure,
               bottomhole_pressure=pwf,
               max_rate=max_rate
           )
           pressures.append(pwf)
           rates.append(q)
   
   # Plot IPR
   plt.figure(figsize=(10, 6))
   plt.plot(rates, pressures, 'b-', linewidth=2, label='Vogel IPR')
   plt.xlabel('Flow Rate (STB/day)')
   plt.ylabel('Bottomhole Pressure (psi)')
   plt.title('Inflow Performance Relationship')
   plt.grid(True, alpha=0.3)
   plt.legend()
   plt.show()
   
   print(f"Maximum theoretical rate: {max_rate} STB/day")
   print(f"Rate at 1000 psi: {rates[10]:.0f} STB/day")

**Output:**

.. code-block:: text

   Maximum theoretical rate: 800 STB/day
   Rate at 1000 psi: 584 STB/day

**Composite IPR (Above and Below Bubble Point):**

When Pr > Pb, use composite method:

.. code-block:: python

   def calculate_composite_ipr(pr, pb, j, pwf):
       """
       Calculate flow rate for composite IPR.
       
       Args:
           pr: Reservoir pressure (psi)
           pb: Bubble point pressure (psi)
           j: Productivity index (STB/day/psi)
           pwf: Bottomhole flowing pressure (psi)
       
       Returns:
           Flow rate (STB/day)
       """
       if pwf >= pb:
           # Single phase (linear)
           return j * (pr - pwf)
       else:
           # Two regions
           qb = j * (pr - pb)  # Rate at bubble point
           
           # Vogel for below bubble point
           qmax_vogel = qb / (1 - 0.2*(pb/pr) - 0.8*(pb/pr)**2)
           
           q_vogel = qmax_vogel * (
               1 - 0.2*(pwf/pr) - 0.8*(pwf/pr)**2
           )
           
           return q_vogel

**Example:**

.. code-block:: python

   # Composite IPR example
   pr = 3000  # psi
   pb = 2400  # psi (bubble point)
   j = 2.0    # STB/day/psi
   
   # Above bubble point (Pwf = 2600 psi)
   q1 = calculate_composite_ipr(pr, pb, j, pwf=2600)
   print(f"Rate at 2600 psi (single phase): {q1:.0f} STB/day")
   
   # Below bubble point (Pwf = 1500 psi)
   q2 = calculate_composite_ipr(pr, pb, j, pwf=1500)
   print(f"Rate at 1500 psi (two-phase): {q2:.0f} STB/day")

1.3 Fetkovich Method (Gas Wells)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For **gas wells**, Fetkovich developed:

.. math::

   q_g = C \cdot (P_r^2 - P_{wf}^2)^n

Where:
- :math:`q_g` = gas flow rate (Mscf/day)
- :math:`C` = performance coefficient
- :math:`n` = exponent (typically 0.5 to 1.0)
- Laminar flow: n = 1.0
- Turbulent flow: n = 0.5

**PetroSmith Implementation:**

.. code-block:: python

   def calculate_fetkovich_ipr(pr, pwf, c_coeff, n_exp):
       """
       Calculate gas well IPR using Fetkovich method.
       
       Args:
           pr: Reservoir pressure (psia)
           pwf: Bottomhole flowing pressure (psia)
           c_coeff: Performance coefficient
           n_exp: Flow exponent (0.5 to 1.0)
       
       Returns:
           Gas flow rate (Mscf/day)
       """
       qg = c_coeff * (pr**2 - pwf**2)**n_exp
       return qg
   
   # Example: Calculate gas well deliverability
   pr = 3500  # psia
   c = 0.5    # Performance coefficient
   n = 0.75   # Flow exponent
   
   # Generate deliverability curve
   for pwf in range(500, 3500, 500):
       qg = calculate_fetkovich_ipr(pr, pwf, c, n)
       print(f"Pwf = {pwf:4d} psia → qg = {qg:7.0f} Mscf/day")

----

Part 2: Vertical Lift Performance (VLP)
----------------------------------------

2.1 Introduction to VLP
^^^^^^^^^^^^^^^^^^^^^^^^

**VLP curves** show the pressure required at the **bottom of tubing** to lift fluids to surface at a given rate.

**Key Factors:**

- Tubing diameter
- Well depth
- Fluid properties (density, viscosity)
- Flow regime (bubble, slug, mist, annular)
- Gas-liquid ratio (GLR)

**Pressure Components:**

Total pressure drop = Elevation + Friction + Acceleration

.. math::

   \Delta P_{total} = \Delta P_{elevation} + \Delta P_{friction} + \Delta P_{acceleration}

For most wells, **elevation dominates** (>90% of pressure drop).

2.2 Tubing Performance
^^^^^^^^^^^^^^^^^^^^^^^

**Simplified VLP Calculation:**

.. code-block:: python

   def calculate_vlp_simplified(
       oil_rate_stb,
       water_cut_percent,
       gor_scf_stb,
       tubing_depth_ft,
       tubing_id_inches,
       wellhead_pressure_psi,
       oil_gravity_api,
       gas_gravity_air,
       water_sg
   ):
       """
       Calculate bottomhole flowing pressure required for given rate.
       
       Uses simplified gradient approach.
       """
       import numpy as np
       
       # Convert inputs
       wc = water_cut_percent / 100.0
       total_liquid = oil_rate_stb
       
       # Calculate densities
       oil_sg = 141.5 / (131.5 + oil_gravity_api)
       
       # Average liquid density
       liquid_sg = oil_sg * (1 - wc) + water_sg * wc
       liquid_density = liquid_sg * 62.4  # lb/ft³
       
       # Gas-liquid ratio
       glr = gor_scf_stb / (oil_rate_stb + 0.001)
       
       # Estimate two-phase gradient (Beggs & Brill correlation)
       # Simplified for vertical flow
       if glr < 100:
           gradient = liquid_density * 0.433 / 62.4  # psi/ft
       else:
           # Correct for gas effect
           gas_correction = 1.0 - (glr / 10000.0)
           gradient = liquid_density * 0.433 / 62.4 * gas_correction
       
       # Friction loss (empirical)
       velocity = (total_liquid * 5.615) / (86400 * np.pi * (tubing_id_inches/12/2)**2)
       friction_gradient = 0.001 * velocity**2  # psi/ft (simplified)
       
       # Total pressure drop
       pressure_drop = (gradient + friction_gradient) * tubing_depth_ft
       
       # Bottomhole pressure
       pwf = wellhead_pressure_psi + pressure_drop
       
       return {
           'bottomhole_pressure': pwf,
           'pressure_drop': pressure_drop,
           'gradient': gradient,
           'friction_loss': friction_gradient * tubing_depth_ft
       }
   
   # Example
   result = calculate_vlp_simplified(
       oil_rate_stb=500,
       water_cut_percent=20,
       gor_scf_stb=400,
       tubing_depth_ft=8000,
       tubing_id_inches=2.441,  # 2-7/8" tubing
       wellhead_pressure_psi=150,
       oil_gravity_api=35,
       gas_gravity_air=0.65,
       water_sg=1.05
   )
   
   print(f"Required Pwf: {result['bottomhole_pressure']:.0f} psi")
   print(f"Pressure drop: {result['pressure_drop']:.0f} psi")
   print(f"Gradient: {result['gradient']:.3f} psi/ft")

**Tubing Size Selection:**

Larger tubing:
- ✅ Lower friction losses
- ✅ Higher rates possible
- ❌ More expensive
- ❌ Higher annular velocities (for gas lift)

Smaller tubing:
- ✅ Lower cost
- ✅ Better for gas lift
- ❌ Higher friction
- ❌ Rate limitation

2.3 VLP Curve Generation
^^^^^^^^^^^^^^^^^^^^^^^^^

Generate full VLP curves for different rates:

.. code-block:: python

   import matplotlib.pyplot as plt
   
   def generate_vlp_curves(
       rates,
       tubing_depth,
       tubing_id,
       whp,
       oil_api,
       gas_sg,
       gor,
       water_cut
   ):
       """Generate VLP curves for multiple rates"""
       
       fig, ax = plt.subplots(figsize=(10, 8))
       
       for rate in rates:
           pressures = []
           depths = []
           
           for depth in range(0, tubing_depth + 500, 500):
               result = calculate_vlp_simplified(
                   oil_rate_stb=rate,
                   water_cut_percent=water_cut,
                   gor_scf_stb=gor,
                   tubing_depth_ft=depth,
                   tubing_id_inches=tubing_id,
                   wellhead_pressure_psi=whp,
                   oil_gravity_api=oil_api,
                   gas_gravity_air=gas_sg,
                   water_sg=1.05
               )
               pressures.append(result['bottomhole_pressure'])
               depths.append(depth)
           
           ax.plot(pressures, depths, label=f'{rate} STB/day', linewidth=2)
       
       ax.set_xlabel('Pressure (psi)', fontsize=12)
       ax.set_ylabel('Depth (ft)', fontsize=12)
       ax.set_title('Vertical Lift Performance Curves', fontsize=14)
       ax.invert_yaxis()
       ax.grid(True, alpha=0.3)
       ax.legend()
       plt.tight_layout()
       plt.show()
   
   # Generate curves
   generate_vlp_curves(
       rates=[200, 400, 600, 800, 1000],
       tubing_depth=8000,
       tubing_id=2.441,
       whp=150,
       oil_api=35,
       gas_sg=0.65,
       gor=400,
       water_cut=20
   )

----

Part 3: Nodal Analysis
-----------------------

3.1 Introduction to Nodal Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Nodal analysis** finds the **operating point** where IPR meets VLP.

**The Node:**

The "node" is typically placed at the **bottom of tubing** (most common) but can be placed anywhere:

- Wellhead
- Tubing bottom (most common)
- Perforations
- Separator

**System Analysis:**

1. **Inflow (IPR):** Reservoir → Node
2. **Outflow (VLP):** Node → Surface
3. **Operating point:** Where curves intersect

**Mathematical Statement:**

At equilibrium:

.. math::

   P_{wf,IPR} = P_{wf,VLP}

3.2 Nodal Analysis Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Complete Implementation:**

.. code-block:: python

   from petrosmith.core import ProductionCalculations
   import numpy as np
   import matplotlib.pyplot as plt
   
   def perform_nodal_analysis(
       reservoir_pressure,
       max_ipr_rate,
       tubing_depth,
       tubing_id,
       wellhead_pressure,
       oil_api,
       gas_sg,
       gor,
       water_cut
   ):
       """
       Perform complete nodal analysis.
       
       Returns:
           Operating rate, pressure, and analysis curves
       """
       
       # Generate rate range
       rates = np.linspace(0, max_ipr_rate, 50)
       
       ipr_pressures = []
       vlp_pressures = []
       
       for rate in rates:
           # IPR: Calculate Pwf from rate
           if rate == 0:
               pwf_ipr = reservoir_pressure
           else:
               # Vogel IPR
               rate_fraction = rate / max_ipr_rate
               pwf_pr = np.roots([
                   0.8,
                   0.2,
                   -(1 - rate_fraction)
               ])
               pwf_pr = pwf_pr[pwf_pr >= 0][0]
               pwf_ipr = pwf_pr * reservoir_pressure
           
           ipr_pressures.append(pwf_ipr)
           
           # VLP: Calculate Pwf required from surface
           if rate == 0:
               pwf_vlp = wellhead_pressure
           else:
               result = calculate_vlp_simplified(
                   oil_rate_stb=rate,
                   water_cut_percent=water_cut,
                   gor_scf_stb=gor,
                   tubing_depth_ft=tubing_depth,
                   tubing_id_inches=tubing_id,
                   wellhead_pressure_psi=wellhead_pressure,
                   oil_gravity_api=oil_api,
                   gas_gravity_air=gas_sg,
                   water_sg=1.05
               )
               pwf_vlp = result['bottomhole_pressure']
           
           vlp_pressures.append(pwf_vlp)
       
       # Find intersection (operating point)
       ipr_arr = np.array(ipr_pressures)
       vlp_arr = np.array(vlp_pressures)
       
       # Find where curves cross
       diff = np.abs(ipr_arr - vlp_arr)
       idx = np.argmin(diff)
       
       operating_rate = rates[idx]
       operating_pressure = ipr_arr[idx]
       
       # Plot results
       fig, ax = plt.subplots(figsize=(12, 8))
       
       ax.plot(rates, ipr_pressures, 'b-', linewidth=3, label='IPR (Inflow)')
       ax.plot(rates, vlp_pressures, 'r-', linewidth=3, label='VLP (Outflow)')
       ax.plot(operating_rate, operating_pressure, 'go', markersize=15, 
               label=f'Operating Point: {operating_rate:.0f} STB/day @ {operating_pressure:.0f} psi')
       
       ax.set_xlabel('Production Rate (STB/day)', fontsize=12)
       ax.set_ylabel('Bottomhole Pressure (psi)', fontsize=12)
       ax.set_title('Nodal Analysis - IPR vs VLP', fontsize=14, fontweight='bold')
       ax.grid(True, alpha=0.3)
       ax.legend(fontsize=11)
       plt.tight_layout()
       plt.show()
       
       return {
           'operating_rate': operating_rate,
           'operating_pressure': operating_pressure,
           'max_rate': max_ipr_rate,
           'ipr_curve': list(zip(rates, ipr_pressures)),
           'vlp_curve': list(zip(rates, vlp_pressures))
       }
   
   # Example: Perform nodal analysis
   result = perform_nodal_analysis(
       reservoir_pressure=3000,
       max_ipr_rate=1200,
       tubing_depth=8000,
       tubing_id=2.441,  # 2-7/8" tubing
       wellhead_pressure=150,
       oil_api=35,
       gas_sg=0.65,
       gor=400,
       water_cut=20
   )
   
   print("\n=== NODAL ANALYSIS RESULTS ===")
   print(f"Operating Rate: {result['operating_rate']:.0f} STB/day")
   print(f"Operating Pressure: {result['operating_pressure']:.0f} psi")
   print(f"Maximum Theoretical Rate: {result['max_rate']:.0f} STB/day")
   print(f"Production Efficiency: {result['operating_rate']/result['max_rate']*100:.1f}%")

**Interpretation:**

- **Operating rate:** Natural flow rate without artificial lift
- **Below operating point:** Well cannot flow naturally
- **Improvement options:**
  - Reduce wellhead pressure (larger flowline)
  - Larger tubing (reduce friction)
  - Artificial lift (change VLP curve)
  - Stimulation (improve IPR)

3.3 Sensitivity Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^

Evaluate impact of changes:

.. code-block:: python

   def sensitivity_analysis_tubing_size():
       """Analyze impact of tubing size on production"""
       
       tubing_sizes = [
           (1.995, '2-3/8"'),
           (2.441, '2-7/8"'),
           (2.992, '3-1/2"'),
           (3.958, '4-1/2"')
       ]
       
       results = []
       
       for tid, name in tubing_sizes:
           result = perform_nodal_analysis(
               reservoir_pressure=3000,
               max_ipr_rate=1200,
               tubing_depth=8000,
               tubing_id=tid,
               wellhead_pressure=150,
               oil_api=35,
               gas_sg=0.65,
               gor=400,
               water_cut=20
           )
           results.append((name, result['operating_rate']))
       
       # Display results
       print("\n=== TUBING SIZE SENSITIVITY ===")
       for name, rate in results:
           print(f"{name:8s}: {rate:6.0f} STB/day")
       
       return results

----

Part 4: Electric Submersible Pumps (ESP)
-----------------------------------------

4.1 ESP Fundamentals
^^^^^^^^^^^^^^^^^^^^

**Components:**

1. **Submersible motor** - Downhole electric motor
2. **Seal section** - Protects motor from well fluids
3. **Pump** - Centrifugal stages
4. **Cable** - Power from surface
5. **Surface controller** - Variable frequency drive (VFD)

**When to Use ESP:**

✅ High rate wells (500-50,000+ BPD)  
✅ Moderate to high water cut  
✅ Straight to deviated wells  
✅ Wells with moderate gas (<20% free gas)  

❌ High GOR (>500 scf/STB without gas separator)  
❌ Solids production (erosion/wear)  
❌ High H2S or CO2  

4.2 ESP Design Calculations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Required Head Calculation:**

.. math::

   H_{total} = H_{lift} + H_{friction} + H_{surface}

Where:
- :math:`H_{lift}` = Vertical lift (ft)
- :math:`H_{friction}` = Friction in tubing (ft)
- :math:`H_{surface}` = Surface pressure head (ft)

**PetroSmith Implementation:**

.. code-block:: python

   def design_esp_system(
       production_rate_bpd,
       pump_setting_depth_ft,
       static_fluid_level_ft,
       tubing_id_inches,
       wellhead_pressure_psi,
       fluid_sg,
       fluid_viscosity_cp
   ):
       """
       Design ESP system for given conditions.
       
       Returns:
           Recommended pump, stages, horsepower
       """
       import math
       
       # Calculate required head
       lift_head = pump_setting_depth_ft - static_fluid_level_ft
       
       # Friction loss in tubing
       velocity_fps = (production_rate_bpd * 5.615 / 86400) / (
           math.pi * (tubing_id_inches/12/2)**2
       )
       
       # Simplified friction (Darcy-Weisbach)
       f = 0.02  # Friction factor
       friction_head = (f * pump_setting_depth_ft * velocity_fps**2) / (
           2 * 32.2 * (tubing_id_inches/12)
       )
       
       # Surface pressure head
       surface_head = wellhead_pressure_psi * 2.31 / fluid_sg
       
       # Total dynamic head (TDH)
       tdh = lift_head + friction_head + surface_head
       
       # Hydraulic horsepower
       hhp = (production_rate_bpd * tdh * fluid_sg) / 3960
       
       # Motor horsepower (accounting for efficiency)
       pump_efficiency = 0.65  # Typical centrifugal pump
       motor_efficiency = 0.85  # Typical submersible motor
       
       bhp = hhp / (pump_efficiency * motor_efficiency)
       
       # Select pump series (simplified)
       if production_rate_bpd < 500:
           pump_series = "300-series"
           stages_per_100ft = 20
       elif production_rate_bpd < 2000:
           pump_series = "400-series"
           stages_per_100ft = 15
       elif production_rate_bpd < 5000:
           pump_series = "540-series"
           stages_per_100ft = 12
       else:
           pump_series = "675-series"
           stages_per_100ft = 10
       
       # Calculate number of stages
       head_per_stage = 100.0 / stages_per_100ft
       total_stages = math.ceil(tdh / head_per_stage)
       
       return {
           'pump_series': pump_series,
           'total_dynamic_head_ft': tdh,
           'lift_head_ft': lift_head,
           'friction_head_ft': friction_head,
           'surface_head_ft': surface_head,
           'stages_required': total_stages,
           'hydraulic_hp': hhp,
           'brake_hp': bhp,
           'recommended_motor_hp': math.ceil(bhp / 5) * 5,  # Round to 5 HP
           'pump_efficiency': pump_efficiency,
           'motor_efficiency': motor_efficiency
       }
   
   # Example: Design ESP
   esp_design = design_esp_system(
       production_rate_bpd=2000,
       pump_setting_depth_ft=6000,
       static_fluid_level_ft=4500,
       tubing_id_inches=3.958,  # 4-1/2" tubing
       wellhead_pressure_psi=150,
       fluid_sg=0.92,
       fluid_viscosity_cp=5.0
   )
   
   print("\n=== ESP DESIGN SUMMARY ===")
   print(f"Recommended Pump: {esp_design['pump_series']}")
   print(f"Stages Required: {esp_design['stages_required']}")
   print(f"Motor Size: {esp_design['recommended_motor_hp']} HP")
   print(f"\n--- Head Breakdown ---")
   print(f"Lift Head: {esp_design['lift_head_ft']:.0f} ft")
   print(f"Friction Head: {esp_design['friction_head_ft']:.0f} ft")
   print(f"Surface Head: {esp_design['surface_head_ft']:.0f} ft")
   print(f"Total Dynamic Head: {esp_design['total_dynamic_head_ft']:.0f} ft")
   print(f"\n--- Power Requirements ---")
   print(f"Hydraulic HP: {esp_design['hydraulic_hp']:.1f} HP")
   print(f"Brake HP: {esp_design['brake_hp']:.1f} HP")
   print(f"Pump Efficiency: {esp_design['pump_efficiency']*100:.0f}%")
   print(f"Motor Efficiency: {esp_design['motor_efficiency']*100:.0f}%")

4.3 ESP Performance Curves
^^^^^^^^^^^^^^^^^^^^^^^^^^^

ESP manufacturers provide **pump performance curves**:

.. code-block:: python

   def generate_esp_performance_curve(pump_series, speed_hz):
       """
       Generate ESP performance curve for given pump and speed.
       
       Simplified model based on typical characteristics.
       """
       import numpy as np
       
       # Typical pump characteristics (simplified)
       if pump_series == "400-series":
           base_rate = 1500  # BPD at BEP
           base_head = 10    # ft/stage at BEP
           base_efficiency = 0.68
       else:
           base_rate = 2000
           base_head = 12
           base_efficiency = 0.70
       
       # Scale with frequency
       rate_scale = speed_hz / 60.0
       head_scale = (speed_hz / 60.0) ** 2
       
       # Generate curve
       rates = np.linspace(0, base_rate * rate_scale * 1.5, 50)
       heads = []
       efficiencies = []
       
       for rate in rates:
           # Head curve (parabolic)
           rate_fraction = rate / (base_rate * rate_scale)
           head = base_head * head_scale * (1.2 - 0.5 * rate_fraction**2)
           heads.append(head)
           
           # Efficiency curve (peaks at BEP)
           eff = base_efficiency * (1 - 0.8 * (rate_fraction - 1)**2)
           eff = max(0.3, min(0.75, eff))
           efficiencies.append(eff)
       
       return rates, heads, efficiencies
   
   # Plot ESP curves
   rates, heads, effs = generate_esp_performance_curve("400-series", 60)
   
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
   
   ax1.plot(rates, heads, 'b-', linewidth=2)
   ax1.set_xlabel('Rate (BPD)')
   ax1.set_ylabel('Head per Stage (ft)')
   ax1.set_title('ESP Head Curve')
   ax1.grid(True, alpha=0.3)
   
   ax2.plot(rates, [e*100 for e in effs], 'g-', linewidth=2)
   ax2.set_xlabel('Rate (BPD)')
   ax2.set_ylabel('Efficiency (%)')
   ax2.set_title('ESP Efficiency Curve')
   ax2.grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.show()

----

Part 5: Gas Lift Design
------------------------

5.1 Gas Lift Fundamentals
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Principle:**

Inject high-pressure gas into tubing to:
- Reduce fluid density
- Reduce bottomhole pressure
- Increase production rate

**Types:**

1. **Continuous Gas Lift** - Continuous gas injection
2. **Intermittent Gas Lift** - Periodic gas injection (low rate wells)

**When to Use Gas Lift:**

✅ High GOR wells  
✅ Moderate rates (100-5000 BPD)  
✅ Deviated/horizontal wells  
✅ Multiple wells (infrastructure leverage)  
✅ Remote locations  

5.2 Gas Lift Design
^^^^^^^^^^^^^^^^^^^^

**Key Design Parameters:**

- **Injection depth** - Where to inject gas
- **Injection rate** - How much gas
- **Valve spacing** - Unloading sequence
- **Operating valve** - Final injection point

**Design Workflow:**

.. code-block:: python

   def design_continuous_gas_lift(
       reservoir_pressure,
       productivity_index,
       tubing_depth,
       static_gradient,
       available_injection_pressure,
       available_gas_rate_mscfd,
       separator_pressure
   ):
       """
       Design continuous gas lift system.
       
       Returns:
           Optimal injection depth, rate, and predicted production
       """
       import numpy as np
       
       # Step 1: Determine maximum injection depth
       # Gas must have enough pressure to open valve
       
       valve_pressure_drop = 50  # psi (across valve)
       
       # Pressure available at depth
       max_depth = 0
       for depth in range(1000, tubing_depth, 100):
           # Pressure at depth (assuming gas in annulus)
           gas_gradient = 0.1  # psi/ft (approximate for gas)
           pressure_at_depth = available_injection_pressure - gas_gradient * depth
           
           # Required pressure to lift fluid
           required_pressure = static_gradient * depth + separator_pressure
           
           if pressure_at_depth > required_pressure + valve_pressure_drop:
               max_depth = depth
       
       # Step 2: Optimize injection rate
       best_rate = 0
       best_prod = 0
       
       injection_depth = max_depth
       
       for gas_rate in np.linspace(100, available_gas_rate_mscfd, 20):
           # Calculate new gradient with gas injection
           # Simplified: GLR effect on gradient
           glr_total = gas_rate * 1000 / (best_prod + 100)  # scf/bbl
           
           # Reduced gradient due to gas
           effective_gradient = static_gradient * (1 - glr_total / 10000)
           
           # New bottomhole pressure
           pwf = separator_pressure + effective_gradient * tubing_depth
           
           # New production rate from IPR
           prod_rate = productivity_index * (reservoir_pressure - pwf)
           
           if prod_rate > best_prod:
               best_prod = prod_rate
               best_rate = gas_rate
       
       return {
           'injection_depth_ft': injection_depth,
           'injection_rate_mscfd': best_rate,
           'predicted_production_bpd': best_prod,
           'required_injection_pressure_psi': available_injection_pressure,
           'number_of_valves': int(injection_depth / 1000) + 1
       }
   
   # Example: Design gas lift
   gl_design = design_continuous_gas_lift(
       reservoir_pressure=3000,
       productivity_index=2.0,
       tubing_depth=8000,
       static_gradient=0.38,  # psi/ft
       available_injection_pressure=1500,
       available_gas_rate_mscfd=2000,
       separator_pressure=100
   )
   
   print("\n=== GAS LIFT DESIGN ===")
   print(f"Injection Depth: {gl_design['injection_depth_ft']:,.0f} ft")
   print(f"Injection Rate: {gl_design['injection_rate_mscfd']:.0f} Mscf/day")
   print(f"Predicted Production: {gl_design['predicted_production_bpd']:.0f} BPD")
   print(f"Number of Valves: {gl_design['number_of_valves']}")

5.3 Gas Lift Valve Sizing
^^^^^^^^^^^^^^^^^^^^^^^^^^

Gas lift valves must be sized correctly:

.. code-block:: python

   def size_gas_lift_valve(
       gas_rate_mscfd,
       upstream_pressure_psi,
       downstream_pressure_psi,
       temperature_f,
       gas_gravity
   ):
       """
       Size gas lift valve using orifice equation.
       
       Returns port size in 1/64"
       """
       import math
       
       # Convert to absolute pressure
       p1 = upstream_pressure_psi + 14.7
       p2 = downstream_pressure_psi + 14.7
       temp_r = temperature_f + 460
       
       # Check for critical flow
       critical_ratio = (2 / (gas_gravity + 1)) ** (gas_gravity / (gas_gravity - 1))
       pressure_ratio = p2 / p1
       
       # Thornhill-Craver equation (simplified)
       if pressure_ratio < critical_ratio:
           # Critical flow
           c = 0.865  # Discharge coefficient
       else:
           # Subcritical flow
           c = 0.865
       
       # Calculate required orifice area (sq inches)
       area = (gas_rate_mscfd * 1000 * math.sqrt(gas_gravity * temp_r)) / (
           c * p1 * 3.8
       )
       
       # Convert to diameter
       diameter_inches = math.sqrt(4 * area / math.pi)
       
       # Convert to port size (1/64")
       port_size_64ths = diameter_inches * 64
       
       # Round to standard sizes
       standard_sizes = [8, 12, 16, 20, 24, 28, 32, 40]
       port_size = min(standard_sizes, key=lambda x: abs(x - port_size_64ths))
       
       return {
           'port_size_64ths': port_size,
           'port_size_decimal': port_size / 64,
           'calculated_area_sqin': area,
           'flow_regime': 'critical' if pressure_ratio < critical_ratio else 'subcritical'
       }

----

Part 6: Well Testing
--------------------

6.1 Introduction to Well Testing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Purpose of Well Testing:**

- Estimate reservoir properties (k, k·h, skin)
- Determine well productivity
- Identify boundaries/faults
- Measure reservoir pressure
- Characterize reservoir heterogeneity

**Test Types:**

1. **Drawdown Test** - Open well, record pressure decline
2. **Buildup Test** - Shut in well, record pressure increase
3. **Multi-rate Test** - Change rates, analyze response
4. **Interference Test** - Produce one well, observe others

6.2 Buildup Test Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Theory:**

After shut-in, pressure increases according to:

.. math::

   P_{ws} = P_i - \\frac{162.6 q B \\mu}{k h} \\left[ \\log\\left(\\frac{t_p + \\Delta t}{\\Delta t}\\right) \\right]

**Horner Plot:**

Plot Pws vs log[(tp + Δt)/Δt]

.. code-block:: python

   from petrosmith.core.well_testing import BuildupAnalysis, WellTestData, ReservoirProperties
   import numpy as np
   import matplotlib.pyplot as plt
   
   # Create test data
   shut_in_times = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]  # hours
   pressures = [1800, 2100, 2300, 2450, 2600, 2700, 2780, 2850, 2900]  # psi
   
   test_data = WellTestData(
       time_hours=shut_in_times,
       pressure_psi=pressures,
       flow_rate_stb=500,
       test_type='buildup'
   )
   
   fluid_props = ReservoirProperties(
       porosity=0.20,
       permeability=100,
       formation_thickness=50,
       oil_viscosity_cp=2.0,
       oil_fvf_rb_stb=1.2,
       total_compressibility_1_psi=1e-5
   )
   
   # Analyze buildup
   analyzer = BuildupAnalysis(test_data, fluid_props)
   results = analyzer.analyze(
       wellbore_radius_ft=0.328,
       production_time_hours=720  # 30 days
   )
   
   print("\n=== BUILDUP TEST ANALYSIS ===")
   print(f"Permeability: {results['permeability_md']:.1f} md")
   print(f"Skin Factor: {results['skin_factor']:.2f}")
   print(f"Reservoir Pressure: {results['reservoir_pressure_psi']:.0f} psi")
   print(f"k·h: {results['kh']:.0f} md-ft")

**MDH Plot (Miller-Dyes-Hutchinson):**

Alternative to Horner for short tests:

.. code-block:: python

   def create_mdh_plot(shut_in_times, pressures):
       """Create MDH plot: Pws vs log(Δt)"""
       
       fig, ax = plt.subplots(figsize=(10, 7))
       
       ax.semilogx(shut_in_times, pressures, 'bo-', markersize=8, linewidth=2)
       ax.set_xlabel('Shut-in Time, Δt (hours)', fontsize=12)
       ax.set_ylabel('Shut-in Pressure, Pws (psi)', fontsize=12)
       ax.set_title('MDH Plot - Buildup Test', fontsize=14, fontweight='bold')
       ax.grid(True, which='both', alpha=0.3)
       
       # Fit straight line to middle-time region
       log_times = np.log10(shut_in_times[2:7])
       mid_pressures = pressures[2:7]
       
       coeffs = np.polyfit(log_times, mid_pressures, 1)
       slope = coeffs[0]
       
       # Plot fit line
       fit_times = np.logspace(-1, 2, 100)
       fit_pressures = np.polyval(coeffs, np.log10(fit_times))
       ax.plot(fit_times, fit_pressures, 'r--', linewidth=2, label=f'Slope = {slope:.0f} psi/cycle')
       
       ax.legend()
       plt.tight_layout()
       plt.show()
       
       return slope

----

Summary and Key Takeaways
--------------------------

**What You Learned:**

✅ IPR analysis (Vogel, Fetkovich, Composite)  
✅ VLP calculations and curve generation  
✅ Nodal analysis for system optimization  
✅ ESP design and selection  
✅ Gas lift design and valve sizing  
✅ Rod pump design  
✅ Well test analysis and interpretation  

**Production Optimization Workflow:**

1. **Characterize reservoir** (well testing, IPR)
2. **Analyze system** (nodal analysis)
3. **Identify bottleneck** (IPR vs VLP)
4. **Select artificial lift** (based on conditions)
5. **Design system** (ESP/gas lift/rod pump)
6. **Monitor performance** (production surveillance)
7. **Optimize operations** (continuous improvement)

**Artificial Lift Selection Matrix:**

.. code-block:: text

   Condition          ESP    Gas Lift   Rod Pump   PCP
   ─────────────────────────────────────────────────────
   High rate          ✅      ✅         ❌         ❌
   High water cut     ✅      ✅         ⚠️         ✅
   High GOR           ❌      ✅         ❌         ❌
   Deviated well      ✅      ✅         ❌         ⚠️
   Solids             ❌      ✅         ❌         ✅
   Remote location    ❌      ✅         ✅         ❌
   Low rate           ❌      ⚠️         ✅         ✅

**Next Steps:**

- Apply these methods to your field data
- Build production optimization workflows
- Practice nodal analysis with real wells
- Learn advanced well test interpretation

----

Practice Exercises
------------------

**Exercise 6.1:** Nodal Analysis

Given:
- Pr = 2800 psi
- Vogel IPR with qmax = 1000 STB/day
- Depth = 7500 ft
- Tubing = 2-7/8"
- WHP = 100 psi

Task: Find natural flow rate

**Exercise 6.2:** ESP Design

Design ESP for:
- Rate = 1500 BPD
- Pump depth = 5500 ft
- SFL = 4200 ft
- WHP = 150 psi

**Exercise 6.3:** Gas Lift Optimization

Optimize injection rate for maximum production.

----

**Module Complete!** ✅

**Next:** :doc:`module7_well_completions`
