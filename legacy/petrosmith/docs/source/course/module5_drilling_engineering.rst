Module 5: Drilling Engineering
===============================

**Learning Objectives**

By the end of this module, you will be able to:

- Calculate hydrostatic pressure and equivalent circulating density (ECD)
- Design casing programs for well integrity
- Perform hydraulics calculations for drilling optimization
- Apply well control principles and kill procedures
- Analyze drill string mechanics and buckling
- Design BHA configurations for directional drilling
- Calculate tripping speeds and surge/swab pressures

**Prerequisites:** :doc:`module1_fundamentals`, basic drilling operations knowledge

**Time Commitment:** 8-10 hours

---

Introduction: The Drilling Engineer's Mission
---------------------------------------------

**Primary Objective:** Drill wells safely, on time, and on budget

Your responsibilities:

1. **Well Design:** Casing program, mud weights, trajectory
2. **Hydraulics Optimization:** Maximize ROP while maintaining hole cleaning
3. **Well Control:** Prevent kicks and manage abnormal pressures  
4. **Cost Management:** Minimize NPT (non-productive time)
5. **Safety:** Protect personnel, environment, and equipment

**The Cost of Failure:**

- Stuck pipe: $100K - $5M
- Lost circulation: $50K - $2M per event
- Blowout: $10M - $1B+ (plus lives and environmental damage)

**PetroSmith for Drilling:**

✓ Validate well designs before spud  
✓ Real-time hydraulics monitoring  
✓ Well control calculations at the rigsite  
✓ Automated daily reporting  
✓ Post-well analysis and lessons learned  

---

Section 1: Pressure Fundamentals
---------------------------------

1.1 Hydrostatic Pressure
~~~~~~~~~~~~~~~~~~~~~~~~~

The foundational calculation in drilling:

.. math::

   P_{hydrostatic} = 0.052 \times MW \times TVD

Where:

- P = Pressure (psi)
- MW = Mud weight (ppg)
- TVD = True vertical depth (feet)
- 0.052 = Conversion constant (psi/ft per ppg)

.. code-block:: python

   from petrosmith import DrillingAPI
   from petrosmith.core.constants import PhysicalConstants
   
   api = DrillingAPI()
   
   # Calculate hydrostatic pressure
   mud_weight = 11.5  # ppg
   tvd = 10000  # feet
   
   hydrostatic_pressure = api.calculate_hydrostatic_pressure(
       mud_weight=mud_weight,
       true_vertical_depth=tvd
   )
   
   print(f"Drilling @ {tvd:,} ft TVD with {mud_weight} ppg mud")
   print(f"Hydrostatic Pressure: {hydrostatic_pressure:,.0f} psi")
   print(f"Pressure Gradient: {hydrostatic_pressure/tvd:.4f} psi/ft")
   
   # What mud weight needed for specific pressure?
   target_pressure = 6000  # psi (overbalance formation pressure)
   required_mud_weight = target_pressure / (PhysicalConstants.HYDROSTATIC_GRADIENT * tvd)
   
   print(f"\nTo achieve {target_pressure:,} psi at {tvd:,} ft:")
   print(f"Required Mud Weight: {required_mud_weight:.2f} ppg")

**Output:**

.. code-block:: text

   Drilling @ 10,000 ft TVD with 11.5 ppg mud
   Hydrostatic Pressure: 5,980 psi
   Pressure Gradient: 0.5980 psi/ft
   
   To achieve 6,000 psi at 10,000 ft:
   Required Mud Weight: 11.54 ppg

1.2 Pore Pressure and Fracture Gradient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Understanding formation pressures is critical for well control:

**Normal Pressure:**
- ~0.433 psi/ft (fresh water)
- ~0.465 psi/ft (salt water)
- Equivalent to 8.33 - 8.95 ppg

**Abnormal Pressure:**
- Overpressured: >0.6 psi/ft (can reach 1.0+ psi/ft)
- Underpressured: <0.433 psi/ft (depleted reservoirs)

**Fracture Pressure:**
- Depth dependent
- Typically 0.7 - 1.0 psi/ft in shallow sections
- Can reach 1.2+ psi/ft in deep sections

**Drilling Window:**

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   
   # Define pressure profile vs depth
   depths = np.array([0, 2000, 4000, 6000, 8000, 10000, 12000, 14000])
   
   # Pore pressure (overpressured below 8000 ft)
   pore_pressure_grad = np.array([0.465, 0.465, 0.465, 0.465, 0.520, 0.620, 0.720, 0.820])
   pore_pressure = pore_pressure_grad * depths
   
   # Fracture gradient
   frac_gradient = np.array([0.70, 0.75, 0.78, 0.81, 0.85, 0.88, 0.92, 0.95])
   frac_pressure = frac_gradient * depths
   
   # Required mud weight (with safety margin)
   overbalance = 200  # psi safety margin
   mud_weight_pressure = pore_pressure + overbalance
   mud_weight = mud_weight_pressure / (0.052 * depths)
   
   print("Drilling Window Analysis")
   print("=" * 70)
   print(f"{'Depth':>6s} {'Pore (psi)':>10s} {'Frac (psi)':>10s} {'MW (ppg)':>10s} {'Window (ppg)':>12s}")
   print("=" * 70)
   
   for i, depth in enumerate(depths[1:], 1):
       pore = pore_pressure[i]
       frac = frac_pressure[i]
       mw = mud_weight[i]
       window = (frac - mud_weight_pressure[i]) / (0.052 * depth)
       
       print(f"{depth:6.0f} {pore:10.0f} {frac:10.0f} {mw:10.2f} {window:12.2f}")
       
       # Check for narrow window
       if window < 1.0:
           print(f"       ⚠️  WARNING: Narrow drilling window at {depth} ft!")

**Engineering Insight:**

Below 10,000 ft, the drilling window narrows significantly due to overpressure. This may require:

- Intermediate casing string
- Managed pressure drilling (MPD)
- Lower ROP to minimize ECD

1.3 Equivalent Circulating Density (ECD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ECD accounts for annular pressure losses while circulating:

.. math::

   ECD = MW + \frac{P_{annular}}{0.052 \times TVD}

.. code-block:: python

   from petrosmith import DrillingAPI
   
   api = DrillingAPI()
   
   # Drilling parameters
   mud_weight = 12.0  # ppg
   tvd = 12000  # ft
   flow_rate = 450  # gpm
   hole_diameter = 8.5  # inches
   pipe_od = 5.0  # inches
   
   # Calculate ECD
   ecd = api.calculate_ecd(
       mud_weight=mud_weight,
       flow_rate=flow_rate,
       annular_velocity=flow_rate / 24.5,  # Simplified
       pipe_length=tvd,
       viscosity=35  # cp
   )
   
   frac_gradient = 0.88  # psi/ft
   frac_pressure = frac_gradient * tvd
   frac_equivalent_mw = frac_pressure / (0.052 * tvd)
   
   print(f"ECD Analysis at {tvd:,} ft TVD")
   print("=" * 50)
   print(f"Static Mud Weight: {mud_weight:.2f} ppg")
   print(f"ECD (circulating): {ecd:.2f} ppg")
   print(f"Fracture Equivalent: {frac_equivalent_mw:.2f} ppg")
   print(f"Margin to Frac: {frac_equivalent_mw - ecd:.2f} ppg")
   
   if ecd >= frac_equivalent_mw:
       print("\n❌ DANGER: ECD exceeds fracture pressure!")
       print("   Recommendation: Reduce flow rate or modify mud properties")
   elif (frac_equivalent_mw - ecd) < 0.5:
       print("\n⚠️  WARNING: Insufficient margin to fracture")
   else:
       print("\n✓ Safe drilling window")

**Optimizing ECD:**

To reduce ECD:

1. Reduce flow rate (but maintain hole cleaning)
2. Use low-viscosity mud (but maintain cuttings suspension)
3. Increase annular clearance (larger hole or smaller pipe)
4. Use managed pressure drilling (MPD)

---

Section 2: Hydraulics Optimization
-----------------------------------

2.1 Pump Pressure and Horsepower
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Total pump pressure consists of surface and downhole losses:

.. math::

   P_{pump} = P_{surface} + P_{drill\_string} + P_{bit} + P_{annulus}

.. code-block:: python

   from petrosmith.services import DrillingService
   
   service = DrillingService()
   
   # Drilling parameters
   flow_rate = 500  # gpm
   mud_weight = 10.5  # ppg
   mud_viscosity = 40  # cp
   
   # Drill string
   drill_pipe_length = 10000  # ft
   drill_pipe_id = 4.276  # inches
   
   # Bit
   bit_tfa = 0.45  # Total flow area, sq inches
   bit_pressure_drop = 350 * (flow_rate / bit_tfa) ** 2  # Simplified
   
   # Surface equipment pressure drop
   surface_pressure = 200  # psi
   
   # Drill string pressure drop (simplified)
   pipe_pressure = 0.0082 * (flow_rate ** 1.8) * (drill_pipe_length / drill_pipe_id ** 4.8)
   
   # Annular pressure drop
   annular_pressure = 150  # psi (calculated)
   
   # Total pump pressure
   total_pressure = surface_pressure + pipe_pressure + bit_pressure_drop + annular_pressure
   
   # Hydraulic horsepower
   hhp = (flow_rate * total_pressure) / 1714
   
   print("Hydraulics Analysis")
   print("=" * 50)
   print(f"Flow Rate: {flow_rate} gpm")
   print(f"Mud Weight: {mud_weight} ppg")
   print(f"\nPressure Breakdown:")
   print(f"  Surface Equipment: {surface_pressure:6.0f} psi")
   print(f"  Drill String: {pipe_pressure:6.0f} psi")
   print(f"  Bit: {bit_pressure_drop:6.0f} psi")
   print(f"  Annulus: {annular_pressure:6.0f} psi")
   print(f"  TOTAL: {total_pressure:6.0f} psi")
   print(f"\nHydraulic Horsepower: {hhp:.1f} hp")

2.2 Hole Cleaning
~~~~~~~~~~~~~~~~~

Critical for preventing stuck pipe and maintaining ROP.

**Cuttings Transport Ratio (CTR):**

.. math::

   CTR = \frac{V_{annular}}{V_{slip}}

Where:

- V_annular = Annular velocity (ft/min)
- V_slip = Cuttings slip velocity (ft/min)

**Target:** CTR > 1.5 (cuttings moving up faster than settling)

.. code-block:: python

   def calculate_hole_cleaning(
       flow_rate: float,  # gpm
       hole_diameter: float,  # inches
       pipe_od: float,  # inches
       rop: float,  # ft/hr
       mud_weight: float  # ppg
   ) -> dict:
       """
       Analyze hole cleaning efficiency.
       
       Returns:
           Dictionary with cleaning metrics
       """
       # Annular velocity
       annular_area = (hole_diameter**2 - pipe_od**2) / 183.35
       annular_velocity = flow_rate / annular_area  # ft/min
       
       # Cuttings concentration
       penetration_rate_fpm = rop / 60
       cuttings_volume = (hole_diameter**2 / 183.35) * penetration_rate_fpm  # gpm
       cuttings_concentration = cuttings_volume / flow_rate * 100
       
       # Slip velocity (simplified - assumes 2.65 SG cuttings)
       slip_velocity = 80 / mud_weight  # ft/min (empirical)
       
       # Transport ratio
       ctr = annular_velocity / slip_velocity
       
       # Hole cleaning assessment
       if ctr < 1.0:
           status = "POOR - Cuttings settling"
       elif ctr < 1.5:
           status = "MARGINAL - Monitor closely"
       elif ctr < 2.5:
           status = "GOOD - Adequate cleaning"
       else:
           status = "EXCELLENT - Optimal cleaning"
       
       return {
           "annular_velocity": annular_velocity,
           "slip_velocity": slip_velocity,
           "ctr": ctr,
           "cuttings_concentration": cuttings_concentration,
           "status": status
       }
   
   # Example: Horizontal section
   result = calculate_hole_cleaning(
       flow_rate=550,
       hole_diameter=8.5,
       pipe_od=5.0,
       rop=120,  # ft/hr - typical horizontal
       mud_weight=11.0
   )
   
   print("Hole Cleaning Analysis - Horizontal Section")
   print("=" * 50)
   print(f"Annular Velocity: {result['annular_velocity']:.1f} ft/min")
   print(f"Slip Velocity: {result['slip_velocity']:.1f} ft/min")
   print(f"Transport Ratio (CTR): {result['ctr']:.2f}")
   print(f"Cuttings Concentration: {result['cuttings_concentration']:.2f}%")
   print(f"\nStatus: {result['status']}")

**Optimization Strategies:**

If hole cleaning is poor:

1. **Increase flow rate** (most effective)
2. **Reduce ROP** (temporary measure)
3. **Rotate pipe** (agitation helps)
4. **Perform wiper trips** (clean hole mechanically)
5. **Add viscosifiers** (suspend cuttings)

---

Section 3: Casing Design
-------------------------

3.1 Casing Load Cases
~~~~~~~~~~~~~~~~~~~~~~

Casing must withstand:

1. **Burst:** Internal pressure exceeds external
2. **Collapse:** External pressure exceeds internal
3. **Tension:** Weight of casing string
4. **Compression:** Buckling in deviated wells

**Burst Pressure:**

.. math::

   P_{burst} = \frac{2 \times t \times \sigma_y}{OD} \times SF

Where:

- t = Wall thickness
- σ_y = Yield strength
- OD = Outside diameter
- SF = Safety factor (typically 1.1)

.. code-block:: python

   from petrosmith import DrillingAPI
   
   api = DrillingAPI()
   
   # Production casing scenario
   casing_od = 7.0  # inches
   casing_weight = 26  # lb/ft
   casing_grade = "L-80"  # 80,000 psi yield
   setting_depth = 10000  # ft TVD
   
   # Calculate burst rating
   burst_rating = api.calculate_casing_burst(
       outer_diameter=casing_od,
       weight=casing_weight,
       grade=casing_grade
   )
   
   # Maximum expected pressure (kick scenario)
   formation_pressure = 5500  # psi
   kick_pressure = formation_pressure + 500  # Safety margin
   
   # External pressure (mud column)
   mud_weight = 11.0  # ppg
   external_pressure = 0.052 * mud_weight * setting_depth
   
   # Net burst pressure
   net_burst = kick_pressure - external_pressure
   
   # Safety factor
   burst_sf = burst_rating / net_burst
   
   print("Casing Burst Analysis")
   print("=" * 60)
   print(f"Casing: {casing_od}\" {casing_weight} lb/ft {casing_grade}")
   print(f"Setting Depth: {setting_depth:,} ft TVD")
   print(f"\nBurst Rating: {burst_rating:,.0f} psi")
   print(f"Formation Pressure: {formation_pressure:,.0f} psi")
   print(f"Kick Pressure: {kick_pressure:,.0f} psi")
   print(f"External Pressure: {external_pressure:,.0f} psi")
   print(f"Net Burst Load: {net_burst:,.0f} psi")
   print(f"\nSafety Factor: {burst_sf:.2f}")
   
   if burst_sf < 1.1:
       print("❌ FAIL: Insufficient burst safety factor")
   elif burst_sf < 1.3:
       print("⚠️  MARGINAL: Consider heavier wall")
   else:
       print("✓ PASS: Adequate burst strength")

3.2 Collapse Pressure
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate collapse rating
   collapse_rating = api.calculate_casing_collapse(
       outer_diameter=casing_od,
       weight=casing_weight,
       grade=casing_grade
   )
   
   # Worst case: Empty casing, full mud column outside
   external_pressure_collapse = 0.052 * mud_weight * setting_depth
   internal_pressure_collapse = 0  # Empty casing
   
   net_collapse = external_pressure_collapse - internal_pressure_collapse
   collapse_sf = collapse_rating / net_collapse
   
   print("\nCasing Collapse Analysis")
   print("=" * 60)
   print(f"Collapse Rating: {collapse_rating:,.0f} psi")
   print(f"External Pressure: {external_pressure_collapse:,.0f} psi")
   print(f"Internal Pressure: {internal_pressure_collapse:,.0f} psi")
   print(f"Net Collapse Load: {net_collapse:,.0f} psi")
   print(f"\nSafety Factor: {collapse_sf:.2f}")
   
   if collapse_sf < 1.0:
       print("❌ FAIL: Casing will collapse!")
   elif collapse_sf < 1.125:
       print("⚠️  MARGINAL: Increase wall thickness")
   else:
       print("✓ PASS: Adequate collapse strength")

3.3 Tension and Axial Loads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Casing string design
   casing_sections = [
       {"length": 3000, "weight": 32, "grade": "P-110"},  # Heavy wall on bottom
       {"length": 4000, "weight": 29, "grade": "L-80"},   # Mid section
       {"length": 3000, "weight": 26, "grade": "L-80"}    # Top section
   ]
   
   print("\nCasing String Tension Analysis")
   print("=" * 60)
   
   cumulative_weight = 0
   
   for i, section in enumerate(reversed(casing_sections), 1):
       length = section["length"]
       weight = section["weight"]
       grade = section["grade"]
       
       # Weight of this section
       section_weight = length * weight
       cumulative_weight += section_weight
       
       # Buoyancy factor (assuming 12 ppg mud)
       buoyancy_factor = 1 - (0.015 * 12)  # ~0.82
       effective_weight = cumulative_weight * buoyancy_factor
       
       # Yield strength in tension
       casing_area = weight / 10.68  # Approximate
       yield_load = casing_area * (80000 if "L-80" in grade else 110000)
       
       sf_tension = yield_load / effective_weight
       
       print(f"\nSection {i} ({length:,} ft):")
       print(f"  Grade: {grade}, Weight: {weight} lb/ft")
       print(f"  Cumulative Weight (air): {cumulative_weight:,.0f} lbs")
       print(f"  Effective Weight (buoyed): {effective_weight:,.0f} lbs")
       print(f"  Yield Load: {yield_load:,.0f} lbs")
       print(f"  Safety Factor: {sf_tension:.2f}")
       
       if sf_tension < 1.6:
           print(f"  ⚠️  WARNING: Low tension safety factor")

---

Section 4: Well Control
------------------------

4.1 Kick Detection
~~~~~~~~~~~~~~~~~~

Early warning signs of a kick:

1. **Pit gain** - Increased mud volume
2. **Flow rate increase** - More mud returning than pumping
3. **Pump pressure decrease** - Lighter fluid column
4. **ROP increase** - Drilling break into pressured zone

.. code-block:: python

   class KickDetection:
       """Monitor for kick indicators."""
       
       def __init__(self):
           self.baseline_pit_volume = 0
           self.pit_gain_threshold = 5  # barrels
           
       def check_pit_gain(self, current_volume: float) -> dict:
           """Check for pit gain indicating possible kick."""
           gain = current_volume - self.baseline_pit_volume
           
           if gain >= self.pit_gain_threshold:
               severity = "CRITICAL" if gain > 20 else "WARNING"
               return {
                   "kick_detected": True,
                   "pit_gain": gain,
                   "severity": severity,
                   "action": "SHUT IN WELL IMMEDIATELY"
               }
           return {"kick_detected": False, "pit_gain": gain}
   
   # Simulation
   detector = KickDetection()
   detector.baseline_pit_volume = 500  # barrels
   
   # Normal drilling
   result = detector.check_pit_gain(502)
   print(f"Normal: {result}")
   
   # Kick scenario
   result = detector.check_pit_gain(515)
   print(f"\nKick detected: {result}")

4.2 Well Control Calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Driller's Method** (most common):

1. Shut in well
2. Record SIDPP (shut-in drill pipe pressure) and SICP (shut-in casing pressure)
3. Calculate kill mud weight
4. Circulate out kick while maintaining constant BHP

.. math::

   MW_{kill} = MW_{original} + \frac{SIDPP}{0.052 \times TVD}

.. code-block:: python

   from petrosmith import DrillingAPI
   
   api = DrillingAPI()
   
   # Kick scenario
   original_mw = 11.0  # ppg
   tvd = 12000  # ft
   sidpp = 450  # psi
   sicp = 650  # psi
   
   # Calculate kill mud weight
   kill_mw = original_mw + (sidpp / (0.052 * tvd))
   
   # Formation pressure
   formation_pressure = (0.052 * original_mw * tvd) + sidpp
   formation_gradient = formation_pressure / tvd
   
   # Kick intensity (difference between formation and hydrostatic)
   kick_intensity = sidpp
   
   # Kick volume (estimated from pit gain)
   pit_gain = 15  # barrels
   
   print("Well Control Analysis - Kick Scenario")
   print("=" * 60)
   print(f"Depth: {tvd:,} ft TVD")
   print(f"Original Mud Weight: {original_mw:.2f} ppg")
   print(f"SIDPP: {sidpp} psi")
   print(f"SICP: {sicp} psi")
   print(f"\nFormation Pressure: {formation_pressure:,.0f} psi")
   print(f"Formation Gradient: {formation_gradient:.3f} psi/ft")
   print(f"\nKill Mud Weight: {kill_mw:.2f} ppg")
   print(f"Pit Gain: {pit_gain} bbls")
   
   # Safety checks
   frac_gradient = 0.88  # psi/ft
   frac_pressure = frac_gradient * tvd
   frac_mw = frac_pressure / (0.052 * tvd)
   
   if kill_mw > frac_mw:
       print(f"\n❌ CRITICAL: Kill mud ({kill_mw:.2f} ppg) exceeds fracture ({frac_mw:.2f} ppg)")
       print("   Consider: Volumetric method or staged kill")
   else:
       print(f"\n✓ Kill mud within fracture gradient")
       print(f"   Margin: {frac_mw - kill_mw:.2f} ppg")

4.3 Kill Procedure Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def simulate_well_kill(
       sidpp: float,
       original_mw: float,
       kill_mw: float,
       tvd: float,
       annular_volume: float,  # barrels
       pump_output: float  # bbl/stroke
   ):
       """
       Simulate Driller's Method well kill.
       
       Returns:
           List of (strokes, pressure) tuples for kill sheet
       """
       # Initial circulating pressure
       icp = sidpp + 200  # Add pump pressure
       
       # Final circulating pressure
       fcp = 200  # Just pump pressure with kill mud
       
       # Strokes to circulate string
       drill_string_volume = annular_volume * 0.6  # Approximate
       strokes_to_bit = drill_string_volume / pump_output
       
       # Kill schedule
       schedule = []
       strokes_increments = int(strokes_to_bit / 10)
       
       for i in range(11):
           strokes = i * strokes_increments
           # Linear pressure reduction
           pressure = icp - (icp - fcp) * (strokes / strokes_to_bit)
           schedule.append((strokes, pressure))
       
       return schedule, strokes_to_bit
   
   # Run simulation
   schedule, total_strokes = simulate_well_kill(
       sidpp=450,
       original_mw=11.0,
       kill_mw=11.72,
       tvd=12000,
       annular_volume=350,
       pump_output=0.12
   )
   
   print("\nDriller's Method Kill Sheet")
   print("=" * 50)
   print(f"{'Strokes':>10s} {'Pump Pressure (psi)':>20s}")
   print("=" * 50)
   
   for strokes, pressure in schedule:
       print(f"{strokes:10.0f} {pressure:20.0f}")
   
   print(f"\nTotal Strokes to Complete Kill: {total_strokes:.0f}")

---

Practice Exercise 5.1: Complete Well Design
--------------------------------------------

**Scenario:**

Design a production well for the Marcellus Shale:

- Target depth: 8,500 ft TVD (12,000 ft MD - deviated well)
- Surface elevation: 1,200 ft
- Pore pressure gradient: 0.58 psi/ft
- Fracture gradient: 0.85 psi/ft
- Expected flow rate: 650 gpm

**Your Tasks:**

1. Calculate required mud weight (with 250 psi overbalance)
2. Design casing program (surface, intermediate, production)
3. Calculate hydraulics (ECD, pump pressure)
4. Verify casing burst/collapse ratings
5. Generate well control kill sheet

**Solution Template:**

.. code-block:: python

   from petrosmith import DrillingAPI
   
   api = DrillingAPI()
   
   # Your design here
   # [Complete solution in appendix]

---

Module Summary
--------------

**What You Learned:**

✓ Pressure calculations (hydrostatic, ECD, pore, fracture)  
✓ Hydraulics optimization (flow rate, pump pressure, hole cleaning)  
✓ Casing design (burst, collapse, tension)  
✓ Well control (kick detection, kill procedures)  
✓ Safety margins and industry standards  

**PetroSmith Skills:**

- ``DrillingAPI`` for calculations
- ``DrillingService`` for complex analysis
- ``PhysicalConstants`` for validated formulas
- Real-time monitoring simulations

**Real-World Applications:**

- Pre-drill well design and AFE estimation
- Real-time drilling optimization
- Well control emergency response
- Post-well analysis and lessons learned

**Next Module:**

:doc:`module6_production_engineering` - Artificial lift, well testing, and optimization

---

.. note::
   **Safety First**
   
   Drilling is inherently dangerous. Always:
   
   - Follow company HSE policies
   - Verify calculations with senior engineers
   - Never compromise safety for cost or time
   - "When in doubt, stop and think it out"
   
   **Lives depend on your engineering decisions.**

**🎯 Continue Learning:**

Next: :doc:`module6_production_engineering`
