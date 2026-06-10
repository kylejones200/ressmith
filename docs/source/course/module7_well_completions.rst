Module 7: Well Completions
===========================

.. meta::
   :description: Complete guide to well completion design including perforating, sand control, and stimulation
   :keywords: well completions, perforating, sand control, hydraulic fracturing, gravel pack

**Learning Objectives**

After completing this module, you will be able to:

- Design optimal completion strategies for different reservoirs
- Calculate perforation parameters for maximum productivity
- Select and design sand control systems
- Design hydraulic fracturing treatments
- Evaluate acidizing operations
- Choose appropriate completion equipment
- Optimize completion costs vs. productivity
- Troubleshoot completion problems

**Time Commitment:** 6-8 hours

**Prerequisites:** Modules 1-6

----

Introduction: Well Completion Design
------------------------------------

**What is a Well Completion?**

A **well completion** is the process of preparing a drilled well for production. It includes:

- **Casing and cementing** - Wellbore integrity
- **Perforating** - Creating flow paths into reservoir
- **Sand control** - Preventing sand production
- **Stimulation** - Enhancing productivity
- **Downhole equipment** - Production tubing, packers, safety valves

**Completion Types:**

1. **Open Hole** - No casing across reservoir
2. **Cased and Perforated** - Most common
3. **Slotted Liner** - For unconsolidated sands
4. **Gravel Pack** - Sand control
5. **Frac Pack** - Fracturing + gravel pack

**Design Considerations:**

- Reservoir characteristics (permeability, lithology, pressure)
- Production objectives (rate, ultimate recovery)
- Well architecture (vertical, horizontal, multilateral)
- Sand production risk
- Formation damage
- Economics (cost vs. value)

**Completion Optimization:**

.. math::

   NPV = \\sum_{t=1}^{n} \\frac{Revenue_t - Cost_t}{(1 + discount)^t} - Completion\\ Cost

Goal: **Maximize NPV** by balancing completion cost against increased productivity and recovery.

----

Part 1: Perforating Design
---------------------------

1.1 Perforation Fundamentals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Purpose:**

Create hydraulic communication between reservoir and wellbore.

**Key Parameters:**

- **Shot density** (shots per foot)
- **Phasing** (0°, 60°, 90°, 120°, 180°)
- **Penetration depth** (inches)
- **Hole diameter** (inches)

**Perforating Methods:**

1. **Wireline** - Most common, flexible
2. **Tubing conveyed** - For deviated wells
3. **Through-tubing** - For selective re-perforation

1.2 Perforation Design Calculations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Productivity Ratio:**

The **perforation skin** reduces productivity:

.. math::

   S_{perf} = S_{crushed} + S_{angle} + S_{vertical} + S_{wellbore}

**PetroSmith Implementation:**

.. code-block:: python

   from petrosmith.models.completion import PerforationDesign
   import numpy as np
   
   def calculate_perforation_skin(
       permeability_md,
       perforation_length_inches,
       perforation_diameter_inches,
       shots_per_foot,
       phasing_degrees,
       wellbore_radius_ft,
       damaged_zone_permeability_md,
       damaged_zone_radius_inches
   ):
       """
       Calculate skin factor for perforated completion.
       
       Based on Karakas-Tariq correlation.
       """
       import math
       
       # Convert units
       rw = wellbore_radius_ft
       rp = perforation_diameter_inches / 2 / 12  # Convert to ft
       Lp = perforation_length_inches / 12  # Convert to ft
       
       # Perforation density
       shots_per_ft = shots_per_foot
       
       # Horizontal perforation skin (due to crushed zone)
       k = permeability_md
       kd = damaged_zone_permeability_md
       rd = damaged_zone_radius_inches / 12  # ft
       
       if kd > 0:
           horizontal_skin = (k / kd - 1) * math.log(rd / rp)
       else:
           horizontal_skin = 0
       
       # Vertical perforating skin (finite perforation length)
       h = 1.0  # Normalize to 1 ft of perforated interval
       
       # Wellbore damage skin
       wellbore_skin = 0  # Assume no additional damage
       
       # Angle (phasing) effect
       if phasing_degrees == 0:
           phase_factor = 1.0
       elif phasing_degrees == 180:
           phase_factor = 0.5
       elif phasing_degrees == 120:
           phase_factor = 0.65
       elif phasing_degrees == 90:
           phase_factor = 0.75
       else:
           phase_factor = 0.8
       
       # Simplified perforation skin
       perf_skin = horizontal_skin + wellbore_skin + (
           math.log(rw / (Lp * phase_factor)) / shots_per_ft
       )
       
       return {
           'total_perforation_skin': perf_skin,
           'horizontal_skin': horizontal_skin,
           'wellbore_skin': wellbore_skin,
           'productivity_ratio': 1 / (1 + perf_skin)  # Relative to open hole
       }
   
   # Example: Calculate perforation skin
   result = calculate_perforation_skin(
       permeability_md=100,
       perforation_length_inches=12,
       perforation_diameter_inches=0.5,
       shots_per_foot=4,
       phasing_degrees=60,
       wellbore_radius_ft=0.328,
       damaged_zone_permeability_md=10,
       damaged_zone_radius_inches=3
   )
   
   print("\n=== PERFORATION DESIGN ANALYSIS ===")
   print(f"Total Perforation Skin: {result['total_perforation_skin']:.2f}")
   print(f"Horizontal Skin: {result['horizontal_skin']:.2f}")
   print(f"Productivity Ratio: {result['productivity_ratio']*100:.1f}%")
   print(f"\nInterpretation:")
   if result['total_perforation_skin'] < 0:
       print("  ✅ Perforations improve productivity")
   elif result['total_perforation_skin'] < 2:
       print("  ✅ Good perforation design")
   elif result['total_perforation_skin'] < 5:
       print("  ⚠️  Moderate perforation skin - consider optimization")
   else:
       print("  ❌ High perforation skin - redesign needed")

**Optimizing Shot Density:**

.. code-block:: python

   def optimize_shot_density(
       permeability,
       wellbore_radius,
       perforation_length,
       perforation_diameter,
       phasing,
       cost_per_shot
   ):
       """
       Find optimal shot density balancing productivity vs cost.
       """
       import matplotlib.pyplot as plt
       
       shot_densities = range(1, 21)  # 1-20 SPF
       skins = []
       costs = []
       
       for spf in shot_densities:
           result = calculate_perforation_skin(
               permeability_md=permeability,
               perforation_length_inches=perforation_length,
               perforation_diameter_inches=perforation_diameter,
               shots_per_foot=spf,
               phasing_degrees=phasing,
               wellbore_radius_ft=wellbore_radius,
               damaged_zone_permeability_md=permeability * 0.1,
               damaged_zone_radius_inches=3
           )
           skins.append(result['total_perforation_skin'])
           costs.append(spf * cost_per_shot * 100)  # Assume 100 ft interval
       
       # Plot results
       fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
       
       ax1.plot(shot_densities, skins, 'b-o', linewidth=2, markersize=6)
       ax1.set_xlabel('Shot Density (SPF)', fontsize=12)
       ax1.set_ylabel('Perforation Skin', fontsize=12)
       ax1.set_title('Skin vs Shot Density', fontsize=13, fontweight='bold')
       ax1.grid(True, alpha=0.3)
       ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
       
       ax2.plot(shot_densities, costs, 'g-o', linewidth=2, markersize=6)
       ax2.set_xlabel('Shot Density (SPF)', fontsize=12)
       ax2.set_ylabel('Perforating Cost ($)', fontsize=12)
       ax2.set_title('Cost vs Shot Density', fontsize=13, fontweight='bold')
       ax2.grid(True, alpha=0.3)
       
       plt.tight_layout()
       plt.show()
       
       # Find diminishing returns point
       skin_reduction = [skins[i] - skins[i+1] for i in range(len(skins)-1)]
       optimal_spf = shot_densities[skin_reduction.index(min(skin_reduction)) + 1]
       
       print(f"\n=== OPTIMIZATION RESULTS ===")
       print(f"Optimal Shot Density: {optimal_spf} SPF")
       print(f"Skin at {optimal_spf} SPF: {skins[optimal_spf-1]:.2f}")
       print(f"Cost at {optimal_spf} SPF: ${costs[optimal_spf-1]:,.0f}")
       
       return optimal_spf
   
   # Run optimization
   optimal = optimize_shot_density(
       permeability=100,
       wellbore_radius=0.328,
       perforation_length=12,
       perforation_diameter=0.5,
       phasing=60,
       cost_per_shot=15
   )

1.3 Underbalanced vs Overbalanced Perforating
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Underbalanced Perforating (UBP):**

Wellbore pressure < reservoir pressure during perforation.

**Benefits:**
- ✅ Cleans perforations immediately
- ✅ Removes crushed zone
- ✅ Minimal skin
- ✅ Better productivity

**Challenges:**
- ❌ More complex operations
- ❌ Requires well control equipment
- ❌ Higher cost

**Calculation:**

.. code-block:: python

   def calculate_underbalance_requirements(
       reservoir_pressure_psi,
       mud_weight_ppg,
       depth_ft,
       desired_underbalance_psi
   ):
       """
       Calculate required underbalance for perforation cleanup.
       """
       
       # Current hydrostatic pressure
       hydrostatic = 0.052 * mud_weight_ppg * depth_ft
       
       # Current overbalance
       current_overbalance = hydrostatic - reservoir_pressure_psi
       
       # Required mud weight reduction
       required_mw = (reservoir_pressure_psi - desired_underbalance_psi) / (0.052 * depth_ft)
       
       # Required underbalance (absolute value)
       ub_required = desired_underbalance_psi
       
       return {
           'current_hydrostatic_psi': hydrostatic,
           'current_overbalance_psi': current_overbalance,
           'required_mud_weight_ppg': required_mw,
           'underbalance_required_psi': ub_required,
           'feasible': required_mw > 0
       }
   
   # Example
   result = calculate_underbalance_requirements(
       reservoir_pressure_psi=5200,
       mud_weight_ppg=12.0,
       depth_ft=10000,
       desired_underbalance_psi=300
   )
   
   print("\n=== UNDERBALANCED PERFORATING ===")
   print(f"Current Hydrostatic: {result['current_hydrostatic_psi']:.0f} psi")
   print(f"Reservoir Pressure: 5200 psi")
   print(f"Current Overbalance: {result['current_overbalance_psi']:.0f} psi")
   print(f"\nFor 300 psi underbalance:")
   print(f"Required Mud Weight: {result['required_mud_weight_ppg']:.2f} ppg")
   
   if result['feasible']:
       print("✅ UBP is feasible")
   else:
       print("❌ UBP not feasible - use TCP or lighter fluid")

----

Part 2: Sand Control
---------------------

2.1 Sand Production Prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Why Sand Production Occurs:**

1. **Unconsolidated formations** - Weak rock
2. **High drawdown** - Exceeds rock strength
3. **Water production** - Weakens bonds
4. **Depletion** - Reduced pore pressure

**Sand Production Consequences:**

- ❌ Equipment erosion
- ❌ Disposal costs
- ❌ Reduced production
- ❌ Wellbore collapse
- ❌ Safety hazards

**Prediction Methods:**

.. code-block:: python

   def predict_sand_production_risk(
       uniaxial_compressive_strength_psi,
       reservoir_pressure_psi,
       bottomhole_pressure_psi,
       rock_cohesion_psi,
       friction_angle_degrees,
       porosity_fraction
   ):
       """
       Predict sand production risk using rock mechanics.
       
       Based on Mohr-Coulomb failure criterion.
       """
       import math
       
       # Drawdown
       drawdown = reservoir_pressure_psi - bottomhole_pressure_psi
       
       # Critical drawdown (simplified)
       ucs = uniaxial_compressive_strength_psi
       
       # Estimate critical drawdown
       # Based on rock strength and cohesion
       phi_rad = math.radians(friction_angle_degrees)
       
       critical_drawdown = ucs * (1 - math.sin(phi_rad)) / (2 * math.cos(phi_rad))
       
       # Sand production risk
       risk_factor = drawdown / critical_drawdown
       
       # Classify risk
       if risk_factor < 0.5:
           risk_level = "LOW"
           recommendation = "No sand control required"
       elif risk_factor < 0.8:
           risk_level = "MODERATE"
           recommendation = "Consider sand control or rate management"
       elif risk_factor < 1.2:
           risk_level = "HIGH"
           recommendation = "Sand control RECOMMENDED"
       else:
           risk_level = "VERY HIGH"
           recommendation = "Sand control REQUIRED"
       
       return {
           'risk_factor': risk_factor,
           'risk_level': risk_level,
           'critical_drawdown_psi': critical_drawdown,
           'current_drawdown_psi': drawdown,
           'recommendation': recommendation,
           'max_safe_rate_fraction': min(1.0, 0.8 / risk_factor) if risk_factor > 0 else 1.0
       }
   
   # Example: Evaluate sand production risk
   risk = predict_sand_production_risk(
       uniaxial_compressive_strength_psi=1500,
       reservoir_pressure_psi=3500,
       bottomhole_pressure_psi=2000,
       rock_cohesion_psi=200,
       friction_angle_degrees=30,
       porosity_fraction=0.28
   )
   
   print("\n=== SAND PRODUCTION RISK ASSESSMENT ===")
   print(f"Current Drawdown: {risk['current_drawdown_psi']:.0f} psi")
   print(f"Critical Drawdown: {risk['critical_drawdown_psi']:.0f} psi")
   print(f"Risk Factor: {risk['risk_factor']:.2f}")
   print(f"Risk Level: {risk['risk_level']}")
   print(f"Recommendation: {risk['recommendation']}")
   print(f"Max Safe Rate: {risk['max_safe_rate_fraction']*100:.0f}% of potential")

2.2 Gravel Pack Design
^^^^^^^^^^^^^^^^^^^^^^^

**Gravel Pack Purpose:**

Create a permeable barrier that:
- ✅ Stops formation sand
- ✅ Allows fluids to flow
- ✅ Stabilizes wellbore

**Gravel Selection (Saucier Criterion):**

.. math::

   \\frac{d_{gravel}}{d_{sand}} = 6 \\text{ to } 10

Where d is median grain size (D50).

**Design Calculations:**

.. code-block:: python

   def design_gravel_pack(
       formation_sand_d50_mesh,
       formation_sand_d10_mesh,
       formation_sand_d90_mesh,
       perforated_interval_ft,
       wellbore_diameter_inches,
       screen_od_inches
   ):
       """
       Design gravel pack system.
       
       Returns:
           Optimal gravel size and volume required
       """
       import math
       
       # Convert mesh to mm (approximate)
       def mesh_to_mm(mesh):
           return 25.4 / mesh
       
       sand_d50_mm = mesh_to_mm(formation_sand_d50_mesh)
       
       # Saucier criterion: gravel/sand = 6
       gravel_d50_mm = sand_d50_mm * 6
       
       # Convert back to mesh
       gravel_mesh = 25.4 / gravel_d50_mm
       
       # Select standard gravel size
       standard_gravels = {
           '20/40': (20, 40, 0.85),  # mesh range and d50 in mm
           '12/20': (12, 20, 1.27),
           '8/12': (8, 12, 1.91),
           '6/12': (6, 12, 2.38)
       }
       
       # Find closest match
       best_match = None
       best_diff = float('inf')
       
       for name, (mesh_min, mesh_max, d50) in standard_gravels.items():
           diff = abs(d50 - gravel_d50_mm)
           if diff < best_diff:
               best_diff = diff
               best_match = name
       
       # Calculate volume required
       wellbore_radius_ft = wellbore_diameter_inches / 24
       screen_radius_ft = screen_od_inches / 24
       
       # Annular volume
       annular_volume_ft3 = math.pi * (wellbore_radius_ft**2 - screen_radius_ft**2) * perforated_interval_ft
       
       # Add 20% for packing efficiency and overflush
       gravel_volume_ft3 = annular_volume_ft3 * 1.2
       
       # Convert to sacks (1 sack = 1 ft³)
       gravel_sacks = math.ceil(gravel_volume_ft3)
       
       return {
           'recommended_gravel': best_match,
           'gravel_d50_mm': gravel_d50_mm,
           'formation_sand_d50_mm': sand_d50_mm,
           'ratio': gravel_d50_mm / sand_d50_mm,
           'gravel_volume_ft3': gravel_volume_ft3,
           'gravel_sacks': gravel_sacks,
           'annular_thickness_inches': (wellbore_radius_ft - screen_radius_ft) * 12
       }
   
   # Example: Design gravel pack
   design = design_gravel_pack(
       formation_sand_d50_mesh=200,  # Very fine sand
       formation_sand_d10_mesh=300,
       formation_sand_d90_mesh=150,
       perforated_interval_ft=100,
       wellbore_diameter_inches=8.5,
       screen_od_inches=4.5
   )
   
   print("\n=== GRAVEL PACK DESIGN ===")
   print(f"Formation Sand D50: {design['formation_sand_d50_mm']:.3f} mm")
   print(f"Recommended Gravel: {design['recommended_gravel']} mesh")
   print(f"Gravel D50: {design['gravel_d50_mm']:.2f} mm")
   print(f"Gravel/Sand Ratio: {design['ratio']:.1f}")
   print(f"Annular Thickness: {design['annular_thickness_inches']:.2f} inches")
   print(f"\n=== MATERIAL REQUIREMENTS ===")
   print(f"Gravel Volume: {design['gravel_volume_ft3']:.1f} ft³")
   print(f"Gravel Sacks: {design['gravel_sacks']} sacks")
   print(f"Estimated Cost: ${design['gravel_sacks'] * 25:,.0f}")

2.3 Screen Selection
^^^^^^^^^^^^^^^^^^^^

**Screen Types:**

1. **Wire-wrapped screen** - Most common, high strength
2. **Premium screen** - Better sand control
3. **Expandable screen** - For open hole
4. **Slotted liner** - Low cost option

**Selection Criteria:**

.. code-block:: python

   def select_screen_type(
       formation_strength,
       sand_production_risk,
       well_deviation_degrees,
       budget_level
   ):
       """
       Select appropriate screen type based on well conditions.
       """
       
       recommendations = []
       
       # Analyze conditions
       if sand_production_risk == "LOW":
           recommendations.append({
               'type': 'Slotted Liner',
               'cost_relative': 1.0,
               'applicability': 'Good for low risk, budget conscious'
           })
       
       if sand_production_risk in ["MODERATE", "HIGH"]:
           recommendations.append({
               'type': 'Wire-Wrapped Screen',
               'cost_relative': 2.5,
               'applicability': 'Industry standard, good performance'
           })
           
           recommendations.append({
               'type': 'Premium Screen (Mesh)',
               'cost_relative': 4.0,
               'applicability': 'Better sand control, higher strength'
           })
       
       if formation_strength < 2000 and well_deviation_degrees < 30:
           recommendations.append({
               'type': 'Expandable Screen',
               'cost_relative': 5.0,
               'applicability': 'Open hole, weak formations'
           })
       
       return recommendations
   
   # Example
   screens = select_screen_type(
       formation_strength=1500,
       sand_production_risk="HIGH",
       well_deviation_degrees=15,
       budget_level="medium"
   )
   
   print("\n=== SCREEN RECOMMENDATIONS ===")
   for screen in screens:
       print(f"\n{screen['type']}:")
       print(f"  Relative Cost: {screen['cost_relative']:.1f}x")
       print(f"  {screen['applicability']}")

----

Part 3: Hydraulic Fracturing
-----------------------------

3.1 Fracturing Fundamentals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Purpose:**

Create high-conductivity pathways in low-permeability reservoirs.

**When to Fracture:**

✅ Tight formations (k < 1 md)  
✅ Shale/unconventional reservoirs  
✅ Damage bypass  
✅ Permeability enhancement  

**Fracture Geometry:**

- **Length (xf)** - Half-length from wellbore
- **Width (w)** - Fracture opening
- **Height (hf)** - Vertical extent
- **Conductivity (kf·w)** - Flow capacity

3.2 Fracture Design
^^^^^^^^^^^^^^^^^^^^

**Dimensionless Fracture Conductivity:**

.. math::

   F_{CD} = \\frac{k_f \\cdot w}{k \\cdot x_f}

Where:
- FCD > 30: Infinite conductivity fracture
- FCD = 1-30: Finite conductivity
- FCD < 1: Poor frac design

**Design Calculations:**

.. code-block:: python

   def design_hydraulic_fracture(
       reservoir_permeability_md,
       reservoir_thickness_ft,
       desired_fracture_half_length_ft,
       proppant_permeability_md,
       proppant_concentration_lb_gal,
       fluid_efficiency_fraction
   ):
       """
       Design hydraulic fracture treatment.
       
       Returns:
           Proppant amount, fluid volume, expected productivity increase
       """
       import math
       
       k = reservoir_permeability_md
       h = reservoir_thickness_ft
       xf = desired_fracture_half_length_ft
       kf = proppant_permeability_md
       
       # Target fracture width (empirical)
       target_width_ft = 0.02  # 0.24 inches typical
       
       # Calculate required proppant volume
       frac_area_ft2 = 2 * xf * h  # Both wings
       proppant_volume_ft3 = frac_area_ft2 * target_width_ft
       
       # Convert to pounds (proppant density ~100 lb/ft³)
       proppant_density = 100  # lb/ft³
       proppant_lb = proppant_volume_ft3 * proppant_density
       
       # Calculate fluid volume
       # Concentration in lb/gal, total proppant in lb
       fluid_volume_gal = proppant_lb / proppant_concentration_lb_gal
       
       # Account for fluid efficiency
       total_fluid_gal = fluid_volume_gal / fluid_efficiency_fraction
       
       # Dimensionless fracture conductivity
       avg_width_ft = target_width_ft
       fcd = (kf * avg_width_ft) / (k * xf)
       
       # Productivity increase (Cinco-Ley-Samaniego)
       # For FCD > 30 (infinite conductivity)
       if fcd > 30:
           productivity_ratio = xf / (0.472 * 1000)  # Assuming re = 1000 ft
       else:
           # Finite conductivity correction
           productivity_ratio = (xf / (0.472 * 1000)) * (fcd / (fcd + 1.5))
       
       return {
           'proppant_required_lb': proppant_lb,
           'proppant_required_tons': proppant_lb / 2000,
           'clean_fluid_volume_gal': fluid_volume_gal,
           'total_fluid_volume_gal': total_fluid_gal,
           'total_fluid_volume_bbl': total_fluid_gal / 42,
           'fracture_half_length_ft': xf,
           'fracture_width_inches': target_width_ft * 12,
           'fracture_height_ft': h,
           'dimensionless_conductivity': fcd,
           'productivity_increase_ratio': productivity_ratio,
           'estimated_cost': proppant_lb / 2000 * 150 + total_fluid_gal / 42 * 200
       }
   
   # Example: Design fracture treatment
   frac_design = design_hydraulic_fracture(
       reservoir_permeability_md=0.1,  # Tight gas
       reservoir_thickness_ft=100,
       desired_fracture_half_length_ft=500,
       proppant_permeability_md=50000,  # 20/40 mesh sand
       proppant_concentration_lb_gal=2.0,
       fluid_efficiency_fraction=0.50
   )
   
   print("\n=== HYDRAULIC FRACTURE DESIGN ===")
   print(f"\n--- Fracture Geometry ---")
   print(f"Half-Length: {frac_design['fracture_half_length_ft']:.0f} ft")
   print(f"Width: {frac_design['fracture_width_inches']:.2f} inches")
   print(f"Height: {frac_design['fracture_height_ft']:.0f} ft")
   print(f"FCD: {frac_design['dimensionless_conductivity']:.1f}")
   
   print(f"\n--- Treatment Size ---")
   print(f"Proppant: {frac_design['proppant_required_tons']:.0f} tons ({frac_design['proppant_required_lb']:,.0f} lb)")
   print(f"Total Fluid: {frac_design['total_fluid_volume_bbl']:.0f} bbls ({frac_design['total_fluid_volume_gal']:,.0f} gal)")
   
   print(f"\n--- Expected Results ---")
   print(f"Productivity Increase: {frac_design['productivity_increase_ratio']:.1f}x")
   print(f"Estimated Cost: ${frac_design['estimated_cost']:,.0f}")
   
   if frac_design['dimensionless_conductivity'] > 30:
       print("\n✅ Infinite conductivity fracture - Excellent design")
   elif frac_design['dimensionless_conductivity'] > 5:
       print("\n✅ Good fracture conductivity")
   else:
       print("\n⚠️  Low conductivity - consider more proppant or better quality")

3.3 Proppant Selection
^^^^^^^^^^^^^^^^^^^^^^^

**Proppant Types:**

.. code-block:: python

   PROPPANT_DATABASE = {
       'White Sand (20/40)': {
           'permeability_md': 50000,
           'density_sg': 2.65,
           'crush_strength_psi': 4000,
           'cost_per_ton': 50
       },
       'Northern White Sand (20/40)': {
           'permeability_md': 80000,
           'density_sg': 2.65,
           'crush_strength_psi': 6000,
           'cost_per_ton': 100
       },
       'Intermediate Strength Ceramic': {
           'permeability_md': 120000,
           'density_sg': 3.2,
           'crush_strength_psi': 10000,
           'cost_per_ton': 300
       },
       'High Strength Ceramic': {
           'permeability_md': 100000,
           'density_sg': 3.5,
           'crush_strength_psi': 15000,
           'cost_per_ton': 600
       }
   }
   
   def select_proppant(
       closure_stress_psi,
       fracture_half_length_ft,
       budget_per_ton
   ):
       """Select appropriate proppant based on stress and budget"""
       
       suitable_proppants = []
       
       for name, props in PROPPANT_DATABASE.items():
           if props['crush_strength_psi'] > closure_stress_psi * 1.5:  # 1.5 safety factor
               if props['cost_per_ton'] <= budget_per_ton:
                   suitable_proppants.append({
                       'name': name,
                       'permeability': props['permeability_md'],
                       'cost': props['cost_per_ton'],
                       'strength': props['crush_strength_psi']
                   })
       
       # Sort by permeability (best first)
       suitable_proppants.sort(key=lambda x: x['permeability'], reverse=True)
       
       return suitable_proppants
   
   # Example
   proppants = select_proppant(
       closure_stress_psi=5000,
       fracture_half_length_ft=500,
       budget_per_ton=200
   )
   
   print("\n=== PROPPANT RECOMMENDATIONS ===")
   for i, prop in enumerate(proppants, 1):
       print(f"\n{i}. {prop['name']}")
       print(f"   Permeability: {prop['permeability']:,} md")
       print(f"   Strength: {prop['strength']:,} psi")
       print(f"   Cost: ${prop['cost']}/ton")

----

Part 4: Matrix Acidizing
-------------------------

4.1 Acidizing Fundamentals
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Purpose:**

- Remove near-wellbore damage
- Enhance natural permeability
- Clean perforations
- Dissolve carbonate/scale

**Acid Types:**

1. **HCl (Hydrochloric)** - Carbonate reservoirs
2. **HF (Hydrofluoric)** - Sandstone reservoirs
3. **Organic acids** - High temperature
4. **Chelating agents** - Iron control

**When to Acidize:**

✅ Formation damage (drilling, completion)  
✅ Scale buildup  
✅ Reduced injectivity/productivity  
✅ Carbonate reservoirs  

4.2 Acid Volume Calculations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Sandstone Acidizing:**

.. code-block:: python

   def design_sandstone_acid_treatment(
       perforated_interval_ft,
       wellbore_radius_ft,
       damaged_zone_radius_ft,
       porosity_fraction,
       acid_concentration_percent,
       overflush_ratio
   ):
       """
       Design sandstone (HF) acid treatment.
       
       Typical: 3% HF + 12% HCl mixture
       """
       import math
       
       # Volume of damaged zone
       h = perforated_interval_ft
       rw = wellbore_radius_ft
       rd = damaged_zone_radius_ft
       phi = porosity_fraction
       
       # Pore volume in damaged zone
       pore_volume_ft3 = math.pi * (rd**2 - rw**2) * h * phi
       
       # Convert to gallons
       pore_volume_gal = pore_volume_ft3 * 7.48
       
       # Acid volume = 2-3 pore volumes
       acid_volume_gal = pore_volume_gal * 2.5
       
       # Add overflush (15% HCl or diesel)
       overflush_volume_gal = acid_volume_gal * overflush_ratio
       
       # Total fluid
       total_fluid_gal = acid_volume_gal + overflush_volume_gal
       
       # Convert to barrels
       total_fluid_bbl = total_fluid_gal / 42
       
       return {
           'damaged_zone_pore_volume_gal': pore_volume_gal,
           'acid_volume_gal': acid_volume_gal,
           'acid_volume_bbl': acid_volume_gal / 42,
           'overflush_volume_gal': overflush_volume_gal,
           'total_fluid_volume_gal': total_fluid_gal,
           'total_fluid_volume_bbl': total_fluid_bbl,
           'acid_concentration': f"{acid_concentration_percent}%",
           'estimated_cost': total_fluid_bbl * 150
       }
   
   # Example: Design sandstone acid job
   acid_design = design_sandstone_acid_treatment(
       perforated_interval_ft=50,
       wellbore_radius_ft=0.35,
       damaged_zone_radius_ft=3.0,  # 3 ft damaged zone
       porosity_fraction=0.20,
       acid_concentration_percent=3.0,  # HF
       overflush_ratio=0.5
   )
   
   print("\n=== SANDSTONE ACID TREATMENT DESIGN ===")
   print(f"Damaged Zone Pore Volume: {acid_design['damaged_zone_pore_volume_gal']:.0f} gal")
   print(f"\n--- Treatment Volumes ---")
   print(f"Acid Stage: {acid_design['acid_volume_bbl']:.0f} bbls (3% HF + 12% HCl)")
   print(f"Overflush: {acid_design['overflush_volume_gal']:.0f} gal (15% HCl)")
   print(f"Total Fluid: {acid_design['total_fluid_volume_bbl']:.0f} bbls")
   print(f"\n--- Economics ---")
   print(f"Estimated Cost: ${acid_design['estimated_cost']:,.0f}")

**Carbonate Acidizing:**

.. code-block:: python

   def design_carbonate_acid_treatment(
       perforated_interval_ft,
       acid_penetration_target_ft,
       acid_concentration_percent
   ):
       """
       Design carbonate (HCl) acid treatment.
       
       Creates wormholes rather than uniform dissolution.
       """
       
       # Empirical: 50-100 gal/ft for effective wormholing
       acid_per_foot = 75  # gal/ft
       
       # Total acid volume
       acid_volume_gal = acid_per_foot * perforated_interval_ft
       acid_volume_bbl = acid_volume_gal / 42
       
       # Wormhole penetration (empirical correlation)
       # Depends on injection rate, acid strength, temperature
       expected_penetration_ft = acid_penetration_target_ft * 0.8  # Conservative
       
       return {
           'acid_volume_gal': acid_volume_gal,
           'acid_volume_bbl': acid_volume_bbl,
           'acid_concentration': f"{acid_concentration_percent}% HCl",
           'acid_per_foot': acid_per_foot,
           'expected_penetration_ft': expected_penetration_ft,
           'estimated_cost': acid_volume_bbl * 100
       }
   
   # Example
   carb_acid = design_carbonate_acid_treatment(
       perforated_interval_ft=100,
       acid_penetration_target_ft=10,
       acid_concentration_percent=15
   )
   
   print("\n=== CARBONATE ACID TREATMENT ===")
   print(f"Total Acid: {carb_acid['acid_volume_bbl']:.0f} bbls of {carb_acid['acid_concentration']}")
   print(f"Rate: {carb_acid['acid_per_foot']:.0f} gal/ft")
   print(f"Expected Penetration: {carb_acid['expected_penetration_ft']:.0f} ft")
   print(f"Cost: ${carb_acid['estimated_cost']:,.0f}")

----

Summary and Key Takeaways
--------------------------

**What You Learned:**

✅ Perforation design and optimization  
✅ Sand control methods and design  
✅ Gravel pack and screen selection  
✅ Hydraulic fracture design  
✅ Proppant selection  
✅ Matrix acidizing (sandstone and carbonate)  
✅ Completion economics  

**Completion Selection Matrix:**

.. code-block:: text

   Formation Type    Sand Risk    Perm (md)    Recommended Completion
   ─────────────────────────────────────────────────────────────────────
   Consolidated      Low          >10          Cased & perforated
   Consolidated      Low          <1           Cased & perforated + frac
   Unconsolidated    High         >100         Gravel pack
   Unconsolidated    High         <10          Frac pack
   Carbonate         Low          >1           Cased & perforated + acid
   Shale/Tight       Low          <0.1         Multistage frac

**Design Workflow:**

1. **Characterize reservoir** (geology, rock mechanics)
2. **Assess risks** (sand production, damage)
3. **Select completion type**
4. **Design specifics** (perfs, sand control, stimulation)
5. **Optimize economics** (cost vs. productivity)
6. **Execute and monitor**

**Cost vs. Value:**

- Simple completion: $500K-$1M
- Gravel pack: +$200-500K
- Frac pack: +$500K-$1M
- Multistage frac (horizontal): $3-8M

**Must balance:**
- Upfront cost
- Expected production increase
- Well longevity
- Operating costs

----

Practice Exercises
------------------

**Exercise 7.1:** Perforation Design

Design perforations for:
- k = 50 md
- 4 SPF vs 8 SPF comparison
- Calculate skin and cost

**Exercise 7.2:** Sand Control Selection

Given:
- UCS = 2000 psi
- Drawdown = 1200 psi
- Do you need sand control?

**Exercise 7.3:** Fracture Design

Design frac for:
- k = 0.05 md (tight gas)
- h = 80 ft
- Target xf = 400 ft
- Calculate proppant, fluid, cost

----

**Module Complete!** ✅

You now have comprehensive knowledge of well completion design.

**Next:** :doc:`module8_integrated_workflows`
