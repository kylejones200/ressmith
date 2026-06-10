Module 10: Field Development Planning
======================================

.. meta::
   :description: Comprehensive field development planning from discovery to abandonment
   :keywords: field development, reservoir management, economic evaluation, project management

**Learning Objectives**

After completing this module, you will be able to:

- Design comprehensive field development plans
- Evaluate development concepts and select optimal strategy
- Perform economic evaluation with NPV/IRR analysis
- Plan drilling campaigns and facility requirements
- Design production profiles and forecast reserves
- Assess and mitigate project risks
- Create integrated subsurface-surface models
- Manage reservoir performance throughout field life
- Plan abandonment and decommissioning

**Time Commitment:** 8-10 hours

**Prerequisites:** All previous modules

----

Introduction: From Discovery to First Oil
------------------------------------------

**The Field Development Journey:**

.. code-block:: text

   DISCOVERY
      ↓
   APPRAISAL (2-3 years)
   • Delineation wells
   • Reservoir characterization
   • Resource estimation
      ↓
   CONCEPT SELECT (1-2 years)
   • Development concepts
   • Facility design
   • Economics
   • FID (Final Investment Decision)
      ↓
   EXECUTION (3-5 years)
   • Detailed engineering
   • Drilling campaign
   • Facilities construction
      ↓
   PRODUCTION (20-40 years)
   • Ramp-up
   • Plateau
   • Decline
   • Optimization
      ↓
   ABANDONMENT (2-5 years)
   • P&A wells
   • Remove facilities
   • Site restoration

**Typical Timeline:** 5-10 years from discovery to production

**Typical Costs:**

- Appraisal: $50-200M
- Development: $500M-10B+
- Operating: $15-40/bbl
- Abandonment: $100-500M

**This Module's Focus:**

We'll build a **complete field development plan** including:

1. Reservoir assessment
2. Development concept selection
3. Drilling and completions plan
4. Facilities design
5. Production forecast
6. Economic evaluation
7. Risk assessment
8. Execution plan

----

Part 1: Reservoir Assessment and Characterization
--------------------------------------------------

1.1 Resource Estimation
^^^^^^^^^^^^^^^^^^^^^^^^

**Deterministic vs. Probabilistic:**

.. code-block:: python

   import numpy as np
   from scipy import stats
   import matplotlib.pyplot as plt
   
   class ReservoirResourceEstimation:
       """
       Resource estimation using probabilistic methods.
       """
       
       def __init__(self, field_name):
           self.field_name = field_name
       
       def deterministic_ooip(self, area, thickness, porosity, So, Bo):
           """Simple deterministic OOIP"""
           return 7758 * area * thickness * porosity * So / Bo
       
       def probabilistic_ooip(self, 
                             area_dist,
                             thickness_dist, 
                             porosity_dist,
                             so_dist,
                             bo_value=1.2,
                             n_simulations=10000):
           """
           Monte Carlo simulation for OOIP uncertainty.
           
           Args:
               area_dist: (P10, P50, P90) in acres
               thickness_dist: (P10, P50, P90) in ft
               porosity_dist: (P10, P50, P90) fraction
               so_dist: (P10, P50, P90) fraction
               bo_value: Formation volume factor
               n_simulations: Number of Monte Carlo iterations
           
           Returns:
               Dictionary with P10, P50, P90 OOIP values
           """
           
           # Create distributions
           # Using triangular distribution (simple but effective)
           
           area_samples = np.random.triangular(
               area_dist[2], area_dist[1], area_dist[0], n_simulations
           )
           
           thickness_samples = np.random.triangular(
               thickness_dist[2], thickness_dist[1], thickness_dist[0], n_simulations
           )
           
           porosity_samples = np.random.triangular(
               porosity_dist[2], porosity_dist[1], porosity_dist[0], n_simulations
           )
           
           so_samples = np.random.triangular(
               so_dist[2], so_dist[1], so_dist[0], n_simulations
           )
           
           # Calculate OOIP for each simulation
           ooip_samples = (7758 * area_samples * thickness_samples * 
                          porosity_samples * so_samples / bo_value)
           
           # Calculate statistics
           p10 = np.percentile(ooip_samples, 90)  # High estimate
           p50 = np.percentile(ooip_samples, 50)  # Best estimate
           p90 = np.percentile(ooip_samples, 10)  # Low estimate
           mean = np.mean(ooip_samples)
           std = np.std(ooip_samples)
           
           return {
               'p10_ooip_mmstb': p10 / 1e6,
               'p50_ooip_mmstb': p50 / 1e6,
               'p90_ooip_mmstb': p90 / 1e6,
               'mean_ooip_mmstb': mean / 1e6,
               'std_ooip_mmstb': std / 1e6,
               'samples': ooip_samples
           }
       
       def calculate_recoverable_reserves(self, ooip_dist, recovery_factor_dist):
           """
           Calculate recoverable reserves from OOIP.
           
           Args:
               ooip_dist: (P10, P50, P90) OOIP in MMSTB
               recovery_factor_dist: (P10, P50, P90) as fraction
           """
           
           # Recoverable = OOIP * RF
           p10_reserves = ooip_dist[0] * recovery_factor_dist[0]
           p50_reserves = ooip_dist[1] * recovery_factor_dist[1]
           p90_reserves = ooip_dist[2] * recovery_factor_dist[2]
           
           return {
               '1P_reserves_mmstb': p90_reserves,  # Proven
               '2P_reserves_mmstb': p50_reserves,  # Proven + Probable
               '3P_reserves_mmstb': p10_reserves,  # Proven + Probable + Possible
               'best_estimate_mmstb': p50_reserves
           }
   
   # Example: Resource estimation
   estimator = ReservoirResourceEstimation('North Field')
   
   print("\n" + "="*70)
   print("         RESERVOIR RESOURCE ESTIMATION")
   print("="*70)
   
   # Input distributions (P10, P50, P90)
   area_dist = (1200, 800, 600)       # acres
   thickness_dist = (80, 60, 40)       # ft
   porosity_dist = (0.25, 0.20, 0.15) # fraction
   so_dist = (0.80, 0.70, 0.60)       # fraction
   
   # Run probabilistic analysis
   ooip_result = estimator.probabilistic_ooip(
       area_dist, thickness_dist, porosity_dist, so_dist
   )
   
   print("\nOOIP Estimates (Monte Carlo - 10,000 simulations):")
   print(f"  P90 (Low):  {ooip_result['p90_ooip_mmstb']:.1f} MMSTB")
   print(f"  P50 (Best): {ooip_result['p50_ooip_mmstb']:.1f} MMSTB")
   print(f"  P10 (High): {ooip_result['p10_ooip_mmstb']:.1f} MMSTB")
   print(f"  Mean:       {ooip_result['mean_ooip_mmstb']:.1f} MMSTB")
   print(f"  Std Dev:    {ooip_result['std_ooip_mmstb']:.1f} MMSTB")
   
   # Calculate recoverable reserves
   ooip_values = (
       ooip_result['p10_ooip_mmstb'],
       ooip_result['p50_ooip_mmstb'],
       ooip_result['p90_ooip_mmstb']
   )
   recovery_factors = (0.35, 0.30, 0.25)  # P10, P50, P90
   
   reserves = estimator.calculate_recoverable_reserves(ooip_values, recovery_factors)
   
   print("\nRecoverable Reserves:")
   print(f"  1P (Proven):              {reserves['1P_reserves_mmstb']:.1f} MMSTB")
   print(f"  2P (Proven + Probable):   {reserves['2P_reserves_mmstb']:.1f} MMSTB")
   print(f"  3P (Proven + Prob + Poss):{reserves['3P_reserves_mmstb']:.1f} MMSTB")

1.2 Development Concept Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Concept Options:**

.. code-block:: python

   class DevelopmentConceptEvaluator:
       """
       Evaluate and compare development concepts.
       """
       
       def __init__(self, field_name, reserves_mmstb):
           self.field_name = field_name
           self.reserves = reserves_mmstb
       
       def evaluate_concept(self, concept_params):
           """
           Evaluate a development concept.
           
           Args:
               concept_params: Dict with concept parameters
           
           Returns:
               Economic and technical metrics
           """
           
           concept = concept_params['concept']
           n_wells = concept_params['n_wells']
           well_cost_mm = concept_params['well_cost_mm']
           facility_cost_mm = concept_params['facility_cost_mm']
           opex_per_bbl = concept_params['opex_per_bbl']
           plateau_rate_mbpd = concept_params['plateau_rate_mbpd']
           plateau_years = concept_params['plateau_years']
           
           # Production profile
           profile = self.generate_production_profile(
               reserves_mmstb=self.reserves,
               plateau_rate_mbpd=plateau_rate_mbpd,
               plateau_years=plateau_years
           )
           
           # Economics
           oil_price = 70  # $/bbl
           capex = n_wells * well_cost_mm + facility_cost_mm
           
           # Calculate NPV
           discount_rate = 0.10
           npv = -capex * 1e6  # Initial investment
           
           cumulative_prod = 0
           for year, prod_mbbl in enumerate(profile['annual_production_mbbl'], 1):
               revenue = prod_mbbl * 1000 * oil_price
               opex = prod_mbbl * 1000 * opex_per_bbl
               cashflow = revenue - opex
               
               npv += cashflow / (1 + discount_rate) ** year
               cumulative_prod += prod_mbbl
           
           # Calculate IRR (simplified)
           total_revenue = sum(profile['annual_production_mbbl']) * 1000 * oil_price
           total_opex = sum(profile['annual_production_mbbl']) * 1000 * opex_per_bbl
           total_cashflow = total_revenue - total_opex
           
           field_life = len(profile['annual_production_mbbl'])
           
           return {
               'concept': concept,
               'npv_mm': npv / 1e6,
               'capex_mm': capex,
               'field_life_years': field_life,
               'peak_rate_mbpd': plateau_rate_mbpd,
               'cumulative_production_mmstb': cumulative_prod / 1000,
               'recovery_factor': cumulative_prod / (self.reserves * 1000),
               'production_profile': profile
           }
       
       def generate_production_profile(self, reserves_mmstb, plateau_rate_mbpd, plateau_years):
           """
           Generate production profile.
           
           Profile shape:
           - Ramp-up (1-2 years)
           - Plateau
           - Exponential decline
           """
           
           profile = []
           annual_prod = []
           cumulative = 0
           year = 0
           
           # Ramp-up (2 years to plateau)
           rampup_years = 2
           for y in range(rampup_years):
               rate = plateau_rate_mbpd * (y + 1) / rampup_years
               prod = rate * 365 / 1000  # MBBL
               profile.append((year + y, rate, prod))
               annual_prod.append(prod)
               cumulative += prod
               year += 1
           
           # Plateau
           for y in range(plateau_years):
               prod = plateau_rate_mbpd * 365 / 1000  # MBBL
               profile.append((year, plateau_rate_mbpd, prod))
               annual_prod.append(prod)
               cumulative += prod
               year += 1
           
           # Decline phase (15% annual decline)
           current_rate = plateau_rate_mbpd
           decline_rate = 0.15
           
           while current_rate > 10 and cumulative / 1000 < reserves_mmstb * 0.95:
               current_rate *= (1 - decline_rate)
               prod = current_rate * 365 / 1000  # MBBL
               profile.append((year, current_rate, prod))
               annual_prod.append(prod)
               cumulative += prod
               year += 1
               
               if year > 40:  # Max field life
                   break
           
           return {
               'annual_production_mbbl': annual_prod,
               'profile': profile,
               'field_life_years': year
           }
       
       def compare_concepts(self, concepts):
           """Compare multiple development concepts"""
           
           results = []
           
           for concept in concepts:
               result = self.evaluate_concept(concept)
               results.append(result)
           
           # Sort by NPV
           results.sort(key=lambda x: x['npv_mm'], reverse=True)
           
           return results
   
   # Example: Concept selection
   evaluator = DevelopmentConceptEvaluator(
       field_name='North Field',
       reserves_mmstb=60  # 2P reserves
   )
   
   # Define concepts
   concepts = [
       {
           'concept': 'Low Cost (Vertical Wells)',
           'n_wells': 15,
           'well_cost_mm': 4,
           'facility_cost_mm': 200,
           'opex_per_bbl': 18,
           'plateau_rate_mbpd': 15,
           'plateau_years': 5
       },
       {
           'concept': 'Base Case (Horizontal Wells)',
           'n_wells': 10,
           'well_cost_mm': 8,
           'facility_cost_mm': 300,
           'opex_per_bbl': 15,
           'plateau_rate_mbpd': 20,
           'plateau_years': 6
       },
       {
           'concept': 'Accelerated (More Wells)',
           'n_wells': 20,
           'well_cost_mm': 8,
           'facility_cost_mm': 400,
           'opex_per_bbl': 15,
           'plateau_rate_mbpd': 30,
           'plateau_years': 4
       }
   ]
   
   results = evaluator.compare_concepts(concepts)
   
   print("\n" + "="*80)
   print("              DEVELOPMENT CONCEPT COMPARISON")
   print("="*80)
   
   for i, result in enumerate(results, 1):
       print(f"\n{i}. {result['concept']}")
       print(f"   NPV: ${result['npv_mm']:.0f} MM")
       print(f"   Capex: ${result['capex_mm']:.0f} MM")
       print(f"   Peak Rate: {result['peak_rate_mbpd']:.0f} MBPD")
       print(f"   Field Life: {result['field_life_years']} years")
       print(f"   Cumulative Production: {result['cumulative_production_mmstb']:.1f} MMSTB")
       print(f"   Recovery Factor: {result['recovery_factor']*100:.1f}%")
   
   print(f"\n{'✅ RECOMMENDED CONCEPT:':<25} {results[0]['concept']}")

----

Part 2: Drilling and Facilities Planning
-----------------------------------------

2.1 Drilling Campaign
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   class DrillingCampaignPlanner:
       """
       Plan and optimize drilling campaign.
       """
       
       def __init__(self, n_wells, rig_cost_per_day, days_per_well):
           self.n_wells = n_wells
           self.rig_cost_per_day = rig_cost_per_day
           self.days_per_well = days_per_well
       
       def plan_sequential_drilling(self):
           """Plan sequential drilling (one rig)"""
           
           total_days = self.n_wells * self.days_per_well
           total_cost = total_days * self.rig_cost_per_day
           
           return {
               'strategy': 'Sequential (1 rig)',
               'total_days': total_days,
               'total_months': total_days / 30,
               'total_cost_mm': total_cost / 1e6,
               'rigs_required': 1
           }
       
       def plan_parallel_drilling(self, n_rigs):
           """Plan parallel drilling (multiple rigs)"""
           
           wells_per_rig = np.ceil(self.n_wells / n_rigs)
           total_days = wells_per_rig * self.days_per_well
           total_cost = total_days * self.rig_cost_per_day * n_rigs
           
           return {
               'strategy': f'Parallel ({n_rigs} rigs)',
               'total_days': total_days,
               'total_months': total_days / 30,
               'total_cost_mm': total_cost / 1e6,
               'rigs_required': n_rigs
           }
       
       def optimize_rig_count(self, max_rigs=5):
           """Optimize number of rigs (time vs cost)"""
           
           results = []
           
           for n_rigs in range(1, min(max_rigs, self.n_wells) + 1):
               result = self.plan_parallel_drilling(n_rigs)
               results.append(result)
           
           return results
   
   # Example: Drilling campaign
   planner = DrillingCampaignPlanner(
       n_wells=15,
       rig_cost_per_day=150000,  # $150k/day
       days_per_well=60  # 60 days per well
   )
   
   print("\n" + "="*70)
   print("           DRILLING CAMPAIGN OPTIMIZATION")
   print("="*70)
   
   strategies = planner.optimize_rig_count(max_rigs=4)
   
   print(f"\n{'Strategy':<20} {'Duration':<15} {'Cost':<15} {'Rigs':<8}")
   print(f"{'':20} {'(months)':<15} {'($MM)':<15} {'(#)':<8}")
   print("-" * 70)
   
   for strat in strategies:
       print(f"{strat['strategy']:<20} "
             f"{strat['total_months']:<15.1f} "
             f"{strat['total_cost_mm']:<15.0f} "
             f"{strat['rigs_required']:<8}")
   
   print(f"\nTradeoff: More rigs = Faster but more expensive")
   print(f"Recommendation: Balance time-to-market vs capital efficiency")

2.2 Facilities Design
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   class FacilitiesDesigner:
       """
       Design production facilities.
       """
       
       def __init__(self, peak_rate_bpd, water_cut_max):
           self.peak_oil_rate = peak_rate_bpd
           self.water_cut = water_cut_max
           self.peak_total_liquid = peak_rate_bpd / (1 - water_cut_max)
       
       def size_separator(self):
           """Size three-phase separator"""
           
           # Rule of thumb: 3-5 minutes retention time
           retention_time_min = 4
           
           # Convert rate to GPM
           total_liquid_gpm = self.peak_total_liquid * 42 / 1440
           
           # Required volume (gallons)
           required_volume_gal = total_liquid_gpm * retention_time_min
           
           # Convert to barrels
           required_volume_bbl = required_volume_gal / 42
           
           # Add 20% safety factor
           design_volume_bbl = required_volume_bbl * 1.2
           
           return {
               'design_capacity_bpd': self.peak_total_liquid,
               'required_volume_bbl': required_volume_bbl,
               'design_volume_bbl': design_volume_bbl,
               'estimated_cost_mm': design_volume_bbl / 100  # Simplified
           }
       
       def size_water_treatment(self):
           """Size water treatment system"""
           
           water_rate_bpd = self.peak_total_liquid * self.water_cut
           
           # Typical costs
           cost_per_bwpd = 5000  # $5k per barrel water per day capacity
           
           return {
               'design_capacity_bwpd': water_rate_bpd,
               'estimated_cost_mm': water_rate_bpd * cost_per_bwpd / 1e6
           }
       
       def size_storage(self, days_storage=3):
           """Size crude oil storage"""
           
           storage_volume_bbl = self.peak_oil_rate * days_storage
           
           # Tank costs (simplified)
           cost_per_bbl = 500  # $500/bbl capacity
           
           return {
               'design_capacity_bbl': storage_volume_bbl,
               'days_storage': days_storage,
               'estimated_cost_mm': storage_volume_bbl * cost_per_bbl / 1e6
           }
       
       def complete_facilities_design(self):
           """Complete facilities cost estimate"""
           
           separator = self.size_separator()
           water_treatment = self.size_water_treatment()
           storage = self.size_storage()
           
           # Additional systems
           pipeline_mm = 50  # Gathering system
           power_mm = 30     # Power generation
           other_mm = 50     # Control, utilities, etc.
           
           total_cost = (
               separator['estimated_cost_mm'] +
               water_treatment['estimated_cost_mm'] +
               storage['estimated_cost_mm'] +
               pipeline_mm +
               power_mm +
               other_mm
           )
           
           return {
               'separator': separator,
               'water_treatment': water_treatment,
               'storage': storage,
               'pipeline_mm': pipeline_mm,
               'power_mm': power_mm,
               'other_mm': other_mm,
               'total_facilities_cost_mm': total_cost
           }
   
   # Example: Facilities design
   designer = FacilitiesDesigner(
       peak_rate_bpd=20000,
       water_cut_max=0.30
   )
   
   facilities = designer.complete_facilities_design()
   
   print("\n" + "="*70)
   print("           PRODUCTION FACILITIES DESIGN")
   print("="*70)
   
   print(f"\nDesign Basis:")
   print(f"  Peak Oil Rate: {designer.peak_oil_rate:,} BPD")
   print(f"  Peak Total Liquid: {designer.peak_total_liquid:,.0f} BPD")
   print(f"  Water Cut: {designer.water_cut*100:.0f}%")
   
   print(f"\nFacilities Summary:")
   print(f"  Separator: {facilities['separator']['design_volume_bbl']:.0f} bbl (${facilities['separator']['estimated_cost_mm']:.0f}M)")
   print(f"  Water Treatment: {facilities['water_treatment']['design_capacity_bwpd']:,.0f} BWPD (${facilities['water_treatment']['estimated_cost_mm']:.0f}M)")
   print(f"  Storage: {facilities['storage']['design_capacity_bbl']:,.0f} bbl (${facilities['storage']['estimated_cost_mm']:.0f}M)")
   print(f"  Pipeline & Gathering: ${facilities['pipeline_mm']:.0f}M")
   print(f"  Power Generation: ${facilities['power_mm']:.0f}M")
   print(f"  Other: ${facilities['other_mm']:.0f}M")
   
   print(f"\n{'TOTAL FACILITIES COST:':<30} ${facilities['total_facilities_cost_mm']:.0f} MM")

----

Part 3: Economic Evaluation
----------------------------

3.1 Comprehensive NPV Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   class FieldEconomics:
       """
       Comprehensive field development economics.
       """
       
       def __init__(self):
           self.oil_price = 70        # $/bbl
           self.gas_price = 3.0       # $/Mscf
           self.opex_per_bbl = 15     # $/bbl
           self.discount_rate = 0.10  # 10%
           self.royalty_rate = 0.125  # 12.5%
           self.tax_rate = 0.35       # 35%
       
       def calculate_full_cycle_economics(
           self,
           production_profile_mbbl,
           gas_profile_mmscf,
           drilling_capex_mm,
           facilities_capex_mm,
           abandonment_cost_mm
       ):
           """
           Calculate full cycle economics with taxes and royalties.
           """
           
           cashflows = []
           cumulative_npv = 0
           payout_year = None
           
           # Year 0: Initial capex
           year_0_cashflow = -(drilling_capex_mm + facilities_capex_mm) * 1e6
           cashflows.append(year_0_cashflow)
           cumulative_npv = year_0_cashflow
           
           # Production years
           for year, (oil_mbbl, gas_mmscf) in enumerate(
               zip(production_profile_mbbl, gas_profile_mmscf), 1
           ):
               # Revenue
               oil_revenue = oil_mbbl * 1000 * self.oil_price
               gas_revenue = gas_mmscf * 1000 * self.gas_price
               gross_revenue = oil_revenue + gas_revenue
               
               # Royalty
               royalty = gross_revenue * self.royalty_rate
               revenue_after_royalty = gross_revenue - royalty
               
               # Operating costs
               opex = oil_mbbl * 1000 * self.opex_per_bbl
               
               # Taxable income
               taxable_income = revenue_after_royalty - opex
               
               # Tax
               tax = max(0, taxable_income * self.tax_rate)
               
               # Net cashflow
               net_cashflow = taxable_income - tax
               
               # Discount
               pv_cashflow = net_cashflow / (1 + self.discount_rate) ** year
               cumulative_npv += pv_cashflow
               
               cashflows.append(net_cashflow)
               
               # Check payout
               if payout_year is None and cumulative_npv > 0:
                   payout_year = year
           
           # Abandonment cost (final year)
           final_year = len(cashflows)
           abandonment_pv = -abandonment_cost_mm * 1e6 / (
               (1 + self.discount_rate) ** final_year
           )
           cumulative_npv += abandonment_pv
           
           # Calculate IRR (simplified)
           total_undiscounted_cashflow = sum(cashflows[1:]) + abandonment_pv
           avg_annual_cashflow = total_undiscounted_cashflow / (final_year - 1)
           initial_investment = -cashflows[0]
           
           if avg_annual_cashflow > 0:
               irr_approx = (avg_annual_cashflow / initial_investment) * 100
           else:
               irr_approx = 0
           
           return {
               'npv_mm': cumulative_npv / 1e6,
               'irr_percent': irr_approx,
               'payout_years': payout_year,
               'total_capex_mm': (drilling_capex_mm + facilities_capex_mm),
               'field_life_years': final_year - 1,
               'cashflows': cashflows
           }
       
       def sensitivity_analysis(self, base_case):
           """Perform sensitivity analysis on key variables"""
           
           sensitivities = {}
           
           # Oil price sensitivity
           oil_prices = [50, 60, 70, 80, 90]
           oil_price_npvs = []
           
           for price in oil_prices:
               self.oil_price = price
               result = self.calculate_full_cycle_economics(**base_case)
               oil_price_npvs.append(result['npv_mm'])
           
           sensitivities['oil_price'] = {
               'prices': oil_prices,
               'npvs': oil_price_npvs
           }
           
           # Reset to base
           self.oil_price = 70
           
           # Opex sensitivity
           opex_values = [10, 12, 15, 18, 20]
           opex_npvs = []
           
           for opex in opex_values:
               self.opex_per_bbl = opex
               result = self.calculate_full_cycle_economics(**base_case)
               opex_npvs.append(result['npv_mm'])
           
           sensitivities['opex'] = {
               'values': opex_values,
               'npvs': opex_npvs
           }
           
           # Reset
           self.opex_per_bbl = 15
           
           return sensitivities
   
   # Example: Full economic evaluation
   economics = FieldEconomics()
   
   # Production profiles (from earlier concept evaluation)
   oil_profile = [10, 20, 25, 25, 25, 25, 22, 19, 16, 14, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2]  # MBBL/year
   gas_profile = [5, 10, 12, 12, 12, 12, 11, 10, 8, 7, 6, 5, 4, 4, 3, 3, 2, 2, 1, 1]  # MMscf/year
   
   result = economics.calculate_full_cycle_economics(
       production_profile_mbbl=oil_profile,
       gas_profile_mmscf=gas_profile,
       drilling_capex_mm=80,     # 10 wells @ $8MM
       facilities_capex_mm=300,
       abandonment_cost_mm=50
   )
   
   print("\n" + "="*70)
   print("           FIELD DEVELOPMENT ECONOMICS")
   print("="*70)
   
   print(f"\nEconomic Assumptions:")
   print(f"  Oil Price: ${economics.oil_price}/bbl")
   print(f"  Gas Price: ${economics.gas_price}/Mscf")
   print(f"  Opex: ${economics.opex_per_bbl}/bbl")
   print(f"  Discount Rate: {economics.discount_rate*100:.0f}%")
   print(f"  Royalty: {economics.royalty_rate*100:.1f}%")
   print(f"  Tax Rate: {economics.tax_rate*100:.0f}%")
   
   print(f"\nResults:")
   print(f"  NPV @ {economics.discount_rate*100:.0f}%: ${result['npv_mm']:.0f} MM")
   print(f"  IRR (approx): {result['irr_percent']:.1f}%")
   print(f"  Payout: {result['payout_years']} years")
   print(f"  Total Capex: ${result['total_capex_mm']:.0f} MM")
   print(f"  Field Life: {result['field_life_years']} years")
   
   if result['npv_mm'] > 0 and result['irr_percent'] > 15:
       print(f"\n✅ PROJECT ECONOMICS: ATTRACTIVE")
       print(f"   Recommend proceeding to FID")
   elif result['npv_mm'] > 0:
       print(f"\n⚠️  PROJECT ECONOMICS: MARGINAL")
       print(f"   Consider optimization or wait for better market")
   else:
       print(f"\n❌ PROJECT ECONOMICS: UNECONOMIC")
       print(f"   Not recommended")

----

Part 4: Risk Assessment and Mitigation
---------------------------------------

4.1 Risk Register
^^^^^^^^^^^^^^^^^

.. code-block:: python

   class RiskRegister:
       """
       Track and quantify project risks.
       """
       
       def __init__(self, project_name):
           self.project_name = project_name
           self.risks = []
       
       def add_risk(self, category, description, probability, impact_mm, mitigation):
           """
           Add risk to register.
           
           Args:
               category: 'Technical', 'Commercial', 'HSE', 'Regulatory'
               description: Risk description
               probability: 0-1 probability of occurrence
               impact_mm: Financial impact if occurs ($MM)
               mitigation: Mitigation strategy
           """
           
           risk_value = probability * impact_mm
           
           self.risks.append({
               'category': category,
               'description': description,
               'probability': probability,
               'impact_mm': impact_mm,
               'risk_value_mm': risk_value,
               'mitigation': mitigation
           })
       
       def calculate_total_risk_exposure(self):
           """Calculate total risk-weighted exposure"""
           total = sum(risk['risk_value_mm'] for risk in self.risks)
           return total
       
       def get_top_risks(self, n=5):
           """Get top N risks by risk value"""
           sorted_risks = sorted(
               self.risks, 
               key=lambda x: x['risk_value_mm'], 
               reverse=True
           )
           return sorted_risks[:n]
   
   # Example: Risk assessment
   risk_register = RiskRegister('North Field Development')
   
   # Add risks
   risk_register.add_risk(
       category='Technical',
       description='Reservoir uncertainty - OOIP 20% below P50',
       probability=0.20,
       impact_mm=150,
       mitigation='Additional appraisal well, pressure management'
   )
   
   risk_register.add_risk(
       category='Technical',
       description='Well productivity below forecast',
       probability=0.30,
       impact_mm=80,
       mitigation='Contingency wells in budget, stimulation program'
   )
   
   risk_register.add_risk(
       category='Commercial',
       description='Oil price drops to $50/bbl',
       probability=0.25,
       impact_mm=200,
       mitigation='Cost reduction program, hedge strategy'
   )
   
   risk_register.add_risk(
       category='Execution',
       description='Drilling campaign 30% over budget',
       probability=0.35,
       impact_mm=100,
       mitigation='Rigorous well planning, drilling optimization'
   )
   
   risk_register.add_risk(
       category='Regulatory',
       description='Delay in environmental permits',
       probability=0.15,
       impact_mm=50,
       mitigation='Early engagement, parallel permitting'
   )
   
   risk_register.add_risk(
       category='HSE',
       description='Major safety incident',
       probability=0.05,
       impact_mm=500,
       mitigation='Comprehensive HSE program, audits'
   )
   
   print("\n" + "="*80)
   print("                    PROJECT RISK REGISTER")
   print("="*80)
   
   total_exposure = risk_register.calculate_total_risk_exposure()
   print(f"\nTotal Risk-Weighted Exposure: ${total_exposure:.0f} MM")
   
   top_risks = risk_register.get_top_risks(n=5)
   
   print(f"\nTop 5 Risks:")
   print(f"{'#':<4} {'Category':<15} {'Probability':<12} {'Impact':<12} {'Risk Value':<12}")
   print(f"{'':4} {'':15} {'':12} {'($MM)':<12} {'($MM)':<12}")
   print("-" * 80)
   
   for i, risk in enumerate(top_risks, 1):
       print(f"{i:<4} {risk['category']:<15} "
             f"{risk['probability']*100:>10.0f}% "
             f"{risk['impact_mm']:>10.0f} "
             f"{risk['risk_value_mm']:>10.0f}")
       print(f"     {risk['description']}")
       print(f"     Mitigation: {risk['mitigation']}\n")

----

Summary and Key Takeaways
--------------------------

**What You Learned:**

✅ Probabilistic resource estimation  
✅ Development concept selection  
✅ Drilling campaign planning  
✅ Facilities design and sizing  
✅ Comprehensive economic evaluation  
✅ Risk assessment and mitigation  
✅ Integrated field development planning  

**Critical Success Factors:**

1. **Robust reservoir understanding** - Invest in appraisal
2. **Fit-for-purpose design** - Don't over/under-design
3. **Integrated planning** - Subsurface + surface + economics
4. **Risk management** - Identify and mitigate early
5. **Phased execution** - De-risk through stages
6. **Continuous optimization** - Adapt as you learn

**Typical Project Timeline:**

.. code-block:: text

   Discovery → Appraisal → Concept Select → FID → Execute → Produce
   Today        2-3 yrs      4-5 yrs        6 yrs  10 yrs   30+ yrs

**Economic Thresholds:**

- NPV > $0: Technically economic
- IRR > 15%: Attractive project
- Payout < 5 years: Low risk
- NPV/Capex > 0.5: Good value

**Key Documents:**

1. Field Development Plan (FDP)
2. Concept Selection Report
3. Economic Model
4. Risk Register
5. Execution Plan
6. Abandonment Plan

----

Final Project Exercise
----------------------

**Your Assignment:**

Design a complete field development plan for:

- **Reservoir:** 80 MMSTB OOIP (P50), 30% RF
- **Location:** Onshore
- **Budget:** $500MM total capex
- **Timeline:** First oil in 4 years

**Deliverables:**

1. Resource estimation (probabilistic)
2. Three development concepts
3. Recommended concept with justification
4. Drilling and facilities plan
5. Production profile (20 years)
6. Economic evaluation (NPV, IRR)
7. Risk register (top 10 risks)
8. Execution schedule

**Evaluation Criteria:**

- Technical soundness
- Economic attractiveness
- Risk management
- Integrated thinking
- Practical implementation

----

**Module Complete!** ✅

Congratulations! You've completed the field development planning module.

You now have comprehensive knowledge to:
- Lead field development projects
- Evaluate development concepts
- Perform economic analysis
- Manage project risks

**Next:** :doc:`case_studies` - Apply your knowledge to real-world scenarios!
