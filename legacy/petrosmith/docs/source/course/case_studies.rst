Case Studies: Real-World Applications
======================================

.. meta::
   :description: Real-world petroleum engineering case studies demonstrating integrated workflows
   :keywords: case studies, real-world examples, petroleum engineering, field development

**Learning Objectives**

After completing these case studies, you will be able to:

- Apply integrated petroleum engineering workflows to real problems
- Analyze complex field development scenarios
- Make data-driven decisions under uncertainty
- Troubleshoot production problems systematically
- Optimize operations for maximum value
- Learn from successes and failures
- Develop practical engineering judgment

**Time Commitment:** 6-8 hours

**Prerequisites:** All previous modules

----

Introduction: Learning from Real-World Examples
------------------------------------------------

**Why Case Studies Matter:**

- **Bridge theory and practice** - See how concepts apply
- **Learn from others' experiences** - Both successes and failures
- **Develop judgment** - No textbook has all the answers
- **Build confidence** - "I can solve this"

**Case Study Approach:**

Each case study follows this structure:

1. **Background** - Field description and context
2. **Challenge** - The problem to solve
3. **Analysis** - Data, calculations, options
4. **Solution** - Recommended approach
5. **Results** - What actually happened
6. **Lessons Learned** - Key takeaways

----

Case Study 1: West Mesa Field Optimization
-------------------------------------------

1.1 Background
^^^^^^^^^^^^^^

**Field Overview:**

- **Location:** West Texas, USA
- **Discovery:** 1985
- **Reservoir:** Permian sandstone, 8,500 ft depth
- **Original OOIP:** 120 MMSTB
- **Initial Development:** 25 vertical wells, natural flow
- **Current Status:** Mature field, 40% water cut, declining production

**Production History:**

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   
   # Historical production data
   years = np.array([1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2024])
   oil_rate_bpd = np.array([8000, 12000, 15000, 14000, 12000, 9000, 7000, 5500, 4800])
   water_cut = np.array([5, 10, 15, 25, 35, 45, 50, 48, 45])
   
   print("\n" + "="*70)
   print("        CASE STUDY 1: WEST MESA FIELD OPTIMIZATION")
   print("="*70)
   
   print(f"\nCurrent Status (2024):")
   print(f"  Oil Rate: {oil_rate_bpd[-1]:,} BPD")
   print(f"  Water Cut: {water_cut[-1]:.0f}%")
   print(f"  Cumulative Production: 45 MMSTB (38% of OOIP)")
   print(f"  Remaining Reserves: ~30 MMSTB")
   print(f"  Wells: 25 (12 on artificial lift, 8 shut-in)")

1.2 The Challenge
^^^^^^^^^^^^^^^^^

**Problem Statement:**

Management wants to reverse production decline and increase recovery. Options include:

1. **Infill drilling** - Add horizontal wells
2. **Waterflood** - Implement pressure maintenance
3. **Workover campaign** - Optimize existing wells
4. **Artificial lift optimization** - Maximize production from current wells

**Constraints:**

- Budget: $50MM over 3 years
- Target: Increase production to 8,000 BPD
- Maximize NPV
- No major environmental issues

1.3 Analysis
^^^^^^^^^^^^

**Step 1: Diagnostic Analysis**

.. code-block:: python

   class WestMesaDiagnostics:
       """Analyze West Mesa field performance"""
       
       def __init__(self):
           self.reservoir_pressure = 2200  # psi (depleted from 3500)
           self.bubble_point = 2800  # psi
           self.current_rate = 4800  # BPD
           self.n_wells_active = 17
       
       def analyze_production_decline(self):
           """Analyze decline and identify causes"""
           
           findings = []
           
           # 1. Pressure depletion
           pressure_depletion = (3500 - self.reservoir_pressure) / 3500 * 100
           findings.append({
               'issue': 'Severe pressure depletion',
               'metric': f'{pressure_depletion:.0f}% pressure loss',
               'impact': 'Reduced driving force, gas coming out of solution',
               'severity': 'HIGH'
           })
           
           # 2. High water cut
           findings.append({
               'issue': 'High water production',
               'metric': '45% water cut',
               'impact': 'Lifting costs increased, reduced oil productivity',
               'severity': 'MEDIUM'
           })
           
           # 3. Shut-in wells
           shut_in_wells = 25 - 17
           findings.append({
               'issue': 'Wells shut-in due to low productivity',
               'metric': f'{shut_in_wells} wells idle',
               'impact': 'Stranded reserves, lost production',
               'severity': 'MEDIUM'
           })
           
           # 4. Suboptimal artificial lift
           findings.append({
               'issue': 'Artificial lift not optimized',
               'metric': '12 wells on lift, varying efficiency',
               'impact': '10-20% production loss potential',
               'severity': 'LOW-MEDIUM'
           })
           
           return findings
       
       def estimate_recovery_opportunities(self):
           """Estimate recovery potential from each option"""
           
           opportunities = []
           
           # Option 1: Infill drilling
           infill_wells = 10
           rate_per_well = 400  # BPD (horizontal)
           infill_cost = 8 * infill_wells  # $8MM per well
           
           opportunities.append({
               'option': 'Infill Horizontal Wells',
               'wells': infill_wells,
               'incremental_rate_bpd': rate_per_well * infill_wells,
               'capex_mm': infill_cost,
               'incremental_reserves_mmstb': 15,
               'npv_mm': 120  # Estimated
           })
           
           # Option 2: Waterflood
           waterflood_rate_increase = 2500  # BPD from pressure support
           waterflood_cost = 35  # Facilities
           
           opportunities.append({
               'option': 'Waterflood Implementation',
               'wells': 'Existing + 5 injectors',
               'incremental_rate_bpd': waterflood_rate_increase,
               'capex_mm': waterflood_cost,
               'incremental_reserves_mmstb': 12,
               'npv_mm': 85
           })
           
           # Option 3: Workover campaign
           workover_wells = 15
           rate_increase_per_well = 150  # BPD average
           workover_cost = 0.5 * workover_wells  # $500k per WO
           
           opportunities.append({
               'option': 'Workover Campaign',
               'wells': workover_wells,
               'incremental_rate_bpd': rate_increase_per_well * workover_wells,
               'capex_mm': workover_cost,
               'incremental_reserves_mmstb': 5,
               'npv_mm': 35
           })
           
           # Option 4: Artificial lift optimization
           lift_wells = 17
           rate_increase_per_well = 80  # BPD (10-15% improvement)
           lift_cost = 2  # ESP replacements, VFDs
           
           opportunities.append({
               'option': 'Artificial Lift Optimization',
               'wells': lift_wells,
               'incremental_rate_bpd': rate_increase_per_well * lift_wells,
               'capex_mm': lift_cost,
               'incremental_reserves_mmstb': 2,
               'npv_mm': 18
           })
           
           return opportunities
   
   # Run diagnostics
   diagnostics = WestMesaDiagnostics()
   
   findings = diagnostics.analyze_production_decline()
   
   print(f"\n=== DIAGNOSTIC FINDINGS ===")
   for i, finding in enumerate(findings, 1):
       print(f"\n{i}. {finding['issue']} (Severity: {finding['severity']})")
       print(f"   Metric: {finding['metric']}")
       print(f"   Impact: {finding['impact']}")
   
   # Evaluate opportunities
   opportunities = diagnostics.estimate_recovery_opportunities()
   
   print(f"\n=== RECOVERY OPPORTUNITIES ===")
   print(f"{'Option':<30} {'Rate Increase':<15} {'Capex':<12} {'NPV':<12}")
   print(f"{'':30} {'(BPD)':<15} {'($MM)':<12} {'($MM)':<12}")
   print("-" * 75)
   
   for opp in opportunities:
       print(f"{opp['option']:<30} "
             f"{opp['incremental_rate_bpd']:<15,.0f} "
             f"{opp['capex_mm']:<12.0f} "
             f"{opp['npv_mm']:<12.0f}")

**Step 2: Integrated Development Plan**

.. code-block:: python

   def create_integrated_plan(budget_mm=50):
       """
       Create optimal development plan within budget.
       
       Use multiple approaches for maximum value.
       """
       
       plan = []
       remaining_budget = budget_mm
       total_rate_increase = 0
       total_npv = 0
       
       # Phase 1: Quick wins - Artificial lift optimization (Year 1)
       if remaining_budget >= 2:
           plan.append({
               'phase': 1,
               'action': 'Artificial Lift Optimization',
               'year': 1,
               'capex_mm': 2,
               'rate_increase_bpd': 1360,
               'npv_mm': 18,
               'timeline_months': 6
           })
           remaining_budget -= 2
           total_rate_increase += 1360
           total_npv += 18
       
       # Phase 2: Workover campaign (Year 1-2)
       if remaining_budget >= 7.5:
           plan.append({
               'phase': 2,
               'action': 'Workover 15 Wells',
               'year': 1,
               'capex_mm': 7.5,
               'rate_increase_bpd': 2250,
               'npv_mm': 35,
               'timeline_months': 12
           })
           remaining_budget -= 7.5
           total_rate_increase += 2250
           total_npv += 35
       
       # Phase 3: Waterflood (Year 2-3)
       if remaining_budget >= 35:
           plan.append({
               'phase': 3,
               'action': 'Implement Waterflood',
               'year': 2,
               'capex_mm': 35,
               'rate_increase_bpd': 2500,
               'npv_mm': 85,
               'timeline_months': 18
           })
           remaining_budget -= 35
           total_rate_increase += 2500
           total_npv += 85
       
       # Phase 4: Infill drilling if budget allows
       if remaining_budget >= 8:
           n_wells = int(remaining_budget / 8)
           plan.append({
               'phase': 4,
               'action': f'{n_wells} Infill Horizontal Wells',
               'year': 3,
               'capex_mm': n_wells * 8,
               'rate_increase_bpd': n_wells * 400,
               'npv_mm': n_wells * 12,  # Scaled NPV
               'timeline_months': n_wells * 3
           })
           total_rate_increase += n_wells * 400
           total_npv += n_wells * 12
       
       return {
           'plan': plan,
           'total_capex_mm': budget_mm - remaining_budget,
           'total_rate_increase_bpd': total_rate_increase,
           'total_npv_mm': total_npv,
           'target_rate_bpd': 4800 + total_rate_increase
       }
   
   # Create plan
   integrated_plan = create_integrated_plan(budget_mm=50)
   
   print(f"\n=== INTEGRATED DEVELOPMENT PLAN ===")
   print(f"Budget: $50 MM")
   print(f"Target Rate: 8,000 BPD\n")
   
   for phase in integrated_plan['plan']:
       print(f"Phase {phase['phase']} (Year {phase['year']}):")
       print(f"  Action: {phase['action']}")
       print(f"  Capex: ${phase['capex_mm']:.1f} MM")
       print(f"  Rate Increase: {phase['rate_increase_bpd']:,.0f} BPD")
       print(f"  NPV: ${phase['npv_mm']:.0f} MM")
       print(f"  Timeline: {phase['timeline_months']} months\n")
   
   print(f"=== SUMMARY ===")
   print(f"Total Capex: ${integrated_plan['total_capex_mm']:.1f} MM")
   print(f"Total Rate Increase: {integrated_plan['total_rate_increase_bpd']:,.0f} BPD")
   print(f"Target Production: {integrated_plan['target_rate_bpd']:,.0f} BPD")
   print(f"Total NPV: ${integrated_plan['total_npv_mm']:.0f} MM")
   print(f"NPV/Capex Ratio: {integrated_plan['total_npv_mm']/integrated_plan['total_capex_mm']:.1f}")

1.4 Recommended Solution
^^^^^^^^^^^^^^^^^^^^^^^^^

**Phased Approach:**

✅ **Phase 1 (Year 1):** Artificial lift optimization + Workover campaign
- Quick production increase
- Low risk
- Generate cash flow
- Rate increase: ~3,600 BPD

✅ **Phase 2 (Year 2-3):** Waterflood implementation
- Pressure maintenance
- Maximize recovery
- Rate increase: ~2,500 BPD

✅ **Phase 3 (Year 3+):** Selective infill drilling if justified
- Target remaining reserves
- Data-driven locations

**Total Investment:** ~$45MM  
**Production Target:** 8,000-9,000 BPD ✅  
**NPV:** $138MM  
**Payout:** 2.5 years  

1.5 Results (What Actually Happened)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Implementation:**

- Phase 1 completed on time and budget
- Artificial lift optimization achieved 12% production increase
- Workovers exceeded expectations (avg 180 BPD increase vs 150 forecast)
- Waterflood implemented successfully

**Actual Results (After 3 years):**

- Production increased to **8,400 BPD** (target: 8,000) ✅
- Water cut stabilized at **42%** (vs rising trend)
- Reservoir pressure increased from 2,200 to 2,650 psi
- 8 previously shut-in wells brought back online
- Incremental reserves: 14 MMSTB
- Actual NPV: **$145MM** (forecast: $138MM)

1.6 Lessons Learned
^^^^^^^^^^^^^^^^^^^

✅ **Phased approach works** - De-risk through stages  
✅ **Quick wins matter** - Early cashflow funds later phases  
✅ **Pressure maintenance critical** - Waterflood extended field life  
✅ **Integration is key** - Multiple solutions better than single approach  
✅ **Data-driven decisions** - Surveillance identified best opportunities  

**Key Quote:**

    *"We didn't need the highest-tech solution. We needed the right combination
    of proven technologies applied systematically."*
    
    — Field Manager, West Mesa Field

----

Case Study 2: Deepwater Development Challenge
----------------------------------------------

2.1 Background
^^^^^^^^^^^^^^

**Project Overview:**

- **Location:** Gulf of Mexico, 7,000 ft water depth
- **Discovery:** 2018
- **Reservoir:** Turbidite sandstone, 18,000 ft subsea
- **OOIP:** 350 MMSTB (P50)
- **Estimated Recoverable:** 140 MMSTB (40% RF)
- **Challenges:** Deepwater, high pressure/high temperature

**Technical Parameters:**

- Reservoir pressure: 14,500 psi
- Temperature: 285°F
- Water depth: 7,000 ft
- Pay thickness: 150 ft (gross), 100 ft (net)
- Permeability: 500 md (excellent)
- API gravity: 28° (medium oil)

2.2 The Challenge
^^^^^^^^^^^^^^^^^

**Decision Point:**

Select optimal development concept for FID:

**Option A: Subsea Tieback**
- 6 subsea wells
- Tieback to existing platform (40 miles)
- Capex: $1.2B
- Lower risk, lower recovery

**Option B: Standalone FPSO**
- 12 subsea wells
- Floating production facility
- Capex: $2.8B
- Higher risk, higher recovery

**Option C: Phased Development**
- Phase 1: 4 wells + tieback
- Phase 2: Add FPSO if justified
- Capex: $0.8B + $2.0B
- Flexible but potentially higher total cost

2.3 Analysis
^^^^^^^^^^^^

.. code-block:: python

   class DeepwaterEconomics:
       """Evaluate deepwater development options"""
       
       def __init__(self, reserves_mmstb=140):
           self.reserves = reserves_mmstb
           self.oil_price = 75  # $/bbl
           self.discount_rate = 0.12  # Higher for deepwater risk
       
       def evaluate_option(self, concept_params):
           """Evaluate development concept"""
           
           # Production profile
           if concept_params['name'] == 'Subsea Tieback':
               plateau_rate = 35000  # BPD (limited by tieback capacity)
               plateau_years = 4
               capex = 1200  # $MM
               opex_per_bbl = 12
               time_to_first_oil = 3  # years
           
           elif concept_params['name'] == 'Standalone FPSO':
               plateau_rate = 80000  # BPD
               plateau_years = 5
               capex = 2800  # $MM
               opex_per_bbl = 18  # Higher for FPSO
               time_to_first_oil = 5  # years (longer development)
           
           else:  # Phased
               plateau_rate = 25000  # BPD (Phase 1)
               plateau_years = 3
               capex = 800  # $MM (Phase 1 only)
               opex_per_bbl = 14
               time_to_first_oil = 2.5  # years
           
           # Generate production profile
           years = []
           production = []
           
           # Ramp-up
           years.append(time_to_first_oil)
           production.append(plateau_rate * 0.5 * 365 / 1000)  # Half year
           
           # Plateau
           for y in range(int(plateau_years)):
               years.append(time_to_first_oil + y + 1)
               production.append(plateau_rate * 365 / 1000)  # MBBL
           
           # Decline (12% annual)
           current_rate = plateau_rate
           year = time_to_first_oil + plateau_years + 1
           
           while current_rate > 5000 and sum(production) < self.reserves * 1000 * 0.9:
               current_rate *= 0.88  # 12% decline
               years.append(year)
               production.append(current_rate * 365 / 1000)  # MBBL
               year += 1
               
               if year > 30:
                   break
           
           # Calculate NPV
           npv = -capex * 1e6  # Initial investment
           
           for y, prod_mbbl in zip(years, production):
               revenue = prod_mbbl * 1000 * self.oil_price
               opex = prod_mbbl * 1000 * opex_per_bbl
               cashflow = revenue - opex
               
               npv += cashflow / (1 + self.discount_rate) ** y
           
           cumulative_prod = sum(production) / 1000  # MMSTB
           
           return {
               'name': concept_params['name'],
               'npv_mm': npv / 1e6,
               'capex_mm': capex,
               'peak_rate_bpd': plateau_rate,
               'cumulative_production_mmstb': cumulative_prod,
               'recovery_factor': cumulative_prod / self.reserves,
               'time_to_first_oil_years': time_to_first_oil,
               'field_life_years': len(years),
               'production_profile': list(zip(years, production))
           }
   
   # Evaluate all options
   evaluator = DeepwaterEconomics(reserves_mmstb=140)
   
   options = [
       {'name': 'Subsea Tieback'},
       {'name': 'Standalone FPSO'},
       {'name': 'Phased Development'}
   ]
   
   results = [evaluator.evaluate_option(opt) for opt in options]
   
   print("\n" + "="*80)
   print("     CASE STUDY 2: DEEPWATER DEVELOPMENT CONCEPT SELECTION")
   print("="*80)
   
   print(f"\n{'Concept':<25} {'NPV':<12} {'Capex':<12} {'Peak Rate':<12} {'Recovery':<10}")
   print(f"{'':25} {'($MM)':<12} {'($MM)':<12} {'(BPD)':<12} {'(%)':<10}")
   print("-" * 80)
   
   for r in sorted(results, key=lambda x: x['npv_mm'], reverse=True):
       print(f"{r['name']:<25} "
             f"{r['npv_mm']:<12,.0f} "
             f"{r['capex_mm']:<12,.0f} "
             f"{r['peak_rate_bpd']:<12,.0f} "
             f"{r['recovery_factor']*100:<10.0f}%")

2.4 Decision Analysis
^^^^^^^^^^^^^^^^^^^^^

**Risk-Adjusted Analysis:**

.. code-block:: python

   def risk_adjusted_npv(base_npv_mm, concept):
       """Apply risk factors to NPV"""
       
       risk_factors = {
           'Subsea Tieback': {
               'technical_success': 0.90,
               'execution_risk': 0.95,
               'reservoir_risk': 0.85,
               'overall': 0.90 * 0.95 * 0.85
           },
           'Standalone FPSO': {
               'technical_success': 0.80,
               'execution_risk': 0.85,
               'reservoir_risk': 0.85,
               'overall': 0.80 * 0.85 * 0.85
           },
           'Phased Development': {
               'technical_success': 0.92,
               'execution_risk': 0.90,
               'reservoir_risk': 0.85,
               'overall': 0.92 * 0.90 * 0.85
           }
       }
       
       risk = risk_factors[concept]
       risk_adjusted = base_npv_mm * risk['overall']
       
       return {
           'base_npv_mm': base_npv_mm,
           'risk_adjusted_npv_mm': risk_adjusted,
           'probability_of_success': risk['overall'],
           'risk_factors': risk
       }
   
   print(f"\n=== RISK-ADJUSTED NPV ===")
   
   for result in results:
       risk_analysis = risk_adjusted_npv(result['npv_mm'], result['name'])
       
       print(f"\n{result['name']}:")
       print(f"  Base NPV: ${risk_analysis['base_npv_mm']:,.0f} MM")
       print(f"  Probability of Success: {risk_analysis['probability_of_success']*100:.0f}%")
       print(f"  Risk-Adjusted NPV: ${risk_analysis['risk_adjusted_npv_mm']:,.0f} MM")

2.5 Recommended Solution
^^^^^^^^^^^^^^^^^^^^^^^^^

**Decision: Phased Development**

**Rationale:**

✅ **Lowest initial capital** - $800MM vs $1.2B or $2.8B  
✅ **Fastest to first oil** - 2.5 years  
✅ **Flexibility** - Evaluate Phase 2 with production data  
✅ **Risk mitigation** - Prove reservoir before major investment  
✅ **Highest risk-adjusted NPV** - Accounting for uncertainty  

**Implementation Plan:**

**Phase 1 (Years 0-3):**
- Drill 4 subsea wells
- Tieback to existing facility
- Target production: 25,000 BPD
- Gather production data

**Decision Gate (Year 3):**
- Evaluate reservoir performance
- Update reserves estimate
- Decide on Phase 2

**Phase 2 (Years 4-6, if justified):**
- Install FPSO
- Drill 8 additional wells
- Increase to 80,000 BPD
- Maximize recovery

2.6 Results
^^^^^^^^^^^

**Phase 1 Results (Actual):**

- First oil: Month 31 (vs 30 forecast) ✅
- Production: 28,000 BPD (vs 25,000 forecast) ✅
- Wells performed better than expected
- Reservoir pressure higher than predicted
- Identified additional 50 MMSTB potential

**Phase 2 Decision:**

✅ **FID approved** based on Phase 1 success  
- Updated reserves: 175 MMSTB (vs 140 original)
- Phase 2 NPV: $1.8B
- Combined NPV: $2.5B

**Final Outcome:**

- Total investment: $2.8B
- Total NPV: $2.5B
- Production: 85,000 BPD (peak)
- Recovery: 45% (vs 40% original)

2.7 Lessons Learned
^^^^^^^^^^^^^^^^^^^

✅ **Optionality has value** - Phased approach reduced risk  
✅ **Learn before scaling** - Phase 1 data improved Phase 2 design  
✅ **Conservative assumptions** - Reservoir exceeded expectations  
✅ **Flexibility beats optimization** - Phased won despite lower initial NPV  
✅ **Risk matters** - Risk-adjusted NPV drove correct decision  

----

Case Study 3: Unconventional Shale Development
-----------------------------------------------

3.1 Background
^^^^^^^^^^^^^^

**Play Overview:**

- **Location:** Eagle Ford Shale, South Texas
- **Operator:** Independent E&P company
- **Acreage:** 50,000 acres (80% working interest)
- **Resource:** 200 MMSTB oil equivalent (company share)
- **Well Type:** Horizontal wells, multistage fracturing

**Challenge:**

Optimize development for maximum value with limited capital.

3.2 Optimization Problem
^^^^^^^^^^^^^^^^^^^^^^^^

**Key Decisions:**

1. **Well spacing** - 80 vs 120 vs 160 acre spacing?
2. **Lateral length** - 5,000 vs 7,500 vs 10,000 ft?
3. **Frac design** - Number of stages, proppant loading?
4. **Development pace** - How many wells per year?

.. code-block:: python

   class ShaleOptimization:
       """Optimize unconventional development"""
       
       def __init__(self, acreage, oil_price=65):
           self.acreage = acreage
           self.oil_price = oil_price
       
       def evaluate_spacing(self, spacing_acres):
           """Evaluate well spacing option"""
           
           # Number of wells
           n_wells = self.acreage / spacing_acres
           
           # EUR per well (decreases with tighter spacing due to interference)
           if spacing_acres == 80:
               eur_per_well = 350000  # STB (interference)
           elif spacing_acres == 120:
               eur_per_well = 420000  # STB (sweet spot)
           else:  # 160
               eur_per_well = 450000  # STB (less interference)
           
           # Economics
           well_cost = 8.0  # $MM per well
           total_capex = n_wells * well_cost
           total_eur = n_wells * eur_per_well / 1e6  # MMSTB
           
           # Simplified NPV (50% of revenue - capex)
           revenue = total_eur * 1e6 * self.oil_price
           npv = revenue * 0.50 - total_capex * 1e6
           
           return {
               'spacing_acres': spacing_acres,
               'n_wells': int(n_wells),
               'eur_per_well_mstb': eur_per_well / 1000,
               'total_eur_mmstb': total_eur,
               'total_capex_mm': total_capex,
               'npv_mm': npv / 1e6,
               'npv_per_well_mm': npv / 1e6 / n_wells
           }
       
       def optimize_development(self):
           """Find optimal spacing"""
           
           spacings = [80, 120, 160]
           results = [self.evaluate_spacing(s) for s in spacings]
           
           # Sort by NPV
           results.sort(key=lambda x: x['npv_mm'], reverse=True)
           
           return results
   
   # Run optimization
   optimizer = ShaleOptimization(acreage=50000, oil_price=65)
   results = optimizer.optimize_development()
   
   print("\n" + "="*80)
   print("     CASE STUDY 3: UNCONVENTIONAL SHALE OPTIMIZATION")
   print("="*80)
   
   print(f"\n{'Spacing':<12} {'Wells':<10} {'EUR/Well':<12} {'Total EUR':<12} {'NPV':<12}")
   print(f"{'(acres)':<12} {'(#)':<10} {'(MSTB)':<12} {'(MMSTB)':<12} {'($MM)':<12}")
   print("-" * 75)
   
   for r in results:
       print(f"{r['spacing_acres']:<12} "
             f"{r['n_wells']:<10} "
             f"{r['eur_per_well_mstb']:<12.0f} "
             f"{r['total_eur_mmstb']:<12.0f} "
             f"{r['npv_mm']:<12,.0f}")
   
   print(f"\n✅ Optimal: {results[0]['spacing_acres']} acre spacing")

3.3 Solution
^^^^^^^^^^^^

**Optimal Development Plan:**

- **Spacing:** 120 acres (sweet spot between density and interference)
- **Wells:** 417 wells over 8 years
- **Pace:** 50-60 wells/year
- **Lateral length:** 7,500 ft (cost/benefit optimized)
- **Frac design:** 35 stages, 2,500 lb/ft proppant

**Total Investment:** $3.3B  
**Total NPV:** $1.9B  
**Peak Production:** 55,000 BOEPD  

3.4 Lessons Learned
^^^^^^^^^^^^^^^^^^^

✅ **Sweet spot exists** - Not always maximum density  
✅ **Data-driven** - Use pilot wells to optimize  
✅ **Continuous improvement** - Adjust as you learn  
✅ **Manufacturing approach** - Standardize for efficiency  

----

Summary: Key Themes Across All Case Studies
--------------------------------------------

**Common Success Factors:**

1. **Integrated analysis** - Consider all aspects
2. **Phased approach** - Reduce risk, maintain flexibility
3. **Data-driven decisions** - Use evidence, not assumptions
4. **Risk management** - Identify and mitigate early
5. **Economic focus** - Maximize NPV, not production
6. **Continuous optimization** - Monitor and adjust

**Common Pitfalls to Avoid:**

❌ Single solution focus - Consider multiple options  
❌ Ignoring uncertainty - Use probabilistic methods  
❌ Technology for technology's sake - Focus on value  
❌ Underestimating execution risk - Plan realistically  
❌ Forgetting the basics - Pressure, productivity, economics  

**Decision-Making Framework:**

.. code-block:: text

   1. Define objective (maximize NPV, minimize risk, etc.)
   2. Generate options (multiple approaches)
   3. Analyze technically (reserves, rates, costs)
   4. Evaluate economically (NPV, IRR, payout)
   5. Assess risks (probability, impact, mitigation)
   6. Make decision (risk-adjusted value)
   7. Implement (phased, monitored)
   8. Optimize (continuous improvement)

----

Practice Exercise: Your Turn
-----------------------------

**Scenario:**

You're the lead engineer for a marginal offshore field:

- **Reserves:** 25 MMSTB (P50)
- **Water depth:** 300 ft
- **Distance to shore:** 15 miles
- **Current economics:** Breakeven at $60/bbl
- **Challenge:** Oil price dropped to $55/bbl

**Your Task:**

1. Identify cost reduction opportunities
2. Evaluate development options
3. Recommend go/no-go decision
4. Justify your recommendation

**Deliverables:**

- Technical analysis
- Economic evaluation
- Risk assessment
- Recommendation with rationale

----

**Module Complete!** ✅

Congratulations! You've completed all case studies and the entire course.

**You now have:**

✅ Comprehensive petroleum engineering knowledge  
✅ Practical problem-solving skills  
✅ Real-world application experience  
✅ Integrated workflow capabilities  
✅ Economic evaluation expertise  
✅ Risk assessment abilities  

**What's Next?**

- Apply these skills to your projects
- Continue learning from experience
- Share knowledge with colleagues
- Keep building your expertise

**Remember:**

    *"Engineering judgment comes from experience, and experience
    comes from bad judgment. Learn from every project!"*

----

**COURSE COMPLETE!** 🎓

Thank you for taking this journey through petroleum engineering.

Go build something amazing! 🚀
