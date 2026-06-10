Module 8: Integrated Workflows
===============================

.. meta::
   :description: Build end-to-end petroleum engineering workflows integrating all disciplines
   :keywords: integrated workflows, field development, reservoir management, production optimization

**Learning Objectives**

After completing this module, you will be able to:

- Build end-to-end reservoir-to-surface workflows
- Integrate drilling, completion, and production engineering
- Create automated analysis pipelines
- Design field development strategies
- Optimize well spacing and facilities
- Build digital twin applications
- Implement real-time surveillance systems
- Apply systems thinking to petroleum engineering

**Time Commitment:** 6-8 hours

**Prerequisites:** Modules 1-7

----

Introduction: The Value of Integration
---------------------------------------

**The Problem with Silos:**

Traditional petroleum engineering operates in silos:

- 📊 **Geologists** build models
- 🔬 **Reservoir engineers** calculate reserves
- 🔧 **Drilling engineers** design wells
- ⚙️ **Production engineers** optimize lift
- 💰 **Economists** evaluate projects

**But they don't talk to each other!**

**Consequences:**

- ❌ Drilling designs that ignore production constraints
- ❌ Completions that don't match reservoir quality
- ❌ Production optimization without reservoir updates
- ❌ Suboptimal field development
- ❌ Missed opportunities for value creation

**The Integrated Approach:**

✅ **Shared data and models**  
✅ **Cross-discipline workflows**  
✅ **Automated decision support**  
✅ **Real-time optimization**  
✅ **System-level thinking**  

**Value Creation:**

Studies show integrated workflows can deliver:

- 15-30% production increase
- 20-40% cost reduction
- 50%+ faster decisions
- Better ultimate recovery

----

Part 1: Reservoir-to-Surface Workflow
--------------------------------------

1.1 Complete System Model
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Components:**

.. code-block:: text

   ┌─────────────────────────────────────────────────┐
   │                  RESERVOIR                       │
   │  • Permeability                                  │
   │  • Pressure                                      │
   │  • Saturation                                    │
   │  • IPR curve                                     │
   └──────────────────┬──────────────────────────────┘
                      │
   ┌──────────────────▼──────────────────────────────┐
   │              WELL COMPLETION                     │
   │  • Perforations                                  │
   │  • Sand control                                  │
   │  • Stimulation                                   │
   │  • Completion skin                               │
   └──────────────────┬──────────────────────────────┘
                      │
   ┌──────────────────▼──────────────────────────────┐
   │            ARTIFICIAL LIFT                       │
   │  • ESP/Gas lift/Rod pump                         │
   │  • Lift curve                                    │
   │  • Operating point                               │
   └──────────────────┬──────────────────────────────┘
                      │
   ┌──────────────────▼──────────────────────────────┐
   │          GATHERING SYSTEM                        │
   │  • Flowlines                                     │
   │  • Manifolds                                     │
   │  • Pressure drop                                 │
   └──────────────────┬──────────────────────────────┘
                      │
   ┌──────────────────▼──────────────────────────────┐
   │            SEPARATOR                             │
   │  • Operating pressure                            │
   │  • Phase separation                              │
   │  • Production rates                              │
   └──────────────────────────────────────────────────┘

**Implementation:**

.. code-block:: python

   from petrosmith.api import ReservoirAPI, ProductionAPI, WellAPI
   from petrosmith.core import ProductionCalculations, DrillingCalculations
   import numpy as np
   import matplotlib.pyplot as plt
   
   class IntegratedWellModel:
       """
       Integrated reservoir-to-surface well model.
       
       Combines IPR, VLP, artificial lift, and gathering system.
       """
       
       def __init__(self, well_id, well_config):
           self.well_id = well_id
           self.config = well_config
           
           # Initialize APIs
           self.reservoir_api = ReservoirAPI()
           self.production_api = ProductionAPI()
           self.well_api = WellAPI()
       
       def calculate_ipr(self, reservoir_pressure, productivity_index):
           """Calculate inflow performance"""
           pressures = np.linspace(0, reservoir_pressure, 50)
           rates = []
           
           for pwf in pressures:
               # Vogel IPR for undersaturated oil
               qmax = productivity_index * reservoir_pressure / 1.8
               pr_ratio = pwf / reservoir_pressure
               
               q = qmax * (1 - 0.2*pr_ratio - 0.8*pr_ratio**2)
               rates.append(q)
           
           return pressures, rates
       
       def calculate_vlp(self, rates, separator_pressure):
           """Calculate vertical lift performance"""
           pressures = []
           
           for rate in rates:
               if rate == 0:
                   pwf = separator_pressure
               else:
                   # Simplified gradient calculation
                   gradient = 0.35  # psi/ft (with gas effect)
                   friction = 0.001 * rate  # Simplified friction
                   
                   pwf = separator_pressure + (
                       gradient * self.config['depth'] + 
                       friction
                   )
               
               pressures.append(pwf)
           
           return pressures
       
       def find_operating_point(self, reservoir_pressure, pi, sep_pressure):
           """Find natural flow operating point"""
           
           # IPR curve
           ipr_p, ipr_q = self.calculate_ipr(reservoir_pressure, pi)
           
           # VLP curve
           vlp_p = self.calculate_vlp(ipr_q, sep_pressure)
           
           # Find intersection
           ipr_arr = np.array(ipr_p)
           vlp_arr = np.array(vlp_p)
           
           diff = np.abs(ipr_arr - vlp_arr)
           idx = np.argmin(diff)
           
           operating_rate = ipr_q[idx]
           operating_pressure = ipr_p[idx]
           
           return {
               'rate': operating_rate,
               'pressure': operating_pressure,
               'ipr_curve': (ipr_p, ipr_q),
               'vlp_curve': (ipr_q, vlp_p)
           }
       
       def optimize_artificial_lift(self, reservoir_pressure, pi, target_rate):
           """Determine if artificial lift is needed and optimize"""
           
           # Check natural flow capability
           natural_flow = self.find_operating_point(
               reservoir_pressure, 
               pi, 
               self.config['separator_pressure']
           )
           
           if natural_flow['rate'] >= target_rate:
               return {
                   'lift_required': False,
                   'natural_flow_rate': natural_flow['rate'],
                   'recommendation': 'Natural flow sufficient'
               }
           else:
               # Need artificial lift
               shortfall = target_rate - natural_flow['rate']
               
               # Evaluate lift options
               lift_options = self.evaluate_lift_methods(
                   target_rate, 
                   reservoir_pressure
               )
               
               return {
                   'lift_required': True,
                   'natural_flow_rate': natural_flow['rate'],
                   'target_rate': target_rate,
                   'shortfall': shortfall,
                   'recommended_lift': lift_options[0],
                   'all_options': lift_options
               }
       
       def evaluate_lift_methods(self, target_rate, reservoir_pressure):
           """Evaluate artificial lift methods"""
           
           options = []
           
           # ESP evaluation
           if target_rate > 200 and self.config['water_cut'] < 80:
               options.append({
                   'method': 'ESP',
                   'applicability_score': 0.9,
                   'estimated_cost': 200000,
                   'operating_cost_per_day': 150,
                   'reliability': 'Good',
                   'pros': ['High rate capability', 'Efficient'],
                   'cons': ['High upfront cost', 'GOR limited']
               })
           
           # Gas lift evaluation
           if self.config.get('gas_available', False):
               options.append({
                   'method': 'Gas Lift',
                   'applicability_score': 0.85,
                   'estimated_cost': 150000,
                   'operating_cost_per_day': 200,  # Gas cost
                   'reliability': 'Excellent',
                   'pros': ['Reliable', 'Simple', 'Handles gas'],
                   'cons': ['Needs gas source', 'Moderate efficiency']
               })
           
           # Rod pump evaluation
           if target_rate < 500 and self.config['deviation'] < 10:
               options.append({
                   'method': 'Rod Pump',
                   'applicability_score': 0.75,
                   'estimated_cost': 100000,
                   'operating_cost_per_day': 50,
                   'reliability': 'Good',
                   'pros': ['Low cost', 'Simple', 'Mature technology'],
                   'cons': ['Rate limited', 'Vertical wells only']
               })
           
           # Sort by applicability
           options.sort(key=lambda x: x['applicability_score'], reverse=True)
           
           return options
       
       def complete_system_analysis(self, reservoir_pressure, pi, target_rate):
           """Run complete integrated analysis"""
           
           results = {
               'well_id': self.well_id,
               'reservoir_pressure': reservoir_pressure,
               'productivity_index': pi,
               'target_rate': target_rate
           }
           
           # Natural flow analysis
           natural_flow = self.find_operating_point(
               reservoir_pressure, 
               pi, 
               self.config['separator_pressure']
           )
           results['natural_flow'] = natural_flow
           
           # Artificial lift analysis
           lift_analysis = self.optimize_artificial_lift(
               reservoir_pressure, 
               pi, 
               target_rate
           )
           results['artificial_lift'] = lift_analysis
           
           # Economic analysis
           results['economics'] = self.calculate_economics(
               natural_flow['rate'] if not lift_analysis['lift_required'] else target_rate,
               lift_analysis.get('recommended_lift')
           )
           
           return results
       
       def calculate_economics(self, production_rate, lift_method=None):
           """Calculate project economics"""
           
           # Revenue
           oil_price = 70  # $/bbl
           annual_revenue = production_rate * 365 * oil_price
           
           # Operating costs
           base_opex_per_bbl = 15
           annual_opex = production_rate * 365 * base_opex_per_bbl
           
           if lift_method:
               annual_opex += lift_method['operating_cost_per_day'] * 365
               capex = lift_method['estimated_cost']
           else:
               capex = 0
           
           # Simple NPV (5 years, 10% discount)
           years = 5
           discount = 0.10
           
           npv = -capex
           for year in range(1, years + 1):
               cashflow = annual_revenue - annual_opex
               npv += cashflow / (1 + discount) ** year
           
           return {
               'annual_revenue': annual_revenue,
               'annual_opex': annual_opex,
               'capex': capex,
               'npv_5yr': npv,
               'payout_years': capex / (annual_revenue - annual_opex) if annual_revenue > annual_opex else None
           }
   
   # Example: Run integrated analysis
   well_config = {
       'depth': 8000,
       'separator_pressure': 100,
       'water_cut': 30,
       'gor': 500,
       'deviation': 5,
       'gas_available': True
   }
   
   model = IntegratedWellModel('WELL-001', well_config)
   
   results = model.complete_system_analysis(
       reservoir_pressure=3000,
       pi=2.0,
       target_rate=800
   )
   
   print("\n" + "="*60)
   print("       INTEGRATED WELL ANALYSIS RESULTS")
   print("="*60)
   
   print(f"\nWell ID: {results['well_id']}")
   print(f"Reservoir Pressure: {results['reservoir_pressure']} psi")
   print(f"Productivity Index: {results['productivity_index']} STB/day/psi")
   print(f"Target Rate: {results['target_rate']} STB/day")
   
   print(f"\n--- NATURAL FLOW ---")
   print(f"Operating Rate: {results['natural_flow']['rate']:.0f} STB/day")
   print(f"Operating Pressure: {results['natural_flow']['pressure']:.0f} psi")
   
   print(f"\n--- ARTIFICIAL LIFT ---")
   if results['artificial_lift']['lift_required']:
       print(f"✅ Artificial Lift REQUIRED")
       print(f"Shortfall: {results['artificial_lift']['shortfall']:.0f} STB/day")
       rec = results['artificial_lift']['recommended_lift']
       print(f"\nRecommended: {rec['method']}")
       print(f"  Capex: ${rec['estimated_cost']:,.0f}")
       print(f"  Opex: ${rec['operating_cost_per_day']:.0f}/day")
       print(f"  Pros: {', '.join(rec['pros'])}")
       print(f"  Cons: {', '.join(rec['cons'])}")
   else:
       print(f"✅ Natural Flow Sufficient")
   
   print(f"\n--- ECONOMICS ---")
   econ = results['economics']
   print(f"Annual Revenue: ${econ['annual_revenue']:,.0f}")
   print(f"Annual Opex: ${econ['annual_opex']:,.0f}")
   print(f"Capex: ${econ['capex']:,.0f}")
   print(f"NPV (5 years): ${econ['npv_5yr']:,.0f}")
   if econ['payout_years']:
       print(f"Payout: {econ['payout_years']:.1f} years")

1.2 Multi-Well Field Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Field-Level Integration:**

.. code-block:: python

   class FieldModel:
       """
       Integrated field model with multiple wells and facilities.
       """
       
       def __init__(self, field_name):
           self.field_name = field_name
           self.wells = {}
           self.facilities = {}
       
       def add_well(self, well_id, well_model):
           """Add well to field"""
           self.wells[well_id] = well_model
       
       def add_facility(self, facility_id, capacity_bpd, operating_pressure):
           """Add production facility"""
           self.facilities[facility_id] = {
               'capacity': capacity_bpd,
               'operating_pressure': operating_pressure,
               'connected_wells': []
           }
       
       def connect_well_to_facility(self, well_id, facility_id):
           """Connect well to facility"""
           if facility_id in self.facilities:
               self.facilities[facility_id]['connected_wells'].append(well_id)
       
       def optimize_field_production(self):
           """Optimize production across all wells"""
           
           total_production = 0
           well_allocations = {}
           
           for well_id, well_model in self.wells.items():
               # Get well capability
               analysis = well_model.complete_system_analysis(
                   reservoir_pressure=3000,  # Would come from reservoir model
                   pi=2.0,
                   target_rate=1000
               )
               
               if analysis['artificial_lift']['lift_required']:
                   max_rate = analysis['target_rate']
               else:
                   max_rate = analysis['natural_flow']['rate']
               
               well_allocations[well_id] = {
                   'max_rate': max_rate,
                   'economics': analysis['economics']
               }
               
               total_production += max_rate
           
           # Check facility constraints
           for facility_id, facility in self.facilities.items():
               facility_production = sum(
                   well_allocations[wid]['max_rate'] 
                   for wid in facility['connected_wells']
               )
               
               if facility_production > facility['capacity']:
                   # Need to allocate
                   self.allocate_production(
                       facility, 
                       well_allocations
                   )
           
           return {
               'total_field_production': total_production,
               'well_allocations': well_allocations
           }
       
       def allocate_production(self, facility, well_allocations):
           """Allocate production when facility is constrained"""
           
           # Sort wells by economics (NPV per barrel)
           connected_wells = facility['connected_wells']
           
           well_values = []
           for wid in connected_wells:
               alloc = well_allocations[wid]
               value_per_bbl = alloc['economics']['npv_5yr'] / (alloc['max_rate'] * 365 * 5)
               well_values.append((wid, value_per_bbl, alloc['max_rate']))
           
           # Sort by value (highest first)
           well_values.sort(key=lambda x: x[1], reverse=True)
           
           # Allocate capacity
           remaining_capacity = facility['capacity']
           
           for wid, value, max_rate in well_values:
               allocated = min(max_rate, remaining_capacity)
               well_allocations[wid]['allocated_rate'] = allocated
               remaining_capacity -= allocated
               
               if remaining_capacity <= 0:
                   break

----

Part 2: Automated Analysis Pipelines
-------------------------------------

2.1 Daily Production Surveillance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Automated Monitoring:**

.. code-block:: python

   class ProductionSurveillanceSystem:
       """
       Automated production surveillance and anomaly detection.
       """
       
       def __init__(self):
           self.wells = {}
           self.alerts = []
       
       def ingest_production_data(self, well_id, date, data):
           """Ingest daily production data"""
           if well_id not in self.wells:
               self.wells[well_id] = {'history': []}
           
           self.wells[well_id]['history'].append({
               'date': date,
               'oil_rate': data['oil_rate'],
               'water_rate': data['water_rate'],
               'gas_rate': data['gas_rate'],
               'wellhead_pressure': data['whp'],
               'choke_size': data['choke']
           })
       
       def detect_anomalies(self, well_id):
           """Detect production anomalies"""
           
           if well_id not in self.wells:
               return []
           
           history = self.wells[well_id]['history']
           
           if len(history) < 7:
               return []  # Need baseline
           
           anomalies = []
           
           # Get recent data
           recent = history[-1]
           baseline = history[-7:-1]  # Last 6 days
           
           # Calculate baselines
           avg_oil = np.mean([d['oil_rate'] for d in baseline])
           avg_water = np.mean([d['water_rate'] for d in baseline])
           avg_whp = np.mean([d['wellhead_pressure'] for d in baseline])
           
           # Check for anomalies
           
           # 1. Sudden production drop
           if recent['oil_rate'] < avg_oil * 0.7:
               anomalies.append({
                   'type': 'PRODUCTION_DROP',
                   'severity': 'HIGH',
                   'message': f"Oil rate dropped {(1 - recent['oil_rate']/avg_oil)*100:.0f}%",
                   'actions': [
                       'Check for ESP failure',
                       'Inspect flowline for blockage',
                       'Review well test data'
                   ]
               })
           
           # 2. Water breakthrough
           current_wc = recent['water_rate'] / (recent['oil_rate'] + recent['water_rate']) * 100
           baseline_wc = avg_water / (avg_oil + avg_water) * 100
           
           if current_wc > baseline_wc + 10:
               anomalies.append({
                   'type': 'WATER_BREAKTHROUGH',
                   'severity': 'MEDIUM',
                   'message': f"Water cut increased to {current_wc:.0f}%",
                   'actions': [
                       'Consider water shutoff',
                       'Review completion integrity',
                       'Update reservoir model'
                   ]
               })
           
           # 3. Pressure anomaly
           if recent['wellhead_pressure'] > avg_whp * 1.2:
               anomalies.append({
                   'type': 'HIGH_PRESSURE',
                   'severity': 'HIGH',
                   'message': f"WHP increased to {recent['wellhead_pressure']:.0f} psi",
                   'actions': [
                       'Check for flowline restriction',
                       'Inspect choke',
                       'Verify separator pressure'
                   ]
               })
           
           return anomalies
       
       def generate_daily_report(self):
           """Generate automated daily report"""
           
           report = {
               'date': 'today',
               'total_oil': 0,
               'total_water': 0,
               'total_gas': 0,
               'wells_online': 0,
               'wells_offline': 0,
               'alerts': []
           }
           
           for well_id, well_data in self.wells.items():
               if well_data['history']:
                   latest = well_data['history'][-1]
                   
                   report['total_oil'] += latest['oil_rate']
                   report['total_water'] += latest['water_rate']
                   report['total_gas'] += latest['gas_rate']
                   
                   if latest['oil_rate'] > 0:
                       report['wells_online'] += 1
                   else:
                       report['wells_offline'] += 1
                   
                   # Check for anomalies
                   anomalies = self.detect_anomalies(well_id)
                   for anomaly in anomalies:
                       report['alerts'].append({
                           'well_id': well_id,
                           **anomaly
                       })
           
           return report
       
       def prioritize_workovers(self):
           """Prioritize wells for workover based on economics"""
           
           workover_candidates = []
           
           for well_id, well_data in self.wells.items():
               anomalies = self.detect_anomalies(well_id)
               
               if anomalies:
                   # Estimate production loss
                   history = well_data['history']
                   if len(history) >= 7:
                       baseline_rate = np.mean([d['oil_rate'] for d in history[-7:-1]])
                       current_rate = history[-1]['oil_rate']
                       loss = baseline_rate - current_rate
                       
                       # Value of fixing
                       oil_price = 70  # $/bbl
                       days_to_fix = 30  # Assumed
                       value = loss * days_to_fix * oil_price
                       
                       # Workover cost estimate
                       wo_cost = 150000  # Typical
                       
                       # NPV of workover
                       npv = value - wo_cost
                       
                       workover_candidates.append({
                           'well_id': well_id,
                           'production_loss': loss,
                           'workover_value': value,
                           'workover_cost': wo_cost,
                           'npv': npv,
                           'anomalies': anomalies
                       })
           
           # Sort by NPV
           workover_candidates.sort(key=lambda x: x['npv'], reverse=True)
           
           return workover_candidates
   
   # Example usage
   surveillance = ProductionSurveillanceSystem()
   
   # Simulate data ingestion
   for day in range(10):
       surveillance.ingest_production_data(
           'WELL-001',
           f'2024-01-{day+1:02d}',
           {
               'oil_rate': 500 - day * 20 if day > 5 else 500,  # Production drop on day 6
               'water_rate': 100,
               'gas_rate': 200000,
               'whp': 350 + day * 10 if day > 7 else 350,  # Pressure increase
               'choke': 32
           }
       )
   
   # Detect anomalies
   anomalies = surveillance.detect_anomalies('WELL-001')
   
   print("\n=== PRODUCTION SURVEILLANCE ===")
   print(f"\nAnomalies Detected: {len(anomalies)}")
   for i, anom in enumerate(anomalies, 1):
       print(f"\n{i}. {anom['type']} (Severity: {anom['severity']})")
       print(f"   {anom['message']}")
       print(f"   Recommended Actions:")
       for action in anom['actions']:
           print(f"   - {action}")
   
   # Workover prioritization
   workovers = surveillance.prioritize_workovers()
   
   print("\n=== WORKOVER PRIORITIZATION ===")
   for i, wo in enumerate(workovers, 1):
       print(f"\n{i}. {wo['well_id']}")
       print(f"   Production Loss: {wo['production_loss']:.0f} STB/day")
       print(f"   Workover Value: ${wo['workover_value']:,.0f}")
       print(f"   Workover Cost: ${wo['workover_cost']:,.0f}")
       print(f"   NPV: ${wo['npv']:,.0f}")
       print(f"   {'✅ RECOMMEND' if wo['npv'] > 0 else '❌ NOT ECONOMIC'}")

2.2 Decline Curve Analysis Automation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Automated DCA:**

.. code-block:: python

   from scipy.optimize import curve_fit
   
   class AutomatedDCA:
       """Automated decline curve analysis"""
       
       @staticmethod
       def fit_exponential_decline(time_days, rate_stb):
           """Fit exponential decline model"""
           
           def exponential(t, qi, di):
               return qi * np.exp(-di * t / 365)
           
           try:
               params, _ = curve_fit(
                   exponential, 
                   time_days, 
                   rate_stb,
                   p0=[rate_stb[0], 0.15],
                   bounds=([0, 0], [rate_stb[0]*2, 1.0])
               )
               qi, di = params
               
               return {
                   'model': 'exponential',
                   'qi': qi,
                   'di': di,
                   'fit_quality': 'good'
               }
           except Exception:
               return None
       
       @staticmethod
       def forecast_production(decline_params, forecast_years):
           """Forecast future production"""
           
           time = np.linspace(0, forecast_years * 365, forecast_years * 12)
           
           qi = decline_params['qi']
           di = decline_params['di']
           
           rates = qi * np.exp(-di * time / 365)
           
           # Calculate EUR
           eur = qi / di * 365 * (1 - np.exp(-di * forecast_years))
           
           return {
               'time_days': time,
               'rates': rates,
               'eur': eur,
               'forecast_years': forecast_years
           }
   
   # Example
   # Simulated production history
   time = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 365])
   rates = np.array([1000, 950, 900, 860, 820, 780, 750, 720, 690, 665, 640, 620, 600])
   
   dca = AutomatedDCA()
   
   # Fit decline
   decline = dca.fit_exponential_decline(time, rates)
   
   print("\n=== AUTOMATED DECLINE CURVE ANALYSIS ===")
   print(f"Model: {decline['model'].title()}")
   print(f"Initial Rate (qi): {decline['qi']:.0f} STB/day")
   print(f"Decline Rate (Di): {decline['di']*100:.1f}% per year")
   
   # Forecast
   forecast = dca.forecast_production(decline, forecast_years=5)
   
   print(f"\n=== 5-YEAR FORECAST ===")
   print(f"EUR: {forecast['eur']:,.0f} STB")
   print(f"Rate in 1 year: {forecast['rates'][12]:.0f} STB/day")
   print(f"Rate in 5 years: {forecast['rates'][-1]:.0f} STB/day")

----

Part 3: Field Development Strategy
-----------------------------------

3.1 Well Spacing Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Optimal Spacing:**

.. code-block:: python

   def optimize_well_spacing(
       reservoir_area_acres,
       reservoir_permeability_md,
       well_cost_mm,
       oil_price_bbl
   ):
       """
       Optimize well spacing for maximum NPV.
       """
       
       spacings = [40, 80, 160, 320, 640]  # Acres per well
       results = []
       
       for spacing in spacings:
           # Number of wells
           num_wells = reservoir_area_acres / spacing
           
           # Drainage radius
           drainage_radius_ft = np.sqrt(spacing * 43560 / np.pi)
           
           # Estimate EUR per well (simplified)
           # Higher density = more wells but lower EUR per well (interference)
           base_eur = 500000  # STB
           interference_factor = 1.0 - (640 / spacing - 1) * 0.1
           interference_factor = max(0.5, interference_factor)
           
           eur_per_well = base_eur * interference_factor
           
           # Economics
           total_eur = eur_per_well * num_wells
           revenue = total_eur * oil_price_bbl
           cost = num_wells * well_cost_mm * 1e6
           npv = revenue - cost
           
           results.append({
               'spacing_acres': spacing,
               'num_wells': num_wells,
               'eur_per_well': eur_per_well,
               'total_eur': total_eur,
               'revenue_mm': revenue / 1e6,
               'cost_mm': cost / 1e6,
               'npv_mm': npv / 1e6
           })
       
       # Find optimal
       results.sort(key=lambda x: x['npv_mm'], reverse=True)
       
       return results
   
   # Example
   results = optimize_well_spacing(
       reservoir_area_acres=10000,
       reservoir_permeability_md=50,
       well_cost_mm=5.0,
       oil_price_bbl=70
   )
   
   print("\n=== WELL SPACING OPTIMIZATION ===")
   print(f"{'Spacing':<12} {'Wells':<8} {'EUR/Well':<12} {'Total EUR':<15} {'NPV':<12}")
   print(f"{'(acres)':<12} {'(#)':<8} {'(MSTB)':<12} {'(MMSTB)':<15} {'($MM)':<12}")
   print("-" * 70)
   
   for r in results:
       print(f"{r['spacing_acres']:<12} "
             f"{r['num_wells']:<8.0f} "
             f"{r['eur_per_well']/1000:<12.0f} "
             f"{r['total_eur']/1e6:<15.1f} "
             f"{r['npv_mm']:<12.0f}")
   
   print(f"\n✅ Optimal Spacing: {results[0]['spacing_acres']} acres")
   print(f"   NPV: ${results[0]['npv_mm']:.0f} MM")

3.2 Development Sequencing
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Phased Development:**

.. code-block:: python

   class DevelopmentPlanner:
       """Plan phased field development"""
       
       def __init__(self, field_name, total_wells):
           self.field_name = field_name
           self.total_wells = total_wells
       
       def create_development_phases(self, wells_per_year):
           """Create phased development plan"""
           
           phases = []
           remaining_wells = self.total_wells
           year = 1
           
           while remaining_wells > 0:
               wells_this_phase = min(wells_per_year, remaining_wells)
               
               phases.append({
                   'phase': year,
                   'wells': wells_this_phase,
                   'start_year': year,
                   'capex_mm': wells_this_phase * 5.0,  # $5MM per well
                   'production_start': year + 0.5  # Mid-year startup
               })
               
               remaining_wells -= wells_this_phase
               year += 1
           
           return phases
       
       def calculate_phase_economics(self, phases):
           """Calculate economics for each phase"""
           
           oil_price = 70
           opex_per_bbl = 15
           discount_rate = 0.10
           
           cashflows = []
           
           for phase in phases:
               # Production profile (simplified)
               phase_cashflow = []
               
               for year in range(20):  # 20 year life
                   if year < phase['start_year']:
                       phase_cashflow.append(0)
                   else:
                       # Production decline
                       years_online = year - phase['start_year'] + 1
                       initial_rate_per_well = 500  # STB/day
                       decline = 0.85 ** (years_online - 1)  # 15% annual decline
                       
                       rate = phase['wells'] * initial_rate_per_well * decline
                       annual_prod = rate * 365
                       
                       revenue = annual_prod * oil_price
                       opex = annual_prod * opex_per_bbl
                       cashflow = revenue - opex
                       
                       phase_cashflow.append(cashflow)
               
               # NPV
               npv = -phase['capex_mm'] * 1e6
               for year, cf in enumerate(cashflows, 1):
                   npv += cf / (1 + discount_rate) ** year
               
               phase['npv_mm'] = npv / 1e6
               phase['cashflows'] = phase_cashflow
               
               cashflows.append(phase_cashflow)
           
           return phases
   
   # Example
   planner = DevelopmentPlanner('East Field', total_wells=30)
   
   phases = planner.create_development_phases(wells_per_year=10)
   
   print("\n=== FIELD DEVELOPMENT PLAN ===")
   print(f"Field: {planner.field_name}")
   print(f"Total Wells: {planner.total_wells}")
   print(f"\n{'Phase':<8} {'Year':<8} {'Wells':<8} {'Capex ($MM)':<15}")
   print("-" * 45)
   
   for phase in phases:
       print(f"{phase['phase']:<8} "
             f"{phase['start_year']:<8} "
             f"{phase['wells']:<8} "
             f"{phase['capex_mm']:<15.1f}")

----

Summary and Key Takeaways
--------------------------

**What You Learned:**

✅ Integrated reservoir-to-surface modeling  
✅ Multi-well field optimization  
✅ Automated surveillance systems  
✅ Decline curve automation  
✅ Field development strategy  
✅ Well spacing optimization  
✅ Development sequencing  

**Integration Benefits:**

- **15-30%** production increase
- **20-40%** cost reduction  
- **50%+** faster decisions
- **Better** ultimate recovery
- **Reduced** operational risk

**Key Principles:**

1. **Systems thinking** - Optimize the whole, not parts
2. **Data integration** - Single source of truth
3. **Automation** - Reduce manual work
4. **Real-time** - Decisions based on current data
5. **Economics** - Always value-focused

**Next Steps:**

- Apply integrated workflows to your fields
- Build automation for repetitive tasks
- Develop cross-functional teams
- Invest in data infrastructure
- Implement real-time surveillance

----

**Module Complete!** ✅

You now understand how to build integrated petroleum engineering workflows.

**Next:** :doc:`module9_optimization`
