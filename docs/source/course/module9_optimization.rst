Module 9: Optimization and Machine Learning
============================================

.. meta::
   :description: Apply optimization algorithms and machine learning to petroleum engineering problems
   :keywords: optimization, machine learning, AI, production optimization, drilling optimization

**Learning Objectives**

After completing this module, you will be able to:

- Apply optimization algorithms to petroleum engineering problems
- Use linear and nonlinear programming for resource allocation
- Implement machine learning models for prediction tasks
- Build neural networks for complex pattern recognition
- Apply genetic algorithms for well placement
- Optimize production operations in real-time
- Forecast production using ML techniques
- Implement digital twin concepts with ML

**Time Commitment:** 8-10 hours

**Prerequisites:** Modules 1-8, Basic Python, NumPy

----

Introduction: The Optimization Revolution
------------------------------------------

**Traditional vs. Optimized Approach:**

.. code-block:: text

   TRADITIONAL ENGINEERING
   ┌──────────────────────────────────────┐
   │ 1. Engineer makes assumptions        │
   │ 2. Calculates a few scenarios        │
   │ 3. Picks "reasonable" solution       │
   │ 4. Implements                        │
   └──────────────────────────────────────┘
   Result: Satisfactory (but suboptimal)
   
   OPTIMIZATION-DRIVEN ENGINEERING
   ┌──────────────────────────────────────┐
   │ 1. Define objective function         │
   │ 2. Set constraints                   │
   │ 3. Run optimizer (1000s scenarios)   │
   │ 4. Implement optimal solution        │
   └──────────────────────────────────────┘
   Result: Provably optimal (or near-optimal)

**Value Creation:**

Real-world examples:
- **Production optimization**: +5-15% through continuous tuning
- **Well placement**: +20-30% NPV vs. manual design
- **Drilling optimization**: -15-25% cost through parameter optimization
- **Artificial lift**: +10-20% efficiency through ML tuning

**This Module's Approach:**

We'll progress from simple to advanced:

1. **Linear programming** (LP) - Production allocation
2. **Nonlinear programming** (NLP) - Well spacing
3. **Genetic algorithms** (GA) - Well placement
4. **Machine learning** - Production forecasting
5. **Neural networks** - Complex pattern recognition

----

Part 1: Linear Programming
---------------------------

1.1 Production Allocation Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem:**

You have multiple wells and limited facility capacity. How do you allocate production to maximize revenue?

**Formulation:**

.. math::

   \\text{Maximize: } \\sum_{i=1}^{n} (revenue_i - cost_i) \\cdot production_i

.. math::

   \\text{Subject to:}

.. math::

   \\sum_{i=1}^{n} production_i \\leq facility\\_capacity

.. math::

   0 \\leq production_i \\leq max\\_rate_i

**Python Implementation:**

.. code-block:: python

   from scipy.optimize import linprog
   import numpy as np
   
   def optimize_production_allocation(
       wells,
       facility_capacity_bpd,
       oil_price_bbl=70,
       gas_price_mscf=3.0
   ):
       """
       Optimize production allocation across wells.
       
       Args:
           wells: List of dicts with well parameters
           facility_capacity_bpd: Total facility capacity
           oil_price_bbl: Oil price
           gas_price_mscf: Gas price
       
       Returns:
           Optimal production rates for each well
       """
       
       n_wells = len(wells)
       
       # Objective function coefficients (negative for maximization)
       c = []
       for well in wells:
           # Revenue per day
           oil_revenue = well['max_oil_rate'] * oil_price_bbl
           gas_revenue = well['max_gas_rate'] / 1000 * gas_price_mscf
           opex = well['max_oil_rate'] * well['opex_per_bbl']
           
           # Net value (negative for linprog minimization)
           net_value = -(oil_revenue + gas_revenue - opex)
           c.append(net_value)
       
       # Inequality constraints: sum(production) <= capacity
       A_ub = [[well['max_oil_rate'] for well in wells]]
       b_ub = [facility_capacity_bpd]
       
       # Bounds: 0 <= rate_fraction <= 1
       bounds = [(0, 1) for _ in range(n_wells)]
       
       # Solve
       result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
       
       if result.success:
           # Convert rate fractions to actual rates
           optimal_rates = []
           for i, well in enumerate(wells):
               rate_fraction = result.x[i]
               oil_rate = well['max_oil_rate'] * rate_fraction
               gas_rate = well['max_gas_rate'] * rate_fraction
               
               optimal_rates.append({
                   'well_id': well['well_id'],
                   'oil_rate': oil_rate,
                   'gas_rate': gas_rate,
                   'rate_fraction': rate_fraction
               })
           
           return {
               'success': True,
               'optimal_rates': optimal_rates,
               'total_oil': sum(r['oil_rate'] for r in optimal_rates),
               'total_value': -result.fun,
               'constrained': result.x.sum() * wells[0]['max_oil_rate'] >= facility_capacity_bpd
           }
       else:
           return {
               'success': False,
               'message': result.message
           }
   
   # Example: Optimize production allocation
   wells = [
       {
           'well_id': 'WELL-001',
           'max_oil_rate': 500,  # STB/day
           'max_gas_rate': 250000,  # scf/day
           'opex_per_bbl': 12
       },
       {
           'well_id': 'WELL-002',
           'max_oil_rate': 800,
           'max_gas_rate': 400000,
           'opex_per_bbl': 15
       },
       {
           'well_id': 'WELL-003',
           'max_oil_rate': 300,
           'max_gas_rate': 600000,  # High GOR well
           'opex_per_bbl': 10
       },
       {
           'well_id': 'WELL-004',
           'max_oil_rate': 1000,
           'max_gas_rate': 300000,
           'opex_per_bbl': 18  # Higher opex
       }
   ]
   
   result = optimize_production_allocation(
       wells,
       facility_capacity_bpd=2000  # Limited capacity
   )
   
   print("\n" + "="*60)
   print("    PRODUCTION ALLOCATION OPTIMIZATION")
   print("="*60)
   
   if result['success']:
       print(f"\nFacility Capacity: 2000 BPD")
       print(f"Total Allocated: {result['total_oil']:.0f} BPD")
       print(f"Total Daily Value: ${result['total_value']:,.0f}")
       print(f"Constrained: {'YES' if result['constrained'] else 'NO'}")
       
       print(f"\n{'Well ID':<12} {'Oil Rate':<12} {'Gas Rate':<15} {'% of Max':<12}")
       print("-" * 55)
       
       for r in result['optimal_rates']:
           print(f"{r['well_id']:<12} "
                 f"{r['oil_rate']:<12.0f} "
                 f"{r['gas_rate']:<15.0f} "
                 f"{r['rate_fraction']*100:<12.0f}%")
       
       print("\n✅ Optimal allocation found!")
   else:
       print(f"❌ Optimization failed: {result['message']}")

1.2 Gas Lift Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem:**

Allocate limited gas injection capacity across multiple gas lift wells to maximize total oil production.

.. code-block:: python

   from scipy.optimize import minimize
   
   def optimize_gas_lift_allocation(wells, total_gas_available_mscfd):
       """
       Optimize gas lift allocation using nonlinear programming.
       
       Gas lift response is nonlinear: q_oil = f(q_gas)
       """
       
       def gas_lift_response(gas_rate_mscfd, well_params):
           """Model oil response to gas injection (empirical)"""
           qg = gas_rate_mscfd
           a = well_params['a']  # Response coefficient
           b = well_params['b']  # Diminishing returns factor
           
           # Oil rate increases with gas, but with diminishing returns
           q_oil = a * qg / (1 + b * qg)
           return q_oil
       
       def objective(gas_rates):
           """Total oil production (negative for minimization)"""
           total_oil = 0
           for i, well in enumerate(wells):
               oil = gas_lift_response(gas_rates[i], well)
               total_oil += oil
           return -total_oil  # Negative for maximization
       
       def constraint_total_gas(gas_rates):
           """Total gas must not exceed available"""
           return total_gas_available_mscfd - sum(gas_rates)
       
       # Initial guess: equal allocation
       x0 = [total_gas_available_mscfd / len(wells)] * len(wells)
       
       # Bounds: 0 to max capacity per well
       bounds = [(0, well.get('max_gas_injection', 2000)) for well in wells]
       
       # Constraints
       constraints = [
           {'type': 'ineq', 'fun': constraint_total_gas}
       ]
       
       # Optimize
       result = minimize(
           objective,
           x0,
           method='SLSQP',
           bounds=bounds,
           constraints=constraints
       )
       
       if result.success:
           optimal_allocation = []
           for i, well in enumerate(wells):
               gas_rate = result.x[i]
               oil_rate = gas_lift_response(gas_rate, well)
               
               optimal_allocation.append({
                   'well_id': well['well_id'],
                   'gas_injection_mscfd': gas_rate,
                   'oil_rate_stb': oil_rate,
                   'gor_incremental': gas_rate * 1000 / oil_rate if oil_rate > 0 else 0
               })
           
           return {
               'success': True,
               'allocation': optimal_allocation,
               'total_oil': -result.fun,
               'total_gas_used': sum(result.x)
           }
       else:
           return {'success': False}
   
   # Example: Gas lift optimization
   gl_wells = [
       {'well_id': 'GL-001', 'a': 800, 'b': 0.0005, 'max_gas_injection': 1500},
       {'well_id': 'GL-002', 'a': 1000, 'b': 0.0008, 'max_gas_injection': 2000},
       {'well_id': 'GL-003', 'a': 600, 'b': 0.0003, 'max_gas_injection': 1000},
   ]
   
   gl_result = optimize_gas_lift_allocation(gl_wells, total_gas_available_mscfd=3000)
   
   print("\n" + "="*60)
   print("       GAS LIFT ALLOCATION OPTIMIZATION")
   print("="*60)
   
   if gl_result['success']:
       print(f"\nTotal Gas Available: {3000:.0f} Mscf/day")
       print(f"Total Gas Allocated: {gl_result['total_gas_used']:.0f} Mscf/day")
       print(f"Total Oil Production: {gl_result['total_oil']:.0f} STB/day")
       
       print(f"\n{'Well ID':<12} {'Gas Inj':<15} {'Oil Rate':<15} {'Inc GOR':<12}")
       print(f"{'':12} {'(Mscf/day)':<15} {'(STB/day)':<15} {'(scf/STB)':<12}")
       print("-" * 60)
       
       for alloc in gl_result['allocation']:
           print(f"{alloc['well_id']:<12} "
                 f"{alloc['gas_injection_mscfd']:<15.0f} "
                 f"{alloc['oil_rate_stb']:<15.0f} "
                 f"{alloc['gor_incremental']:<12.0f}")
   else:
       print("❌ Optimization failed")

----

Part 2: Genetic Algorithms
---------------------------

2.1 Well Placement Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem:**

Find optimal locations for N wells in a reservoir to maximize NPV.

**Genetic Algorithm Approach:**

1. **Initialize** population of random well placements
2. **Evaluate** fitness (NPV) of each configuration
3. **Select** best configurations for breeding
4. **Crossover** combine parent configurations
5. **Mutate** introduce random changes
6. **Repeat** until convergence

**Implementation:**

.. code-block:: python

   import random
   
   class WellPlacementGA:
       """
       Genetic algorithm for well placement optimization.
       """
       
       def __init__(self, reservoir_bounds, n_wells, population_size=50):
           """
           Args:
               reservoir_bounds: (xmin, xmax, ymin, ymax) in ft
               n_wells: Number of wells to place
               population_size: GA population size
           """
           self.bounds = reservoir_bounds
           self.n_wells = n_wells
           self.population_size = population_size
           self.population = []
           self.generation = 0
       
       def initialize_population(self):
           """Create random initial population"""
           xmin, xmax, ymin, ymax = self.bounds
           
           for _ in range(self.population_size):
               # Random well locations
               individual = []
               for _ in range(self.n_wells):
                   x = random.uniform(xmin, xmax)
                   y = random.uniform(ymin, ymax)
                   individual.append((x, y))
               
               self.population.append(individual)
       
       def evaluate_fitness(self, individual):
           """
           Evaluate fitness (NPV) of well configuration.
           
           Simplified model:
           - Wells produce from drainage area
           - Interference reduces production
           - NPV = revenue - costs
           """
           
           well_cost = 5_000_000  # $5MM per well
           oil_price = 70  # $/bbl
           
           total_production = 0
           
           for i, (x1, y1) in enumerate(individual):
               # Base EUR per well
               base_eur = 500_000  # STB
               
               # Check interference from other wells
               interference_factor = 1.0
               
               for j, (x2, y2) in enumerate(individual):
                   if i != j:
                       distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                       
                       # Interference increases as wells get closer
                       if distance < 2000:  # ft
                           interference = 0.2 * (1 - distance / 2000)
                           interference_factor -= interference
               
               interference_factor = max(0.3, interference_factor)
               
               well_eur = base_eur * interference_factor
               total_production += well_eur
           
           # NPV calculation
           revenue = total_production * oil_price
           costs = self.n_wells * well_cost
           npv = revenue - costs
           
           return npv
       
       def select_parents(self, fitnesses):
           """Select parents using tournament selection"""
           tournament_size = 5
           
           # Tournament selection
           competitors = random.sample(
               list(zip(self.population, fitnesses)), 
               tournament_size
           )
           winner = max(competitors, key=lambda x: x[1])
           return winner[0]
       
       def crossover(self, parent1, parent2):
           """Combine two parents to create offspring"""
           # Single-point crossover
           crossover_point = random.randint(1, self.n_wells - 1)
           
           child1 = parent1[:crossover_point] + parent2[crossover_point:]
           child2 = parent2[:crossover_point] + parent1[crossover_point:]
           
           return child1, child2
       
       def mutate(self, individual, mutation_rate=0.1):
           """Randomly modify individual"""
           xmin, xmax, ymin, ymax = self.bounds
           
           mutated = []
           for x, y in individual:
               if random.random() < mutation_rate:
                   # Mutate location
                   x = random.uniform(xmin, xmax)
                   y = random.uniform(ymin, ymax)
               mutated.append((x, y))
           
           return mutated
       
       def evolve(self, generations=100):
           """Run genetic algorithm"""
           
           # Initialize
           if not self.population:
               self.initialize_population()
           
           best_fitness_history = []
           
           for gen in range(generations):
               # Evaluate fitness
               fitnesses = [self.evaluate_fitness(ind) for ind in self.population]
               
               # Track best
               best_idx = np.argmax(fitnesses)
               best_fitness = fitnesses[best_idx]
               best_individual = self.population[best_idx]
               
               best_fitness_history.append(best_fitness)
               
               # Create next generation
               new_population = []
               
               # Elitism: keep best individual
               new_population.append(best_individual)
               
               # Generate rest of population
               while len(new_population) < self.population_size:
                   # Select parents
                   parent1 = self.select_parents(fitnesses)
                   parent2 = self.select_parents(fitnesses)
                   
                   # Crossover
                   child1, child2 = self.crossover(parent1, parent2)
                   
                   # Mutate
                   child1 = self.mutate(child1)
                   child2 = self.mutate(child2)
                   
                   new_population.extend([child1, child2])
               
               # Trim to population size
               self.population = new_population[:self.population_size]
               self.generation += 1
               
               # Progress report
               if (gen + 1) % 20 == 0:
                   print(f"Generation {gen+1}: Best NPV = ${best_fitness/1e6:.2f} MM")
           
           # Final result
           fitnesses = [self.evaluate_fitness(ind) for ind in self.population]
           best_idx = np.argmax(fitnesses)
           
           return {
               'best_configuration': self.population[best_idx],
               'best_npv': fitnesses[best_idx],
               'fitness_history': best_fitness_history
           }
   
   # Example: Optimize well placement
   print("\n" + "="*60)
   print("      WELL PLACEMENT OPTIMIZATION (GA)")
   print("="*60)
   
   ga = WellPlacementGA(
       reservoir_bounds=(0, 10000, 0, 8000),  # 10,000 x 8,000 ft
       n_wells=5,
       population_size=30
   )
   
   result = ga.evolve(generations=100)
   
   print(f"\n✅ Optimization Complete!")
   print(f"Best NPV: ${result['best_npv']/1e6:.2f} MM")
   print(f"\nOptimal Well Locations:")
   print(f"{'Well':<8} {'X (ft)':<12} {'Y (ft)':<12}")
   print("-" * 35)
   
   for i, (x, y) in enumerate(result['best_configuration'], 1):
       print(f"Well-{i:<3} {x:<12.0f} {y:<12.0f}")

----

Part 3: Machine Learning for Production Forecasting
----------------------------------------------------

3.1 Linear Regression for Decline Curves
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sklearn.linear_model import LinearRegression
   from sklearn.preprocessing import PolynomialFeatures
   from sklearn.metrics import r2_score, mean_absolute_error
   
   class MLDeclineCurve:
       """Machine learning approach to decline curve analysis"""
       
       def __init__(self):
           self.model = None
           self.poly_features = None
       
       def fit(self, time_days, rate_stb, degree=2):
           """
           Fit decline curve using polynomial regression.
           
           Args:
               time_days: Array of time values
               rate_stb: Array of production rates
               degree: Polynomial degree (2 = quadratic)
           """
           
           # Create polynomial features
           self.poly_features = PolynomialFeatures(degree=degree)
           X_poly = self.poly_features.fit_transform(time_days.reshape(-1, 1))
           
           # Fit model
           self.model = LinearRegression()
           self.model.fit(X_poly, rate_stb)
           
           # Calculate fit quality
           predictions = self.model.predict(X_poly)
           r2 = r2_score(rate_stb, predictions)
           mae = mean_absolute_error(rate_stb, predictions)
           
           return {
               'r2_score': r2,
               'mae': mae,
               'model_fitted': True
           }
       
       def predict(self, time_days):
           """Predict production at given times"""
           if self.model is None:
               raise ValueError("Model not fitted yet")
           
           X_poly = self.poly_features.transform(time_days.reshape(-1, 1))
           predictions = self.model.predict(X_poly)
           
           # Ensure non-negative
           predictions = np.maximum(predictions, 0)
           
           return predictions
       
       def forecast_eur(self, forecast_years, economic_limit_stb=10):
           """
           Forecast EUR until economic limit.
           """
           days = np.arange(0, forecast_years * 365, 30)
           rates = self.predict(days)
           
           # Find when rate drops below economic limit
           economic_life_idx = np.where(rates < economic_limit_stb)[0]
           
           if len(economic_life_idx) > 0:
               economic_life_days = days[economic_life_idx[0]]
           else:
               economic_life_days = forecast_years * 365
           
           # Calculate EUR
           eur = np.trapz(rates[:len(days[days <= economic_life_days])], 
                          days[days <= economic_life_days])
           
           return {
               'eur_stb': eur,
               'economic_life_years': economic_life_days / 365,
               'final_rate_stb': rates[-1]
           }
   
   # Example: ML decline curve
   # Simulated production data
   time = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 365])
   rates = np.array([1000, 950, 900, 860, 820, 780, 750, 720, 690, 665, 640, 620, 600])
   
   ml_dca = MLDeclineCurve()
   
   fit_result = ml_dca.fit(time, rates, degree=2)
   
   print("\n" + "="*60)
   print("    ML DECLINE CURVE ANALYSIS")
   print("="*60)
   
   print(f"\nModel Fit Quality:")
   print(f"R² Score: {fit_result['r2_score']:.4f}")
   print(f"Mean Absolute Error: {fit_result['mae']:.1f} STB/day")
   
   # Forecast
   forecast = ml_dca.forecast_eur(forecast_years=10, economic_limit_stb=50)
   
   print(f"\n10-Year Forecast:")
   print(f"EUR: {forecast['eur_stb']:,.0f} STB")
   print(f"Economic Life: {forecast['economic_life_years']:.1f} years")
   print(f"Final Rate: {forecast['final_rate_stb']:.0f} STB/day")

3.2 Neural Networks for Complex Patterns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**When to use Neural Networks:**

✅ Complex nonlinear relationships  
✅ Multiple input features  
✅ Large datasets available  
✅ Pattern recognition needed  

**Example: Predict Well Performance from Multiple Features**

.. code-block:: python

   from sklearn.neural_network import MLPRegressor
   from sklearn.preprocessing import StandardScaler
   
   class WellPerformancePredictor:
       """
       Neural network to predict well performance from reservoir/completion data.
       """
       
       def __init__(self):
           self.model = MLPRegressor(
               hidden_layer_sizes=(20, 10),
               activation='relu',
               max_iter=1000,
               random_state=42
           )
           self.scaler = StandardScaler()
       
       def train(self, X_features, y_production):
           """
           Train neural network.
           
           Args:
               X_features: Array of shape (n_wells, n_features)
                          Features: permeability, porosity, net_pay, 
                                   completion_skin, fluid_viscosity, etc.
               y_production: Array of production rates
           """
           
           # Scale features
           X_scaled = self.scaler.fit_transform(X_features)
           
           # Train model
           self.model.fit(X_scaled, y_production)
           
           # Calculate training accuracy
           train_score = self.model.score(X_scaled, y_production)
           
           return {
               'training_score': train_score,
               'model_trained': True
           }
       
       def predict(self, X_features):
           """Predict production for new wells"""
           X_scaled = self.scaler.transform(X_features)
           predictions = self.model.predict(X_scaled)
           return predictions
   
   # Example: Train neural network
   # Simulated training data
   np.random.seed(42)
   n_wells = 100
   
   # Features: permeability, porosity, net_pay, skin, viscosity
   X_train = np.column_stack([
       np.random.uniform(50, 200, n_wells),    # Permeability (md)
       np.random.uniform(0.15, 0.30, n_wells), # Porosity
       np.random.uniform(30, 100, n_wells),    # Net pay (ft)
       np.random.uniform(-2, 10, n_wells),     # Skin
       np.random.uniform(1, 5, n_wells)        # Viscosity (cp)
   ])
   
   # Production rate (simplified relationship)
   y_train = (
       X_train[:, 0] * 2 +           # Permeability effect
       X_train[:, 1] * 1000 +        # Porosity effect
       X_train[:, 2] * 5 -           # Net pay effect
       X_train[:, 3] * 20 -          # Skin (negative effect)
       X_train[:, 4] * 50 +          # Viscosity (negative effect)
       np.random.normal(0, 50, n_wells)  # Noise
   )
   
   # Train model
   nn_model = WellPerformancePredictor()
   train_result = nn_model.train(X_train, y_train)
   
   print("\n" + "="*60)
   print("    NEURAL NETWORK WELL PERFORMANCE PREDICTOR")
   print("="*60)
   
   print(f"\nTraining Results:")
   print(f"Training R² Score: {train_result['training_score']:.4f}")
   
   # Predict for new well
   new_well = np.array([[
       150,   # Permeability (md)
       0.22,  # Porosity
       60,    # Net pay (ft)
       2.0,   # Skin
       2.5    # Viscosity (cp)
   ]])
   
   predicted_rate = nn_model.predict(new_well)
   
   print(f"\nPrediction for New Well:")
   print(f"Features:")
   print(f"  Permeability: 150 md")
   print(f"  Porosity: 22%")
   print(f"  Net Pay: 60 ft")
   print(f"  Skin: 2.0")
   print(f"  Viscosity: 2.5 cp")
   print(f"\nPredicted Production: {predicted_rate[0]:.0f} STB/day")

----

Part 4: Real-Time Optimization
-------------------------------

4.1 ESP Speed Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Real-time ESP optimization:**

.. code-block:: python

   class RealTimeESPOptimizer:
       """
       Real-time ESP speed optimization for maximum efficiency.
       """
       
       def __init__(self, well_id):
           self.well_id = well_id
           self.history = []
       
       def calculate_esp_efficiency(self, frequency_hz, measured_rate_bpd, power_kw):
           """Calculate ESP system efficiency"""
           
           # Hydraulic horsepower
           fluid_sg = 0.85
           tdh = 5000  # ft (would be calculated from system)
           
           hhp = (measured_rate_bpd * tdh * fluid_sg) / 3960
           
           # Electrical to hydraulic efficiency
           ehp = power_kw * 1.341  # Convert kW to HP
           
           if ehp > 0:
               efficiency = hhp / ehp
           else:
               efficiency = 0
           
           return efficiency
       
       def find_optimal_frequency(self, current_frequency_hz):
           """
           Find optimal ESP frequency using gradient descent.
           """
           
           # Test frequencies around current
           test_frequencies = [
               current_frequency_hz - 5,
               current_frequency_hz,
               current_frequency_hz + 5
           ]
           
           # Simulate measurements at each frequency
           # (In production, these would be real measurements)
           efficiencies = []
           rates = []
           
           for freq in test_frequencies:
               # Simplified model: rate increases with frequency
               rate = 500 * (freq / 60)  # Scaled to 60 Hz base
               
               # Power increases faster than rate
               power = 50 * (freq / 60) ** 1.3
               
               eff = self.calculate_esp_efficiency(freq, rate, power)
               
               efficiencies.append(eff)
               rates.append(rate)
           
           # Find maximum efficiency
           max_eff_idx = np.argmax(efficiencies)
           optimal_freq = test_frequencies[max_eff_idx]
           
           return {
               'optimal_frequency_hz': optimal_freq,
               'expected_efficiency': efficiencies[max_eff_idx],
               'expected_rate_bpd': rates[max_eff_idx],
               'frequency_change': optimal_freq - current_frequency_hz
           }
       
       def run_optimization_cycle(self, current_freq):
           """Run one optimization cycle"""
           
           result = self.find_optimal_frequency(current_freq)
           
           self.history.append({
               'frequency': result['optimal_frequency_hz'],
               'efficiency': result['expected_efficiency'],
               'rate': result['expected_rate_bpd']
           })
           
           return result
   
   # Example: Real-time ESP optimization
   optimizer = RealTimeESPOptimizer('WELL-ESP-001')
   
   print("\n" + "="*60)
   print("      REAL-TIME ESP OPTIMIZATION")
   print("="*60)
   
   current_freq = 60  # Hz
   
   for cycle in range(5):
       result = optimizer.run_optimization_cycle(current_freq)
       
       print(f"\nCycle {cycle + 1}:")
       print(f"  Current Frequency: {current_freq:.0f} Hz")
       print(f"  Optimal Frequency: {result['optimal_frequency_hz']:.0f} Hz")
       print(f"  Change: {result['frequency_change']:+.0f} Hz")
       print(f"  Expected Efficiency: {result['expected_efficiency']*100:.1f}%")
       print(f"  Expected Rate: {result['expected_rate_bpd']:.0f} BPD")
       
       # Update for next cycle
       current_freq = result['optimal_frequency_hz']

----

Summary and Key Takeaways
--------------------------

**What You Learned:**

✅ Linear programming for resource allocation  
✅ Nonlinear optimization for gas lift  
✅ Genetic algorithms for well placement  
✅ Machine learning for production forecasting  
✅ Neural networks for complex patterns  
✅ Real-time optimization techniques  

**When to Use Each Method:**

.. code-block:: text

   Problem Type                    Best Method
   ──────────────────────────────────────────────────────
   Production allocation           Linear programming
   Gas lift optimization           Nonlinear programming
   Well placement                  Genetic algorithm
   Production forecasting          Machine learning
   Complex patterns                Neural networks
   Real-time tuning                Gradient descent

**Key Principles:**

1. **Define objective clearly** - What are you maximizing?
2. **Understand constraints** - What limits exist?
3. **Choose right algorithm** - Match method to problem
4. **Validate results** - Does it make engineering sense?
5. **Monitor performance** - Continuous improvement

**Value Potential:**

- Production optimization: **+5-15%**
- Well placement: **+20-30% NPV**
- Artificial lift: **+10-20% efficiency**
- Drilling optimization: **-15-25% cost**

**Next Steps:**

- Apply optimization to your operations
- Build ML models with your data
- Implement real-time optimization
- Create digital twins

----

Practice Exercises
------------------

**Exercise 9.1:** Production Allocation

You have 5 wells with different productivities and a facility limit of 5000 BPD. Optimize allocation.

**Exercise 9.2:** ML Decline Curve

Use your well's production data to build an ML decline model and forecast EUR.

**Exercise 9.3:** ESP Optimization

Implement real-time frequency optimization for your ESP wells.

----

**Module Complete!** ✅

You now have powerful optimization and ML tools for petroleum engineering.

**Next:** :doc:`module10_field_development`
