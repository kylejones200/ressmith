Glossary
========

**Petroleum Engineering and PetroSmith Terms**

---

A
-

**API Gravity**
   American Petroleum Institute measure of oil density. Higher values indicate lighter oil.
   Formula: API = (141.5 / SG) - 131.5

**Arps Decline**
   Mathematical models (exponential, hyperbolic, harmonic) for production decline curves.

---

B
-

**Bo (Oil Formation Volume Factor)**
   Ratio of oil volume at reservoir conditions to stock tank conditions. Typical range: 1.0-2.5 rb/stb.

**BHP (Bottomhole Pressure)**
   Pressure at the bottom of the wellbore, either flowing (BHFP) or shut-in (BHSP).

---

C
-

**Completion**
   Equipment and procedures to prepare a well for production (perforations, tubing, packer).

---

D
-

**Darcy's Law**
   Fundamental equation describing fluid flow through porous media.

**Decline Curve Analysis (DCA)**
   Method to forecast production by extrapolating historical decline trends.

---

E
-

**ECD (Equivalent Circulating Density)**
   Effective mud weight while circulating, including annular friction losses.

**EUR (Estimated Ultimate Recovery)**
   Total hydrocarbons expected to be produced from a well over its life.

---

F
-

**FVF (Formation Volume Factor)**
   Relates reservoir volume to surface volume. Applies to oil (Bo), gas (Bg), water (Bw).

---

G
-

**GOR (Gas-Oil Ratio)**
   Standard cubic feet of gas produced per barrel of oil. Classification: <100 (black oil), 100-3000 (volatile oil), >3000 (gas condensate).

---

H
-

**Hydrostatic Pressure**
   Pressure exerted by fluid column. Formula: P = 0.052 × MW × TVD.

---

I
-

**IPR (Inflow Performance Relationship)**
   Relationship between production rate and bottomhole flowing pressure.

---

M
-

**Material Balance**
   Conservation of mass equation used to estimate OOIP and predict reservoir pressure.

**Mud Weight**
   Density of drilling fluid, typically measured in pounds per gallon (ppg).

---

O
-

**OOIP (Original Oil In Place)**
   Total oil in reservoir before production. Calculated using volumetric method.

---

P
-

**Permeability**
   Measure of rock's ability to transmit fluids, measured in millidarcies (md) or darcies (D).

**Porosity**
   Fraction of rock volume that is void space, can store fluids. Typical range: 5-35%.

**Productivity Index (PI)**
   Measure of well productivity: PI = q / (Pr - Pwf). Units: STB/day/psi.

---

R
-

**Recovery Factor**
   Fraction of OOIP that will be recovered. Typical: 5-70% depending on drive mechanism and recovery method.

---

S
-

**Skin Factor**
   Dimensionless measure of near-wellbore damage (positive) or stimulation (negative).

**Sw (Water Saturation)**
   Fraction of pore space filled with water. Hydrocarbon saturation = 1 - Sw.

---

T
-

**TVD (True Vertical Depth)**
   Vertical distance from surface to target, as opposed to measured depth (MD) along wellbore.

---

V
-

**VLP (Vertical Lift Performance)**
   Pressure losses in tubing string, function of rate, fluid properties, and tubing size.

**Volumetric Method**
   Technique to calculate OOIP/OGIP using reservoir volume, porosity, and saturations.

---

**PetroSmith-Specific Terms**

**API Layer (Layer 4)**
   High-level interfaces for end users (WellAPI, ReservoirAPI, etc.)

**Core Layer (Layer 2)**
   Pure calculation functions with no side effects

**Models Layer (Layer 1)**
   Pydantic data models representing petroleum assets

**Services Layer (Layer 3)**
   Orchestration and state management between core calculations

---

For complete definitions, see:

- SPE Glossary: https://www.spe.org/glossary/
- Schlumberger Oilfield Glossary: https://www.glossary.oilfield.slb.com/

---

**Next:** :doc:`references`
