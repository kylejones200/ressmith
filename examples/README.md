# PetroSmith Examples

This directory contains runnable examples demonstrating the PetroSmith library.

## Running the Examples

```bash
# From the project root directory
python examples/01_basic_drilling_calculations.py
python examples/02_well_control_kick_analysis.py
python examples/03_formation_evaluation.py
```

## Examples

### 1. Basic Drilling Calculations (`01_basic_drilling_calculations.py`)
- Hydrostatic pressure
- Equivalent Circulating Density (ECD)
- Casing design (burst/collapse)
- Input validation and error handling

**Output:** Calculations for a 10,000 ft well with 10.5 ppg mud

### 2. Well Control - Kick Analysis (`02_well_control_kick_analysis.py`)
- Kick detection from pit gain and flow changes
- Formation pressure calculation
- Kill mud weight determination
- Driller's Method kill procedure

**Output:** Complete well control response plan for a kick scenario

### 3. Formation Evaluation (`03_formation_evaluation.py`)
- Overburden stress estimation
- Pore pressure prediction (Eaton's method)
- Fracture gradient estimation
- Drilling window analysis
- Rock mechanics and wellbore stability

**Output:** Complete formation evaluation at 10,000 ft depth

## What These Examples Demonstrate

**Real-world scenarios** - Not toy problems  
**Complete workflows** - End-to-end calculations  
**Error handling** - Shows validation in action  
**Best practices** - Industry-standard methods  
**Clear output** - Easy to understand results  

## Next Steps

After running these examples, check out:
- `tests/` - Unit tests showing how to test calculations
- `README.md` - Full library documentation
- `petrosmith/core/` - Source code for all calculations
