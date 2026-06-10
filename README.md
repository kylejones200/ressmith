# PetroSmith - Petroleum Engineering Library

A modern Python library for petroleum engineering calculations and analysis. Built with Python 3.12+, featuring real tests, proper error handling, and clean architecture.

**Status**: Beta - Core calculations are stable and tested. Service layer and workflows are under active development.

[![CI](https://github.com/yourusername/petrosmith/workflows/CI/badge.svg)](https://github.com/yourusername/petrosmith/actions)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Development Status](https://img.shields.io/badge/status-beta-yellow.svg)](https://github.com/yourusername/petrosmith)

---

## Config-Driven Workflows

PetroSmith supports config-driven workflows - run analyses from YAML files without writing code.

```bash
# Generate a config template
python -m petrosmith.cli.main init --output my_analysis.yaml

# Edit the config, then run
python -m petrosmith.cli.main run --config my_analysis.yaml
```

**Features:**
- Run analyses from YAML/JSON configs
- Reproducible workflows
- CLI and Python API
- 4 templates: basic, full, drilling, reservoir
- Built-in validation
- Batch processing and parameter sweeps

**Documentation:**
- [Complete Guide](CONFIG_WORKFLOW_GUIDE.md) - Full documentation
- [Quick Reference](QUICK_REFERENCE.md) - Command cheat sheet  
- [Examples](examples/configs/README.md) - Working examples

---

## Quick Start

### Traditional Python API
```bash
# Install dependencies
uv sync

# Run an example
uv run python examples/01_basic_drilling_calculations.py

# Run tests
uv run pytest tests/ -v
```

### Config-Driven Workflow
```bash
# Install
uv sync

# Generate config
uv run python -m petrosmith.cli.main init --output my_config.yaml

# Run analysis
uv run python -m petrosmith.cli.main run --config my_config.yaml
```

---

## What's Included

### Core Modules

#### **Drilling Engineering**
- Hydrostatic pressure & ECD calculations
- Casing design (burst/collapse)
- Surge & swab pressures
- Bit hydraulics

#### **Well Control (WELCON)**
- Kick detection & analysis
- Formation pressure calculation
- Kill procedures (Driller's Method, Wait & Weight)
- Gas migration modeling

#### **Drilling Fluids**
- Mud weight calculations
- Rheology models (Bingham, Power Law)
- Hydraulics & pressure loss
- Gel strength analysis

#### **Formation Pressure**
- Pore pressure prediction (Eaton's method)
- Overburden stress estimation
- Fracture gradient calculation
- Drilling window analysis

#### **Rock Mechanics**
- Elastic properties from logs
- Rock strength correlations
- In-situ stress estimation
- Wellbore stability analysis

#### **Subsea Drilling**
- Riser analysis & tension
- Deepwater mud management
- Kick tolerance calculations
- Dual gradient drilling

#### **Reservoir & Production**
- OOIP/OGIP calculations
- Material balance
- Flow rate calculations
- Production analysis

#### **Optional: Decline curve analysis (DCA)**
- Install with: `pip install petrosmith[dca]`
- Uses the [decline-curve](https://pypi.org/project/decline-curve/) library for Arps (exponential, hyperbolic, harmonic) and optional ML forecasting
- `ProductionAPI.forecast_production()` uses it automatically when available; otherwise falls back to built-in exponential decline

---

## Usage Examples

### Example 1: Hydrostatic Pressure

```python
from petrosmith.api import DrillingAPI

api = DrillingAPI()
pressure = api.calculate_hydrostatic_pressure(
    mud_weight=10.5,  # ppg
    tvd=10000  # feet
)
print(f"Hydrostatic pressure: {pressure:,.0f} psi")
# Output: Hydrostatic pressure: 5,460 psi
```

### Example 2: Well Control - Kick Detection

```python
from petrosmith.core.well_control import KickDetection

detection = KickDetection.detect_kick(
    pit_volume_gain=15.0,  # bbls
    flow_rate_increase=12.0,  # percent
    connection_flow=True
)

print(detection['status'])
# Output: KICK DETECTED - IMMEDIATE ACTION REQUIRED
print(detection['recommended_action'])
# Output: SHUT IN WELL
```

### Example 3: Formation Pressure Prediction

```python
from petrosmith.core.formation_pressure import PorePressurePrediction

pp = PorePressurePrediction.eatons_method(
    observed_parameter=85,  # Sonic DT
    normal_parameter=70,
    overburden_gradient=1.04,
    exponent=3.0
)

print(f"Pore pressure: {pp['equivalent_mud_weight_ppg']:.2f} ppg")
```

### Example 4: Error Handling

```python
from petrosmith.api import DrillingAPI
from petrosmith.exceptions import InvalidMudWeightError

api = DrillingAPI()

try:
    pressure = api.calculate_hydrostatic_pressure(25.0, 10000)
except InvalidMudWeightError as e:
    print(e)
    # Output: Mud weight 25.00 ppg is outside acceptable range [8.0, 20.0] ppg
```

---

## Complete Examples

See the `examples/` directory for full, runnable examples:

1. **`01_basic_drilling_calculations.py`** - Drilling calculations & error handling
2. **`02_well_control_kick_analysis.py`** - Complete kick response procedure
3. **`03_formation_evaluation.py`** - Pressure analysis & wellbore stability

Run any example:
```bash
python examples/01_basic_drilling_calculations.py
```

### Jupyter Notebooks

Interactive tutorials in the `notebooks/` directory:

```bash
uv sync --group dev
cd notebooks
uv run jupyter notebook
```

- **`01_drilling_basics.ipynb`** - Interactive drilling calculations with visualizations

---

## Testing

We have **66 real unit tests** with 100% pass rate:

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=petrosmith --cov-report=term

# Run specific test file
uv run pytest tests/test_drilling.py -v
```

**Test Coverage:**
- Drilling calculations (20 tests)
- Exception handling (18 tests)
- Input validation
- Edge cases

---

## Architecture

Clean 4-layer architecture with separation of concerns:

```
petrosmith/
├── models/          # Data models (Pydantic)
├── core/            # Business logic & calculations
├── services/        # Orchestration layer
└── api/             # User-facing interfaces
```

**Design Principles:**
- Stateless calculations
- Type hints everywhere
- Custom exceptions for helpful errors
- Input validation at API boundaries
- No magic numbers (constants file)

---

## Installation

### Requirements
- **Python 3.12+** (we don't support older versions)
- NumPy, SciPy, Pydantic

### From Source

```bash
git clone <repository-url>
cd reservoirengineering

# Install dependencies (uv will create a virtual environment automatically)
uv sync

# For development (includes dev dependencies)
uv sync --group dev
```

### From PyPI (Coming Soon)

```bash
pip install petrosmith
```

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Quick Start:**
```bash
# Install dev dependencies
uv sync --group dev

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest tests/ -v

# Format code
uv run black petrosmith/ tests/
uv run isort petrosmith/ tests/

# Run linters
uv run flake8 petrosmith/ tests/ --max-line-length=100
```

**What We Look For:**
- Tests for new features
- Type hints
- Docstrings with examples
- Input validation
- Custom exceptions for errors

---

## Code Quality

| Metric | Status |
|--------|--------|
| **Tests** | 84+ passing |
| **Test Coverage** | Core: ~70%, Overall: ~40% (growing) |
| **Type Hints** | 100% |
| **Custom Exceptions** | 15+ |
| **CI/CD** | GitHub Actions |
| **Python Version** | 3.12+ only |
| **Code Style** | Black + isort |
| **Status** | Beta - Core stable, services in development |

---

## What Makes This Different

### **Well-Tested Core**
- 84+ real unit tests with assertions
- Proper error handling with custom exceptions
- Input validation at API boundaries
- CI/CD pipeline with automated testing

### **Modern Python**
- Python 3.12+ only (fast, modern features)
- Type hints everywhere
- Clean architecture
- No legacy compatibility baggage

### **Well Documented**
- Runnable examples
- Clear error messages
- Inline documentation

### **Industry Standard**
- Calculations follow SPE/API standards
- Based on peer-reviewed methods
- Field-proven algorithms

---

## Logging

PetroSmith uses Python's standard logging module. By default, library logging is disabled (NullHandler).

### Enable Logging

```python
import logging

# Enable INFO level logging for PetroSmith
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('petrosmith')
logger.setLevel(logging.INFO)

# Now run your code - you'll see log messages
from petrosmith.api import DrillingAPI
api = DrillingAPI()
```

### Custom Logging Configuration

```python
import logging

# Custom handler and formatter
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Attach to petrosmith logger
logger = logging.getLogger('petrosmith')
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
```

---

## Documentation

- **Examples:** See `examples/` directory
- **Tests:** See `tests/` directory for usage patterns
- **Code Review:** See `GRIZZLED_DEV_REVIEW.md` for honest assessment
- **API Docs:**  docstrings in all modules

---

## Development

### CI/CD Pipeline

Every push runs:
- Tests on Python 3.12 & 3.13
- Ubuntu Linux (fast, simple)
- Code formatting checks (Black, isort)
- Linting (Flake8)
- Type checking (MyPy)
- Security scanning (Bandit, Safety)


---

## License

MIT License - see LICENSE file for details.

---

## Acknowledgments

Built with input from petroleum engineering professionals and following industry best practices.

**Technologies:**
- Python 3.12+
- NumPy & SciPy
- Pydantic
- Pytest
- GitHub Actions

---

## Support

- **Issues:** Open an issue on GitHub
- **Examples:** Check `examples/` directory
- **Tests:** Check `tests/` for usage patterns
- **Contributing:** See `CONTRIBUTING.md`

---

**Built for petroleum engineers, by developers who care about code quality.**

## Development Status

**Beta Software**: PetroSmith is under active development.

**What's Stable:**
- Core calculation engines (drilling, reservoir, well control)
- Exception handling and error messages
- API layer interfaces
- Type hints and documentation

**What's In Development:**
- Service layer (refactoring to stateless design)
- Database/persistence layer
- Integration tests
- Performance optimization

**Not Yet Production-Ready For:**
- Large-scale deployments
- Multi-user systems
- Mission-critical operations

**Ready For:**
- Internal tools and scripts
- Analysis and prototyping
- Educational purposes
- Single-user applications

