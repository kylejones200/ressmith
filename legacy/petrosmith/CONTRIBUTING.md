# Contributing to PetroSmith

Thank you for your interest in contributing to PetroSmith! This document provides guidelines and instructions for contributing to the project.

## Quick Start

### 1. Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd reservoirengineering

# Install dependencies (uv will create a virtual environment automatically)
uv sync --group dev

# Set up git hooks
chmod +x setup_hooks.sh
./setup_hooks.sh
```

### 2. Development Workflow

```bash
# Create a new branch
git checkout -b feature/your-feature-name

# Make your changes
# ... edit code ...

# Format code
uv run black petrosmith/ test_petrosmith.py
uv run isort petrosmith/ test_petrosmith.py

# Run all checks
chmod +x run_checks.sh
./run_checks.sh

# Commit your changes
git add .
git commit -m "Description of your changes"

# Push (pre-push hook will run automatically)
git push origin feature/your-feature-name
```

## Code Quality Standards

All contributions must meet these standards:

### Required Checks
- **Black**: Code must be formatted with Black
- **isort**: Imports must be properly sorted
- **Flake8**: No linting errors
- **Tests**: All tests must pass

### Recommended Checks
- **MyPy**: Type hints should pass type checking
- **Pylint**: Should score > 8.0
- **Bandit**: No security issues

## Architecture Guidelines

PetroSmith follows a strict 4-layer architecture:

### Layer 1: Models (`petrosmith/models/`)
```python
from dataclasses import dataclass

@dataclass
class Well:
    """Pure data class - no business logic"""
    well_id: str
    depth: float
    diameter: float
```

### Layer 2: Core (`petrosmith/core/`)
```python
class DrillingCalculations:
    """Pure calculation functions"""
    
    @staticmethod
    def calculate_pressure(mud_weight: float, tvd: float) -> float:
        """
        Calculate hydrostatic pressure
        
        Args:
            mud_weight: Mud weight in ppg
            tvd: True vertical depth in ft
            
        Returns:
            Pressure in psi
        """
        return 0.052 * mud_weight * tvd
```

### Layer 3: Services (`petrosmith/services/`)
```python
class WellService:
    """Orchestrates multiple calculations"""
    
    def __init__(self):
        self.calculations = DrillingCalculations()
        self.repository = WellRepository()
```

### Layer 4: API (`petrosmith/api/`)
```python
class DrillingAPI:
    """External interface"""
    
    def __init__(self):
        self.service = WellService()
```

## Documentation Guidelines

### Docstring Format

Use Google-style docstrings:

```python
def calculate_something(param1: float, param2: str, param3: Optional[int] = None) -> Dict:
    """
    Brief description of what this function does
    
    Detailed explanation if needed. Can be multiple paragraphs.
    
    Args:
        param1: Description of param1 with units
        param2: Description of param2
        param3: Optional parameter description
        
    Returns:
        Dict with:
            - key1: Description
            - key2: Description
            
    Raises:
        ValueError: When this happens
        
    Example:
        >>> result = calculate_something(1.5, "test")
        >>> print(result['key1'])
        42
    """
    # Implementation
    return {'key1': 42, 'key2': 'value'}
```

### Type Hints

Always include type hints:

```python
from typing import Dict, List, Optional, Tuple

def process_data(
    values: List[float],
    options: Optional[Dict[str, str]] = None
) -> Tuple[float, float]:
    """Process data and return min and max"""
    return min(values), max(values)
```

## Testing Guidelines

### Test Structure

```python
def test_calculation():
    """Test description"""
    # Arrange
    input_value = 10.0
    expected_result = 52.0
    
    # Act
    result = calculate_pressure(input_value, 100)
    
    # Assert
    assert abs(result - expected_result) < 0.01
```

### Running Tests

```bash
# Run all tests
uv run python test_petrosmith.py

# Run with pytest
uv run pytest -v

# Run with coverage
uv run pytest --cov=petrosmith --cov-report=html
```

## Tools Configuration

All tools are configured in `pyproject.toml`:

- **Black**: Max line length 100
- **isort**: Black-compatible profile
- **Flake8**: Configured in `.flake8`
- **MyPy**: Ignore missing imports
- **Pylint**: Disabled overly strict rules

## Pull Request Process

1. **Create an Issue**: Describe what you want to change
2. **Fork & Branch**: Create a feature branch
3. **Implement**: Make your changes following the guidelines
4. **Test**: Ensure all tests pass
5. **Document**: Update documentation if needed
6. **Format**: Run Black and isort
7. **Check**: Run `./run_checks.sh`
8. **Commit**: Write clear commit messages
9. **Push**: Push to your fork
10. **PR**: Create a pull request

### Commit Message Format

```
<type>: <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Tests
- `chore`: Maintenance

Example:
```
feat: Add kick tolerance calculation for deepwater

Implemented kick tolerance analysis for deepwater operations including:
- Maximum allowable kick size calculation
- Weak formation considerations
- Pressure margin analysis

Closes #123
```

## Bug Reports

Include:
- PetroSmith version
- Python version
- Operating system
- Minimal code to reproduce
- Expected vs actual behavior
- Error messages/stack traces

## Feature Requests

Include:
- Clear use case
- Expected behavior
- Example code (if applicable)
- Why this would be useful

## Code Review Criteria

PRs are reviewed for:
- Follows architecture patterns
- Includes tests
- Passes all checks
- Has documentation
- Clear commit messages
- No breaking changes (or clearly documented)

## Areas for Contribution

We especially welcome contributions in:

### High Priority
- Additional well testing methods
- More rock mechanics failure criteria
- Extended drilling fluids models
- Production optimization algorithms
- Machine learning integration

### Medium Priority
- Performance optimizations
- Additional unit tests
- Documentation improvements
- Example notebooks
- Tutorial content

### Low Priority
- Code cleanup
- Style improvements
- Minor bug fixes

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue
- **Feature Requests**: Open a GitHub Issue
- **Security Issues**: Email directly (don't open public issue)

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions
- Follow professional standards

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Credited in release notes
- Acknowledged in documentation

Thank you for contributing to PetroSmith!
