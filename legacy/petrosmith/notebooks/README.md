# PetroSmith Jupyter Notebooks

Interactive tutorials and examples for the PetroSmith library.

## Getting Started

```bash
# Install Jupyter and dependencies
uv sync --group dev

# Start Jupyter
uv run jupyter notebook

# Open any notebook from the list below
```

## Notebooks

### 1. `01_drilling_basics.ipynb`
Basic drilling calculations with interactive visualizations:
- Hydrostatic pressure
- ECD calculations
- Casing design
- Pressure vs depth plots
- Error handling

**Level:** Beginner  
**Time:** 15 minutes

## Requirements

```bash
# Install all dependencies including dev tools
uv sync --group dev
```

## Running the Notebooks

```bash
# From this directory
uv run jupyter notebook

# Or from project root
cd .. && uv run jupyter notebook notebooks/
```

## Tips

- Run cells with `Shift+Enter`
- Restart kernel if imports fail: `Kernel → Restart`
- All notebooks assume you're running from the `notebooks/` directory
- Modify parameters and re-run cells to experiment

## Next Steps

After completing the notebooks:
- Check out `../examples/` for Python scripts
- Run `pytest ../tests/ -v` to see the test suite
- Read `../README.md` for full documentation
