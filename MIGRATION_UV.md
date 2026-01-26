# Migration to uv

This document outlines the migration from pip/requirements.txt to uv for the ressmith project.

## Overview

The project has been modernized to use `uv` as the default package manager and `pyproject.toml` as the single source of truth for dependencies. The `uv.lock` file is committed to the repository for reproducible builds.

## Command Mapping

### Installation

**Old (pip):**
```bash
pip install ressmith
pip install -e .  # Development install
pip install -r requirements.txt
```

**New (uv):**
```bash
uv pip install ressmith  # End user installation from PyPI
uv sync --group dev  # Development install with dev dependencies
```

### Running Commands

**Old:**
```bash
pytest
ruff check .
black ressmith tests
python -m build
```

**New:**
```bash
uv run pytest
uv run ruff check .
uv run black ressmith tests
uv run python -m build
```

### Testing

**Old:**
```bash
pytest tests/
pytest --cov=ressmith
```

**New:**
```bash
uv run pytest tests/
uv run pytest --cov=ressmith
```

### Linting

**Old:**
```bash
ruff check ressmith tests
black ressmith tests
mypy ressmith/
```

**New:**
```bash
uv run ruff check ressmith tests
uv run black ressmith tests
uv run mypy ressmith/
```

### Building and Publishing

**Old:**
```bash
pip install build twine
python -m build
twine check dist/*
twine upload dist/*
```

**New:**
```bash
uv sync --group dev  # Includes build and twine
uv run python -m build
uv run twine check dist/*
uv run twine upload dist/*
```

## Dependency Management

### Runtime Dependencies

Runtime dependencies are defined in `pyproject.toml` under `[project].dependencies`:

```toml
[project]
dependencies = [
  "numpy>=1.24.0",
  "pandas>=2.0.0",
  "scipy>=1.10.0",
  "timesmith>=0.1.0",
]
```

### Development Dependencies

Development dependencies are defined in `pyproject.toml` under `[dependency-groups].dev`:

```toml
[dependency-groups]
dev = [
  "pytest>=7.0",
  "pytest-cov>=4.0",
  "ruff>=0.1.0",
  "black>=23.0.0",
  "mypy>=1.0.0",
  "build>=1.0.0",
  "twine>=5.0.0",
]
```

### Optional Dependencies

Optional dependencies (extras) are defined in `pyproject.toml` under `[project.optional-dependencies]`:

```toml
[project.optional-dependencies]
viz = ["matplotlib>=3.7.0"]
docs = ["sphinx>=7.0", "sphinx-rtd-theme>=2.0", ...]
examples = ["jupyter>=1.0", "notebook>=7.0", ...]
```

Install with:
```bash
uv sync --extra viz --extra docs
```

## Files Changed

### Removed/Deprecated
- No requirements.txt files were removed (docs/requirements.txt kept for Read the Docs compatibility)

### Updated
- `pyproject.toml` - Added build/twine to dev deps, added myst-parser to docs
- `README.md` - Updated installation instructions
- `CONTRIBUTING.md` - Already using uv commands
- `.github/workflows/ci.yml` - Already using uv
- `.github/workflows/release.yml` - Updated to use uv sync for build tools
- `scripts/smoke.py` - Updated to use uv commands
- `scripts/smoke_all.sh` - Updated to use uv commands
- `RELEASE_NOTES.md` - Updated installation instructions
- `docs/requirements.txt` - Marked as legacy (kept for Read the Docs)

### New
- `uv.lock` - Lock file for reproducible builds (committed to repo)
- `MIGRATION_UV.md` - This file

## CI/CD Changes

### GitHub Actions

The CI workflows now:
1. Install uv using `astral-sh/setup-uv@v5`
2. Use `uv sync --frozen --group dev` to install dependencies
3. Run commands via `uv run` (e.g., `uv run pytest`, `uv run ruff check`)

### Release Workflow

The release workflow:
1. Uses `uv sync --frozen --group dev` (includes build and twine)
2. Builds with `uv run python -m build`
3. Validates with `uv run twine check dist/*`

## Benefits

1. **Faster**: uv is significantly faster than pip
2. **Reproducible**: uv.lock ensures consistent builds across environments
3. **Modern**: Uses PEP 621 standard for pyproject.toml
4. **Single Source of Truth**: All dependencies in pyproject.toml
5. **Better Developer Experience**: Simpler commands, better error messages

## Troubleshooting

### Lock file out of sync

If dependencies change, regenerate the lock file:
```bash
uv sync --group dev
```

### Clean install

To start fresh:
```bash
rm -rf .venv uv.lock
uv sync --group dev
```

### Adding a new dependency

1. Add to `pyproject.toml` (under `[project].dependencies` or `[dependency-groups].dev`)
2. Run `uv sync --group dev` to update lock file
3. Commit both `pyproject.toml` and `uv.lock`

## Further Reading

- [uv documentation](https://docs.astral.sh/uv/)
- [PEP 621 - Project metadata](https://peps.python.org/pep-0621/)
- [PEP 735 - Dependency groups](https://peps.python.org/pep-0735/)

