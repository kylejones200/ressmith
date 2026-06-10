#!/bin/bash
# Setup script for git hooks and pre-commit

set -e

echo "Setting up development environment..."
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if we're in a git repository
if [ ! -d .git ]; then
    echo "ERROR: Not a git repository"
    exit 1
fi

# Install pre-commit if available
if command -v pre-commit &> /dev/null; then
    echo "Installing pre-commit hooks..."
    pre-commit install
    echo -e "${GREEN}[OK]${NC} Pre-commit hooks installed"
else
    echo -e "${YELLOW}[WARN]${NC} pre-commit not found. Install with: uv run pre-commit install"
fi

# Make pre-push hook executable
if [ -f .git/hooks/pre-push ]; then
    chmod +x .git/hooks/pre-push
    echo -e "${GREEN}[OK]${NC} Pre-push hook is executable"
else
    echo -e "${YELLOW}[WARN]${NC} Pre-push hook not found at .git/hooks/pre-push"
fi

# Check if dev dependencies are installed
echo ""
echo "Checking development dependencies..."

MISSING=0

if ! command -v black &> /dev/null; then
    echo "  [MISSING] black not installed"
    MISSING=1
fi

if ! command -v isort &> /dev/null; then
    echo "  [MISSING] isort not installed"
    MISSING=1
fi

if ! command -v flake8 &> /dev/null; then
    echo "  [MISSING] flake8 not installed"
    MISSING=1
fi

if ! command -v mypy &> /dev/null; then
    echo "  [MISSING] mypy not installed"
    MISSING=1
fi

if ! command -v pylint &> /dev/null; then
    echo "  [MISSING] pylint not installed"
    MISSING=1
fi

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "WARNING: Some development dependencies are missing."
    echo "Install them with:"
    echo "  uv sync --group dev"
    echo ""
else
    echo -e "${GREEN}[OK]${NC} All development dependencies installed"
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo -e "${GREEN}[SUCCESS] Development environment setup complete!${NC}"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Install dev dependencies: uv sync --group dev"
echo "  2. Format code: black petrosmith/ test_petrosmith.py"
echo "  3. Sort imports: isort petrosmith/ test_petrosmith.py"
echo "  4. Run checks: ./run_checks.sh"
echo "  5. Run tests: python test_petrosmith.py"
echo ""
echo "Git hooks enabled:"
echo "  • Pre-commit: Auto-format and lint on commit"
echo "  • Pre-push: Run all checks before push"
echo ""
