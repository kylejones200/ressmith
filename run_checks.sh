#!/bin/bash
# Run all code quality checks

set -e

echo "Running all code quality checks..."
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

FAILED=0

print_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_error() {
    echo -e "${RED}[FAIL]${NC} $1"
    FAILED=1
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# 1. Black formatting check
print_header "1. Code Formatting (Black)"
if black --check petrosmith/ test_petrosmith.py; then
    print_success "Code is properly formatted"
else
    print_error "Code formatting issues found"
    echo "Fix with: black petrosmith/ test_petrosmith.py"
fi

# 2. isort import sorting
print_header "2. Import Sorting (isort)"
if isort --check-only petrosmith/ test_petrosmith.py; then
    print_success "Imports are properly sorted"
else
    print_error "Import sorting issues found"
    echo "Fix with: isort petrosmith/ test_petrosmith.py"
fi

# 3. Flake8 linting
print_header "3. Linting (Flake8)"
if flake8 petrosmith/ test_petrosmith.py --count --statistics; then
    print_success "No linting issues found"
else
    print_error "Linting issues found"
fi

# 4. Pylint
print_header "4. Linting (Pylint)"
if pylint petrosmith/ --disable=C0114,C0115,C0116,R0913,R0914,R0915,W0212 --exit-zero; then
    print_success "Pylint check complete"
else
    print_warning "Pylint issues found (non-blocking)"
fi

# 5. MyPy type checking
print_header "5. Type Checking (MyPy)"
if mypy petrosmith/ --ignore-missing-imports --no-strict-optional; then
    print_success "Type checking passed"
else
    print_warning "Type checking issues found (non-blocking)"
fi

# 6. Security checks
print_header "6. Security (Bandit)"
if command -v bandit &> /dev/null; then
    if bandit -r petrosmith/ -ll; then
        print_success "No security issues found"
    else
        print_warning "Security issues found (review recommended)"
    fi
else
    print_warning "Bandit not installed (pip install bandit)"
fi

# 7. Run tests
print_header "7. Running Tests"
if python test_petrosmith.py; then
    print_success "All tests passed"
else
    print_error "Tests failed"
fi

# Summary
echo ""
echo "═══════════════════════════════════════════════════════"
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}[SUCCESS] All checks passed! Code is ready for commit.${NC}"
    echo "═══════════════════════════════════════════════════════"
    exit 0
else
    echo -e "${RED}[FAILED] Some checks failed. Please fix the issues.${NC}"
    echo "═══════════════════════════════════════════════════════"
    echo ""
    echo "Quick fixes:"
    echo "  black petrosmith/ test_petrosmith.py     # Auto-format code"
    echo "  isort petrosmith/ test_petrosmith.py     # Auto-sort imports"
    echo ""
    exit 1
fi
