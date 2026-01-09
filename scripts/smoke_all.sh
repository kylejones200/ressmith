#!/bin/bash
# Smoke test script for monorepo workspace
# Installs timesmith first, then all downstream repos, then runs all smoke tests

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MONOREPO_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

echo "Smith Ecosystem Smoke Test"
echo "=========================="

# Step 1: Install timesmith first
echo ""
echo "1. Installing timesmith..."
if [ -d "$MONOREPO_ROOT/timesmith" ]; then
    cd "$MONOREPO_ROOT/timesmith"
    pip install -e .
    echo "   timesmith installed from local path"
elif [ -d "$REPO_ROOT/../timesmith" ]; then
    cd "$REPO_ROOT/../timesmith"
    pip install -e .
    echo "   timesmith installed from local path"
else
    pip install "timesmith>=0.2.0"
    echo "   timesmith installed from PyPI"
fi

# Step 2: Install each downstream repo
echo ""
echo "2. Installing downstream repos..."
REPOS=("plotsmith" "anomsmith" "ressmith" "geosmith")

for repo in "${REPOS[@]}"; do
    if [ -d "$MONOREPO_ROOT/$repo" ]; then
        echo "   Installing $repo..."
        cd "$MONOREPO_ROOT/$repo"
        pip install -e .
        echo "   $repo installed"
    elif [ -d "$REPO_ROOT/../$repo" ]; then
        echo "   Installing $repo..."
        cd "$REPO_ROOT/../$repo"
        pip install -e .
        echo "   $repo installed"
    else
        echo "   $repo not found, skipping"
    fi
done

# Step 3: Run all smoke tests
echo ""
echo "3. Running smoke tests..."
for repo in "${REPOS[@]}"; do
    if [ -d "$MONOREPO_ROOT/$repo" ] && [ -f "$MONOREPO_ROOT/$repo/scripts/smoke.py" ]; then
        echo ""
        echo "   Running smoke test for $repo..."
        python "$MONOREPO_ROOT/$repo/scripts/smoke.py"
        echo "   $repo smoke test passed"
    elif [ -d "$REPO_ROOT/../$repo" ] && [ -f "$REPO_ROOT/../$repo/scripts/smoke.py" ]; then
        echo ""
        echo "   Running smoke test for $repo..."
        python "$REPO_ROOT/../$repo/scripts/smoke.py"
        echo "   $repo smoke test passed"
    else
        echo "   $repo smoke test not found, skipping"
    fi
done

echo ""
echo "=========================="
echo "All smoke tests passed!"

