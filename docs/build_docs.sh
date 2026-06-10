#!/bin/bash
# Build documentation locally

cd "$(dirname "$0")"
cd ..

# Clean previous build
rm -rf build/

# Build HTML documentation
sphinx-build -b html source build/html

echo ""
echo "Documentation built successfully!"
echo "Open build/html/index.html in your browser"
