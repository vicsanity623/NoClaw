#!/usr/bin/env bash

set -e

# Check if --fix argument was passed
if [[ "$1" == "--fix" ]]; then
    echo "🛠️  Auto-fixing issues..."
    ruff check src/ --fix
    ruff format src/
    echo "✅ Fixes applied."
    echo "-------------------------------------"
fi

echo "🚀 Starting PyOB Validation Suite..."

echo "-------------------------------------"
echo "🧹 1. Running Ruff (Linter & Imports)..."
ruff check src/ --fix

echo "-------------------------------------"
echo "🪄  2. Formatting Code with Ruff..."
ruff format src/ tests/

echo "-------------------------------------"
echo "🔎 3. Running Mypy (Type Checking)..."
mypy src/

echo "-------------------------------------"
echo "🧪 4. Running Pytest (Unit Tests)..."
if [ -d "tests" ] && [ "$(ls -A tests 2>/dev/null)" ]; then
    pytest tests/
else
    echo "⚠️  No tests found in 'tests/' directory."
fi

echo "-------------------------------------"
echo "✅ All checks passed!"
