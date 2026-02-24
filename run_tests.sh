#!/bin/bash

cd "$(dirname "$0")" || exit 1

echo "========================================"
echo "  Supply Chain Pipeline - Test Runner"
echo "========================================"
echo ""

if ! command -v python3 -m pytest &> /dev/null; then
    echo "pytest not found. Installing..."
    pip3 install pytest
    echo ""
fi

echo "Running tests..."
echo "----------------------------------------"
python3 -m pytest tests/test_data_pipeline.py -v --tb=short

EXIT_CODE=$?

echo ""
echo "----------------------------------------"
if [ $EXIT_CODE -eq 0 ]; then
    echo "All tests passed!"
else
    echo "Some tests failed. Check output above."
fi

exit $EXIT_CODE
