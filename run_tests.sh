#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# run_tests.sh — Test runner for AI-based Supply Chain Management pipeline
#
# Usage:
#   ./run_tests.sh              # Run all tests
#   ./run_tests.sh -v           # Verbose output
#   ./run_tests.sh -k "anomaly" # Run only tests matching "anomaly"
#   ./run_tests.sh --cov        # Run with coverage report
#   ./run_tests.sh --html       # Generate HTML coverage report
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ── Resolve project root (directory containing this script) ──────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT"

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  AI-based Supply Chain Management — Test Runner${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Project root:${NC} $PROJECT_ROOT"
echo ""

# ── Parse arguments ──────────────────────────────────────────────────────────
VERBOSE=""
KEYWORD=""
COVERAGE=false
HTML_REPORT=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        -k|--keyword)
            KEYWORD="-k $2"
            shift 2
            ;;
        --cov|--coverage)
            COVERAGE=true
            shift
            ;;
        --html)
            COVERAGE=true
            HTML_REPORT=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose        Verbose pytest output"
            echo "  -k, --keyword EXPR   Only run tests matching EXPR"
            echo "  --cov, --coverage    Run with coverage report"
            echo "  --html               Generate HTML coverage report"
            echo "  -h, --help           Show this help"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ── Step 1: Check Python ────────────────────────────────────────────────────
echo -e "${YELLOW}[1/4] Checking Python...${NC}"
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo -e "${RED}ERROR: Python not found. Please install Python 3.8+.${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON --version 2>&1)
echo -e "  Found: ${GREEN}$PYTHON_VERSION${NC}"

# ── Step 2: Install test dependencies ────────────────────────────────────────
echo ""
echo -e "${YELLOW}[2/4] Installing test dependencies...${NC}"

# Core test deps
DEPS=(pytest numpy pandas scipy pyyaml python-dotenv pyarrow)

if $COVERAGE; then
    DEPS+=(pytest-cov)
fi

$PYTHON -m pip install --quiet --upgrade "${DEPS[@]}" 2>&1 | tail -1 || true
echo -e "  ${GREEN}Dependencies installed.${NC}"

# ── Step 3: Build pytest command ─────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[3/4] Configuring test run...${NC}"

PYTEST_CMD=("$PYTHON" -m pytest)
PYTEST_CMD+=(tests/test_scripts.py)
PYTEST_CMD+=($VERBOSE)

if [[ -n "$KEYWORD" ]]; then
    PYTEST_CMD+=($KEYWORD)
fi

if $COVERAGE; then
    PYTEST_CMD+=(--cov=scripts --cov=dags --cov-report=term-missing)
    if $HTML_REPORT; then
        PYTEST_CMD+=(--cov-report=html:htmlcov)
    fi
fi

PYTEST_CMD+=(-x)             # Stop on first failure
PYTEST_CMD+=(--tb=short)     # Short tracebacks
PYTEST_CMD+=(-q)             # Quieter unless verbose

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    PYTEST_CMD+=("${EXTRA_ARGS[@]}")
fi

echo -e "  Command: ${CYAN}${PYTEST_CMD[*]}${NC}"

# ── Step 4: Run tests ───────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[4/4] Running tests...${NC}"
echo -e "${CYAN}───────────────────────────────────────────────────────────────${NC}"
echo ""

# Run pytest — allow it to fail without killing the script
set +e
"${PYTEST_CMD[@]}"
EXIT_CODE=$?
set -e

echo ""
echo -e "${CYAN}───────────────────────────────────────────────────────────────${NC}"

# ── Results ──────────────────────────────────────────────────────────────────
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
elif [ $EXIT_CODE -eq 5 ]; then
    echo -e "${YELLOW}No tests were collected. Check test file paths.${NC}"
else
    echo -e "${RED}Some tests failed (exit code: $EXIT_CODE).${NC}"
fi

if $HTML_REPORT && [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Coverage report: ${PROJECT_ROOT}/htmlcov/index.html${NC}"
fi

echo ""
exit $EXIT_CODE
