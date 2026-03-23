# Testing Guide

## Prerequisites

```bash
pip3 install pytest pandas numpy
```

## Run All Tests

**Option 1 — Use the bash script:**

```bash
chmod +x run_tests.sh
./run_tests.sh
```

**Option 2 — Run directly:**

```bash
python3 -m pytest tests/test_data_pipeline.py -v --tb=short
```

## Test Coverage

| Category | Tests | What it covers |
|---|---|---|
| Extract | 6 | CSV loading, schema, types, error handling |
| Transform | 25 | Lag/rolling features, calendar features, data leakage, multi-series isolation, metadata |
| Load | 4 | File creation, data integrity after save |
| Integration | 4 | Full end-to-end ETL pipeline with actual dataset |
| Edge Cases | 3 | Minimum rows, too-few rows, large horizon |
| **Total** | **42** | |
