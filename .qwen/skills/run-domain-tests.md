---
name: run-domain-tests
description: Run tests by domain with common filters and diagnose failures
---

# Running Domain Tests

## Quick Commands

```bash
# Single domain
pytest tests/unittests/regression/ -m "not DDP" -v --tb=short

# Single file
pytest tests/unittests/classification/test_accuracy.py -m "not DDP" -v --tb=short

# Single test
pytest tests/unittests/regression/test_r2.py::TestR2Score::test_r2 -v --tb=long

# All domains (full suite)
pytest tests/ -m "not DDP" --timeout=300 -v

# With parallel execution
pytest tests/ -m "not DDP" --timeout=300 -n auto

# Stop on first failure
pytest tests/<domain>/ -m "not DDP" -x --tb=long

# Show slowest 30 tests
pytest tests/<domain>/ -m "not DDP" --durations=30

# Run only DDP tests (needs multi-process setup)
pytest tests/ -m DDP -v
```

## Domain → Directory Mapping

| Domain | Directory | Extras Required |
|--------|-----------|-----------------|
| bases | `tests/unittests/bases/` | — |
| classification | `tests/unittests/classification/` | `[classification]` |
| regression | `tests/unittests/regression/` | `[regression]` |
| retrieval | `tests/unittests/retrieval/` | — |
| image | `tests/unittests/image/` | `[image]` |
| detection | `tests/unittests/detection/` | `[detection]` |
| text | `tests/unittests/text/` | `[text]` |
| nominal | `tests/unittests/nominal/` | `[nominal]` |
| clustering | `tests/unittests/clustering/` | — |
| segmentation | `tests/unittests/segmentation/` | — |
| shape | `tests/unittests/shape/` | — |
| wrappers | `tests/unittests/wrappers/` | — |
| utilities | `tests/unittests/utilities/` | — |

## Diagnosing Failures

### Numerical mismatch
```
AssertionError: Tensor mismatch at element 0:
  rtol=1e-05, atol=1e-05
```
→ Check: type promotion (int→float), accumulation order, reduce function

### Shape mismatch
```
RuntimeError: shape mismatch
```
→ Check: `reshape` vs `view`, broadcast rules, `squeeze`/`unsqueeze` dims

### Type error
```
TypeError: (InvalidType) paddle.xxx only supports ...
```
→ Check: CUDA dtype limits, cast bool/uint8 to int32/float32

### Missing attribute
```
AttributeError: 'Tensor' object has no attribute 'xxx'
```
→ Check: PyTorch-ism (`.values`, `.item()` on bool), API name difference

### DDP-specific failure
```
Only test in DDP mode if:
  - @pytest.mark.DDP is on the parametrize
  - Error mentions "distributed" or "all_reduce"
```
→ Run with `-m "not DDP"` first to isolate

## Test Markers

```bash
# Skip DDP tests
pytest -m "not DDP"

# Only DDP tests
pytest -m DDP

# Skip slow/flaky
pytest -m "not slow"
```

## Coverage

```bash
# Generate coverage report
pytest tests/unittests/<domain>/ --cov=paddlemetrics.<domain> --cov-report=term-missing -m "not DDP"
```
