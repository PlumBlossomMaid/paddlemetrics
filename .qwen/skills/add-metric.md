---
name: add-metric
description: Add a new metric to paddlemetrics — source, functional, tests, and registration
---

# Adding a New Metric

Checklist for adding a metric from torchmetrics to paddlemetrics.

## Step 1 — Source module

Create `paddlemetrics/<domain>/<metric_name>.py`:

```python
from typing import Any, Optional
import paddle
from paddlemetrics.metric import Metric

class MyMetric(Metric):
    full_state_update: bool = True  # or False if update() is append-only

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Register states — name must be a valid Python identifier
        self.add_state("my_state", default=paddle.zeros([]), dist_reduce_fx="sum")

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        # Validate inputs
        # Accumulate into self.my_state
        ...

    def compute(self) -> paddle.Tensor:
        # Derive result from self.my_state
        ...
```

Key rules:
- Subclass `paddlemetrics.metric.Metric` (which is `paddle.nn.Layer`)
- Use `add_state()` for all accumulators — NEVER use raw `self.x = tensor` (nn.Layer __setattr__ bug)
- Set `dist_reduce_fx` to `"sum"`, `"mean"`, `"cat"`, `"max"`, `"min"`, or a callable
- Set `full_state_update = True` if `compute()` reads states that `update()` modifies (default); `False` if update only appends
- Use `paddle.to_tensor()` or `paddle.zeros()` for default values
- Never assume auto type promotion — cast explicitly with `.cast()`

## Step 2 — Functional API

Create `paddlemetrics/functional/<domain>/<metric_name>.py`:

```python
from typing import Optional
import paddle

def my_metric(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    ...
) -> paddle.Tensor:
    """Stateless version of MyMetric."""
    # Pure computation, no state accumulation
    ...
```

## Step 3 — Register in `__init__.py`

### Domain `__init__.py`
Add to `paddlemetrics/<domain>/__init__.py`:
```python
from paddlemetrics.<domain>.<metric_name> import MyMetric
```
And add to `__all__`.

### Functional `__init__.py`
Add to `paddlemetrics/functional/__init__.py` `__all__` list under the correct domain section.

### Top-level lazy imports
Add to `paddlemetrics/__init__.py` `_lazy_imports` dict:
```python
"MyMetric": ("paddlemetrics.<domain>", "MyMetric"),
```

## Step 4 — Write tests

Create `tests/unittests/<domain>/test_<metric_name>.py`:

```python
from functools import partial
import numpy as np
import paddle
import pytest
from sklearn.metrics import ...  # or numpy reference

from paddlemetrics.<domain> import MyMetric
from paddlemetrics.functional.<domain> import my_metric
from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

seed_all(42)

# Fixed deterministic inputs — NEVER use random in comparison tests
_inputs = _Input(
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE),
    target=paddle.rand(NUM_BATCHES, BATCH_SIZE),
)

def _reference_fn(preds, target, **kwargs):
    sk_preds = preds.reshape([-1]).numpy()
    sk_target = target.reshape([-1]).numpy()
    return float(sk_metric(sk_target, sk_preds, **kwargs))

@pytest.mark.parametrize(("preds", "target", "ref_metric"), [
    (_inputs.preds, _inputs.target, _reference_fn),
])
class TestMyMetric(MetricTester):
    atol = 1e-6

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_my_metric(self, preds, target, ref_metric, ddp):
        self.run_class_metric_test(
            ddp, preds, target, MyMetric, ref_metric,
            metric_args={},
        )

    def test_my_metric_functional(self, preds, target, ref_metric):
        self.run_functional_metric_test(
            preds, target, my_metric, ref_metric,
            metric_args={},
        )
```

Key rules:
- Use `seed_all(42)` at module level
- Use deterministic inputs (`paddle.rand` with seed), NOT random per-test
- Reference function must convert paddle → numpy → sklearn
- Always parametrize `ddp` with `[pytest.param(True, marks=pytest.mark.DDP), False]`
- Set `atol` based on numerical precision needs (default 1e-8 may be too tight for float32)

## Step 5 — Verify

```bash
pytest tests/unittests/<domain>/test_<metric_name>.py -m "not DDP" -v --tb=short
```

## Step 6 — Update `__init__.py` for lazy imports

If the metric should be importable as `from paddlemetrics import MyMetric`, add it to the `_lazy_imports` dict in `paddlemetrics/__init__.py`.
