# PaddleMetrics — Project Context

Full port of [torchmetrics](https://github.com/Lightning-AI/torchmetrics) to PaddlePaddle.
~27K tests across 12 domains, all passing on CPU.

## Quick Reference

```bash
# Install (editable, with test deps)
pip install -e ".[tests]"

# Install all domain extras
pip install -e ".[tests,classification,regression,detection,image,text,nominal,clustering,segmentation,retrieval]"

# Run a single domain
pytest tests/unittests/regression/ -m "not DDP" -v --tb=short

# Run everything
pytest tests/ -m "not DDP" --timeout=300 -x

# Lint
ruff check paddlemetrics/ --select E,F,W,I --ignore E501
```

## Architecture

```
paddlemetrics/
├── __init__.py          # Eager: Metric, MetricCollection, aggregation. Lazy: all domain metrics
├── __about__.py         # __version__
├── metric.py            # Metric (nn.Layer ABC), CompositionalMetric (~1054 lines)
├── collections.py       # MetricCollection
├── aggregation.py       # MeanMetric, SumMetric, MaxMetric, MinMetric, CatMetric, RunningMean/Sum
├── utils/               # checks, compute, data, distributed, enums, exceptions, filtering, imports, plot, prints
├── functional/          # Stateless function versions of all metrics (by domain)
├── classification/      # 28 metrics with task dispatch (Binary/Multiclass/Multilabel)
├── regression/          # 21 metrics
├── retrieval/           # RetrievalMAP, RetrievalNormalizedDCG, etc.
├── image/               # SSIM, PSNR, TotalVariation, etc.
├── detection/           # PanopticQuality, etc.
├── text/                # BLEU, ROUGE, WER, CER, Perplexity, etc.
├── clustering/          # AdjustedRandScore, NMI, DBS, etc.
├── nominal/             # CramersV, FleissKappa, TheilsU, etc.
├── segmentation/        # DiceScore, MeanIoU, HausdorffDistance, etc.
├── shape/               # ProcrustesDisparity
├── audio/               # SI-SNR, SI-SDR, PESQ, STOI, etc.
├── video/               # (minimal)
├── multimodal/          # (minimal)
└── wrappers/            # BootStrapper, ClasswiseWrapper, MinMaxMetric, MetricTracker, MultitaskWrapper, MultioutputWrapper
```

## Key Patterns

### Metric class structure
Every metric subclasses `Metric` (which is `paddle.nn.Layer`):
- `add_state(name, default, dist_reduce_fx)` — register persistent state
- `update(preds, target)` — accumulate batch data into states
- `compute()` — derive final result from accumulated states
- `forward(preds, target)` — calls update + compute (via `_forward_reduce_state_update` or `_forward_full_state_update`)
- `reset()` — clear all states back to defaults

### Task dispatch (classification)
Generic names like `Accuracy` auto-dispatch to `BinaryAccuracy` / `MulticlassAccuracy` / `MultilabelAccuracy` based on input shape and `task` kwarg.

### Functional API
Stateless mirrors in `paddlemetrics.functional.*`. Same computation, no state accumulation.

### Wrapper metrics
`WrapperMetric` base overrides `forward()` to raise `NotImplementedError`. Concrete wrappers implement their own `forward()`. For nested wrappers (e.g., `_MultioutputMetric` wrapping `MultioutputWrapper`), delegate `forward()` to the inner wrapper's `__call__()` — do NOT use base `_forward_reduce_state_update`.

### Lazy imports
Domain metrics are lazy-imported in `paddlemetrics/__init__.py` via `__getattr__` to avoid cascading import errors when optional deps are missing.

## Paddle API Differences from PyTorch

These are the patterns that caused systematic failures during migration:

| # | Issue | Fix |
|---|-------|-----|
| 1 | `paddle.sort()` returns single tensor | Use `paddle.argsort()` for indices |
| 2 | `paddle.max(x, axis)` returns values only | Index `[0]` for values, use `argmax` for indices |
| 3 | `paddle.max(a, b)` is reduction | Use `paddle.maximum(a, b)` for element-wise |
| 4 | `.view()` requires contiguous | Use `.reshape()` instead |
| 5 | `one_hot` returns float32 | Cast to int64 if needed |
| 6 | `paddle.unique` no uint8 GPU kernel | Cast to int32 first |
| 7 | `paddle.trapezoid` requires float | Cast int inputs to float64 |
| 8 | `paddle.nan_to_num` requires float32/64 | Cast from float16 |
| 9 | `paddle.equal` returns element-wise | Wrap with `paddle.all()` for boolean |
| 10 | `argsort` propagates `requires_grad` | `.detach()` if not differentiable |
| 11 | `scatter_add_` requires same ndim | Flatten 2D inputs to 1D if needed |
| 12 | `paddle.linalg.norm` with p=2 = spectral | Use `p="fro"` for Frobenius norm |
| 13 | `Tensor.gt(int)` requires tensor arg | Use `paddle.greater_than(x, paddle.to_tensor(val))` |
| 14 | `paddle.split(n)` = n chunks | PyTorch `split(n)` = chunk size n; use `chunk()` for chunks |
| 15 | Strict type promotion | Mixed int/float needs explicit `.cast()` |
| 16 | CUDA dtype limits | Many CUDA kernels don't support bool/uint8/int8; cast to int32/float32 |
| 17 | `nn.Layer.__setattr__` drops Tensor | Use `self.__dict__["attr"]` for non-parameter tensors |
| 18 | `nn.Layer.deepcopy` closure binding | Use pickle round-trip for `clone()` |

## Test Conventions

### Constants (`tests/unittests/__init__.py`)
- `NUM_BATCHES = 4`, `BATCH_SIZE = 32`, `NUM_CLASSES = 5`
- `_Input = NamedTuple("preds", "target")` — standard test input container

### Test class pattern
```python
from unittests._helpers.testers import MetricTester

class TestMyMetric(MetricTester):
    atol = 1e-6  # override tolerance if needed

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_my_metric(self, ddp, ...):
        self.run_class_metric_test(ddp, preds, target, MyMetric, ref_fn, metric_args={...})

    def test_my_metric_functional(self, ...):
        self.run_functional_metric_test(preds, target, my_metric_fn, ref_fn, metric_args={...})
```

### Reference functions
Use sklearn / numpy as reference. Wrap in a function that converts paddle tensors to numpy.

### Markers
- `@pytest.mark.DDP` — distributed data parallel test (always parametrize with `True, marks=...` and `False`)
- `--strict-markers` enabled — only registered markers allowed

## CI

GitHub Actions: `.github/workflows/ci-tests.yml`
- Python 3.9 / 3.10 / 3.11 on ubuntu-22.04
- `paddlepaddle` CPU
- Per-domain test steps, DDP skipped
- Lint with ruff
- Gate job: `testing-guardian`

## Conventions

- **Import style**: `import paddle` (never `from paddle import ...`)
- **Tensor creation**: `paddle.to_tensor(data, dtype=...)` or `paddle.zeros/ones/rand`
- **Boolean masking**: `paddle.where(condition, x, y)` — never `tensor[bool_mask]` on GPU
- **No `.values`**: PyTorch-ism; use direct tensor ops
- **State dict keys**: `add_state` names must be valid Python identifiers
- **Dist reduce**: Use `"sum"`, `"mean"`, `"cat"`, `"max"`, `"min"`, or a callable
- **Git email**: `1589524335@qq.com` (for CLA signing)
