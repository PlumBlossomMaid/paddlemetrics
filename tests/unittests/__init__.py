"""Test utilities for paddlemetrics."""
import os
import warnings
from typing import NamedTuple

import numpy
import paddle
from paddle import Tensor

from unittests.conftest import (
    BATCH_SIZE,
    EXTRA_DIM,
    NUM_BATCHES,
    NUM_CLASSES,
    NUM_PROCESSES,
    THRESHOLD,
    USE_PYTEST_POOL,
    setup_ddp,
)

for tp_name, tp_ins in [("object", object), ("bool", bool), ("int", int), ("float", float)]:
    if not hasattr(numpy, tp_name):
        setattr(numpy, tp_name, tp_ins)

_PATH_UNITTESTS = os.path.dirname(__file__)
_PATH_ALL_TESTS = os.path.dirname(_PATH_UNITTESTS)
_PATH_TEST_CACHE = os.getenv(
    "PYTEST_REFERENCE_CACHE", os.path.join(_PATH_ALL_TESTS, "_cache-references")
)

# Simple cachier-like decorator for caching reference computations
try:
    from cachier import cachier

    _reference_cachier = cachier(cache_dir=_PATH_TEST_CACHE, separate_files=True)
except ImportError:
    # Fallback: no-op decorator
    def _reference_cachier(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.*")

# Suppress TF32 for reproducibility
paddle.set_flags({"FLAGS_cudnn_deterministic": True})


class _Input(NamedTuple):
    """Input for parametrized tests."""
    preds: Tensor
    target: Tensor


class _GroupInput(NamedTuple):
    """Group input for parametrized tests."""
    preds: Tensor
    target: Tensor
    groups: Tensor
