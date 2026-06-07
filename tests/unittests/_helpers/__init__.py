from __future__ import annotations

import importlib.metadata
import os
import random
import sys

import numpy
import paddle
from packaging.version import Version

from unittests._helpers.wrappers import skip_on_connection_issues, skip_on_cuda_oom, skip_on_running_out_of_memory


def seed_all(seed):
    """Set the seed of all computational frameworks."""
    random.seed(seed)
    numpy.random.seed(seed)
    paddle.manual_seed(seed)
    paddle.cuda.manual_seed_all(seed)


__all__ = [
    "seed_all",
    "skip_on_connection_issues",
    "skip_on_cuda_oom",
    "skip_on_running_out_of_memory",
]
_IS_WINDOWS = sys.platform.startswith("win32")


def _requirement_cache(req: str) -> bool:
    """Check if a package requirement is satisfied.

    Simple replacement for ``lightning_utilities.core.imports.RequirementCache``.
    """
    try:
        from packaging.requirements import Requirement as Req

        parsed = Req(req)
        dist = importlib.metadata.distribution(parsed.name)
        installed = Version(dist.version)
        return installed in parsed.specifier
    except (importlib.metadata.PackageNotFoundError, Exception):
        return False


_SKLEARN_GREATER_EQUAL_1_3 = _requirement_cache("scikit-learn>=1.3.0")
_SKLEARN_GREATER_EQUAL_1_7 = _requirement_cache("scikit-learn>=1.7.0")
_TORCH_LESS_THAN_2_1 = _requirement_cache("torch<2.1.0")
_TRANSFORMERS_RANGE_GE_4_50_LT_4_54 = _requirement_cache("transformers>=4.50.0,<4.54.0")
_TRANSFORMERS_GREATER_EQUAL_4_54 = _requirement_cache("transformers>=4.54.0")
_IS_LIGHTNING_CI = os.environ.get("LIGHTNING_CI", "0") == "1"
