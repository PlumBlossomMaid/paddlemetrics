from functools import partial

import numpy as np
import paddle
import pytest
from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

from paddlemetrics.functional.regression.log_cosh import log_cosh_error
from paddlemetrics.regression.log_cosh import LogCoshError

seed_all(42)
NUM_TARGETS = 5
_single_target_inputs = _Input(
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE),
    target=paddle.rand(NUM_BATCHES, BATCH_SIZE),
)
_multi_target_inputs = _Input(
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE, NUM_TARGETS),
    target=paddle.rand(NUM_BATCHES, BATCH_SIZE, NUM_TARGETS),
)


def _reference_log_cosh_error(preds, target):
    preds, target = preds.numpy(), target.numpy()
    diff = preds - target
    if diff.ndim == 1:
        return np.mean(np.log((np.exp(diff) + np.exp(-diff)) / 2))
    return np.mean(np.log((np.exp(diff) + np.exp(-diff)) / 2), axis=0)


@pytest.mark.parametrize(
    ("preds", "target"),
    [
        (_single_target_inputs.preds, _single_target_inputs.target),
        (_multi_target_inputs.preds, _multi_target_inputs.target),
    ],
)
class TestLogCoshError(MetricTester):
    """Test class for `LogCoshError` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_log_cosh_error_class(self, ddp, preds, target):
        """Test class implementation of metric."""
        num_outputs = 1 if preds.ndim == 2 else NUM_TARGETS
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=LogCoshError,
            reference_metric=_reference_log_cosh_error,
            metric_args={"num_outputs": num_outputs},
        )

    def test_log_cosh_error_functional(self, preds, target):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=log_cosh_error,
            reference_metric=_reference_log_cosh_error,
        )

    def test_log_cosh_error_differentiability(self, preds, target):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        num_outputs = 1 if preds.ndim == 2 else NUM_TARGETS
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=partial(LogCoshError, num_outputs=num_outputs),
            metric_functional=log_cosh_error,
        )
