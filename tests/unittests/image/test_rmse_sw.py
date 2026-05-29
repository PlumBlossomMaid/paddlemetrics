from functools import partial
from typing import NamedTuple

import paddle
from paddle import Tensor
import pytest
import sewar
from unittests import BATCH_SIZE, NUM_BATCHES
from unittests._helpers.testers import MetricTester

from paddlemetrics.functional import \
    root_mean_squared_error_using_sliding_window
from paddlemetrics.image import RootMeanSquaredErrorUsingSlidingWindow


class _InputWindowSized(NamedTuple):
    preds: Tensor
    target: Tensor
    window_size: int


_inputs = []
for size, channel, window_size, dtype in [
    (12, 3, 3, paddle.float32),
    (13, 1, 4, paddle.float32),
    (14, 1, 5, paddle.float64),
    (15, 3, 8, paddle.float64),
]:
    preds = paddle.rand(NUM_BATCHES, BATCH_SIZE, channel, size, size, dtype=dtype)
    target = paddle.rand(NUM_BATCHES, BATCH_SIZE, channel, size, size, dtype=dtype)
    _inputs.append(
        _InputWindowSized(preds=preds, target=target, window_size=window_size)
    )


def _reference_sewar_rmse_sw(preds, target, window_size):
    rmse_mean = paddle.tensor(0.0, dtype=preds.dtype)
    preds = preds.permute(0, 2, 3, 1).numpy()
    target = target.permute(0, 2, 3, 1).numpy()
    for idx, (pred, tgt) in enumerate(zip(preds, target)):
        rmse, _ = sewar.rmse_sw(tgt, pred, window_size)
        rmse_mean += (rmse - rmse_mean) / (idx + 1)
    return rmse_mean


@pytest.mark.parametrize(
    ("preds", "target", "window_size"),
    [(i.preds, i.target, i.window_size) for i in _inputs],
)
class TestRootMeanSquareErrorWithSlidingWindow(MetricTester):
    """Testing of Root Mean Square Error With Sliding Window."""

    atol = 0.01

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_rmse_sw(self, preds, target, window_size, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            metric_class=RootMeanSquaredErrorUsingSlidingWindow,
            reference_metric=partial(_reference_sewar_rmse_sw, window_size=window_size),
            metric_args={"window_size": window_size},
        )

    def test_rmse_sw_functional(self, preds, target, window_size):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=root_mean_squared_error_using_sliding_window,
            reference_metric=partial(_reference_sewar_rmse_sw, window_size=window_size),
            metric_args={"window_size": window_size},
        )
