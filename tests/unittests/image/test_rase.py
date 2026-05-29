from functools import partial
from typing import NamedTuple

import paddle
from paddle import Tensor
import pytest
import sewar
from unittests import BATCH_SIZE
from unittests._helpers.testers import MetricTester

from paddlemetrics.functional import relative_average_spectral_error
from paddlemetrics.functional.image.utils import _uniform_filter
from paddlemetrics.image import RelativeAverageSpectralError


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
    preds = paddle.rand(2, BATCH_SIZE, channel, size, size, dtype=dtype)
    target = paddle.rand(2, BATCH_SIZE, channel, size, size, dtype=dtype)
    _inputs.append(
        _InputWindowSized(preds=preds, target=target, window_size=window_size)
    )


def _reference_sewar_rase(preds, target, window_size):
    """Baseline implementation of metric.

    This custom implementation is necessary since sewar only supports single image and aggregation therefore needs
    adjustments.

    """
    target_sum = paddle.sum(
        _uniform_filter(target, window_size) / window_size**2, axis=0
    )
    target_mean = target_sum / target.shape[0]
    target_mean = target_mean.mean(0)
    preds = preds.permute(0, 2, 3, 1).numpy()
    target = target.permute(0, 2, 3, 1).numpy()
    rmse_mean = paddle.zeros(*preds.shape[1:])
    for pred, tgt in zip(preds, target):
        _, rmse_map = sewar.rmse_sw(tgt, pred, window_size)
        rmse_mean += rmse_map
    rmse_mean /= preds.shape[0]
    rase_map = 100 / target_mean * paddle.sqrt(paddle.mean(rmse_mean**2, 2))
    crop_slide = round(window_size / 2)
    return paddle.mean(rase_map[crop_slide:-crop_slide, crop_slide:-crop_slide])


@pytest.mark.parametrize(
    ("preds", "target", "window_size"),
    [(i.preds, i.target, i.window_size) for i in _inputs],
)
class TestRelativeAverageSpectralError(MetricTester):
    """Testing of Relative Average Spectral Error."""

    atol = 0.01

    @pytest.mark.parametrize("ddp", [False])
    def test_rase(self, preds, target, window_size, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            metric_class=RelativeAverageSpectralError,
            reference_metric=partial(_reference_sewar_rase, window_size=window_size),
            metric_args={"window_size": window_size},
            check_batch=False,
        )

    def test_rase_functional(self, preds, target, window_size):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=relative_average_spectral_error,
            reference_metric=partial(_reference_sewar_rase, window_size=window_size),
            metric_args={"window_size": window_size},
        )
