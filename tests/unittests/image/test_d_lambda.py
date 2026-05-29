from functools import partial
from typing import NamedTuple

import numpy as np
import paddle
from paddle import Tensor
import pytest
from unittests import BATCH_SIZE, NUM_BATCHES
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

from paddlemetrics.functional.image.d_lambda import spectral_distortion_index
from paddlemetrics.functional.image.uqi import universal_image_quality_index
from paddlemetrics.image.d_lambda import SpectralDistortionIndex

seed_all(42)


class _Input(NamedTuple):
    preds: Tensor
    target: Tensor
    p: int


_inputs = []
for size, channel, p, dtype in [
    (12, 3, 1, paddle.float32),
    (13, 1, 3, paddle.float32),
    (14, 1, 4, paddle.float64),
    (15, 3, 1, paddle.float64),
]:
    preds = paddle.rand(NUM_BATCHES, BATCH_SIZE, channel, size, size, dtype=dtype)
    target = paddle.rand(NUM_BATCHES, BATCH_SIZE, channel, size, size, dtype=dtype)
    _inputs.append(_Input(preds=preds, target=target, p=p))


def _baseline_d_lambda(preds: np.ndarray, target: np.ndarray, p: int = 1) -> float:
    """NumPy based implementation of Spectral Distortion Index, which uses UQI of TorchMetrics."""
    target, preds = paddle.from_numpy(target), paddle.from_numpy(preds)
    target = target.permute(0, 3, 1, 2)
    preds = preds.permute(0, 3, 1, 2)
    length = preds.shape[1]
    m1 = np.zeros((length, length), dtype=np.float32)
    m2 = np.zeros((length, length), dtype=np.float32)
    for k in range(length):
        for r in range(k, length):
            m1[k, r] = m1[r, k] = universal_image_quality_index(
                target[:, k : k + 1, :, :], target[:, r : r + 1, :, :]
            )
            m2[k, r] = m2[r, k] = universal_image_quality_index(
                preds[:, k : k + 1, :, :], preds[:, r : r + 1, :, :]
            )
    diff = np.abs(m1 - m2) ** p
    if length == 1:
        return diff[0][0] ** (1.0 / p)
    return (1.0 / (length * (length - 1)) * np.sum(diff)) ** (1.0 / p)


def _reference_numpy_d_lambda(preds, target, p):
    c, h, w = preds.shape[-3:]
    np_preds = preds.view(-1, c, h, w).permute(0, 2, 3, 1).numpy()
    np_target = target.view(-1, c, h, w).permute(0, 2, 3, 1).numpy()
    return _baseline_d_lambda(np_preds, np_target, p=p)


@pytest.mark.parametrize(
    ("preds", "target", "p"), [(i.preds, i.target, i.p) for i in _inputs]
)
class TestSpectralDistortionIndex(MetricTester):
    """Test class for `SpectralDistortionIndex` metric."""

    atol = 0.006

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_d_lambda(self, preds, target, p, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            metric_class=SpectralDistortionIndex,
            reference_metric=partial(_reference_numpy_d_lambda, p=p),
            metric_args={"p": p},
        )

    def test_d_lambda_functional(self, preds, target, p):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=spectral_distortion_index,
            reference_metric=partial(_reference_numpy_d_lambda, p=p),
            metric_args={"p": p},
        )

    @pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires cuda")
    def test_d_lambda_half_gpu(self, preds, target, p):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds, target, SpectralDistortionIndex, spectral_distortion_index, {"p": p}
        )


@pytest.mark.parametrize(
    ("preds", "target", "p", "match"),
    [
        (
            [1, 16, 16],
            [1, 16, 16],
            1,
            "Expected `preds` and `target` to have BxCxHxW shape.*",
        ),
        (
            [1, 1, 16, 16],
            [1, 1, 16, 16],
            0,
            "Expected `p` to be a positive integer. Got p: 0.",
        ),
        (
            [1, 1, 16, 16],
            [1, 1, 16, 16],
            -1,
            "Expected `p` to be a positive integer. Got p: -1.",
        ),
    ],
)
def test_d_lambda_invalid_inputs(preds, target, p, match):
    """Test that invalid input raises the correct errors."""
    preds_t = paddle.rand(preds)
    target_t = paddle.rand(target)
    with pytest.raises(ValueError, match=match):
        spectral_distortion_index(preds_t, target_t, p)


def test_d_lambda_invalid_type():
    """Test that error is raised on different dtypes."""
    preds_t = paddle.rand((1, 1, 16, 16))
    target_t = paddle.rand((1, 1, 16, 16), dtype=paddle.float64)
    with pytest.raises(
        TypeError, match="Expected `ms` and `fused` to have the same data type.*"
    ):
        spectral_distortion_index(preds_t, target_t, p=1)


def test_d_lambda_different_sizes():
    """Since d lambda is reference free, it can accept different number of targets and preds."""
    preds = paddle.rand(1, 1, 32, 32)
    target = paddle.rand(1, 1, 16, 16)
    out = spectral_distortion_index(preds, target, p=1)
    assert isinstance(out, paddle.Tensor)
