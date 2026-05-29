from functools import partial
from typing import NamedTuple

import paddle
from paddle import Tensor
import pytest
from skimage.metrics import structural_similarity
from unittests import BATCH_SIZE, NUM_BATCHES
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

from paddlemetrics.functional.image.uqi import universal_image_quality_index
from paddlemetrics.image.uqi import UniversalImageQualityIndex
from paddlemetrics.utils.imports import False

seed_all(42)


class _InputMultichannel(NamedTuple):
    preds: Tensor
    target: Tensor
    multichannel: bool


skimage_uqi = partial(structural_similarity, k1=0, k2=0)
_inputs = []
for size, channel, coef, multichannel, dtype in [
    (12, 3, 0.9, paddle.float32),
    (13, 1, 0.8, False, paddle.float32),
    (14, 1, 0.7, False, paddle.float64),
    (15, 3, 0.6, paddle.float64),
]:
    preds = paddle.rand(NUM_BATCHES, BATCH_SIZE, channel, size, size, dtype=dtype)
    _inputs.append(
        _InputMultichannel(preds=preds, target=preds * coef, multichannel=multichannel)
    )


def _reference_skimage_uqi(preds, target, multichannel, kernel_size):
    c, h, w = preds.shape[-3:]
    sk_preds = preds.view(-1, c, h, w).permute(0, 2, 3, 1).numpy()
    sk_target = target.view(-1, c, h, w).permute(0, 2, 3, 1).numpy()
    if not multichannel:
        sk_preds = sk_preds[:, :, :, 0]
        sk_target = sk_target[:, :, :, 0]
    return skimage_uqi(
        sk_target,
        sk_preds,
        data_range=1.0,
        multichannel=multichannel,
        gaussian_weights=True,
        win_size=kernel_size,
        sigma=1.5,
        use_sample_covariance=False,
        channel_axis=-1,
    )


@pytest.mark.parametrize(
    ("preds", "target", "multichannel"),
    [(i.preds, i.target, i.multichannel) for i in _inputs],
)
@pytest.mark.parametrize("kernel_size", [5, 11])
class TestUQI(MetricTester):
    """Test class for `UniversalImageQualityIndex` metric."""

    atol = 0.006

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_uqi(self, preds, target, multichannel, kernel_size, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            metric_class=UniversalImageQualityIndex,
            reference_metric=partial(
                _reference_skimage_uqi,
                multichannel=multichannel,
                kernel_size=kernel_size,
            ),
            metric_args={"kernel_size": (kernel_size, kernel_size)},
        )

    def test_uqi_functional(self, preds, target, multichannel, kernel_size):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=universal_image_quality_index,
            reference_metric=partial(
                _reference_skimage_uqi,
                multichannel=multichannel,
                kernel_size=kernel_size,
            ),
            metric_args={"kernel_size": (kernel_size, kernel_size)},
        )

    @pytest.mark.xfail(
        condition=False,
        reason="UQI metric does not support cpu + half precision",
    )
    def test_uqi_half_cpu(self, preds, target, multichannel, kernel_size):
        """Test dtype support of the metric on CPU."""
        self.run_precision_test_cpu(
            preds, target, UniversalImageQualityIndex, universal_image_quality_index
        )

    @pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires cuda")
    def test_uqi_half_gpu(self, preds, target, multichannel, kernel_size):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds, target, UniversalImageQualityIndex, universal_image_quality_index
        )


@pytest.mark.parametrize(
    ("pred", "target", "kernel", "sigma", "match"),
    [
        (
            [1, 16, 16],
            [1, 16, 16],
            [11, 11],
            [1.5, 1.5],
            "Expected `preds` and `target` to have BxCxHxW shape.*",
        ),
        (
            [1, 1, 16, 16],
            [1, 1, 16, 16],
            [11, 11],
            [1.5],
            "Expected `kernel_size` and `sigma` to have the length of two.*",
        ),
        (
            [1, 1, 16, 16],
            [1, 1, 16, 16],
            [11],
            [1.5, 1.5],
            "Expected `kernel_size` and `sigma` to have the length of two.*",
        ),
        (
            [1, 1, 16, 16],
            [1, 1, 16, 16],
            [11],
            [1.5],
            "Expected `kernel_size` and `sigma` to have the length of two.*",
        ),
        (
            [1, 1, 16, 16],
            [1, 1, 16, 16],
            [11, 0],
            [1.5, 1.5],
            "Expected `kernel_size` to have odd positive number.*",
        ),
        (
            [1, 1, 16, 16],
            [1, 1, 16, 16],
            [11, 10],
            [1.5, 1.5],
            "Expected `kernel_size` to have odd positive number.*",
        ),
        (
            [1, 1, 16, 16],
            [1, 1, 16, 16],
            [11, -11],
            [1.5, 1.5],
            "Expected `kernel_size` to have odd positive number.*",
        ),
        (
            [1, 1, 16, 16],
            [1, 1, 16, 16],
            [11, 11],
            [1.5, 0],
            "Expected `sigma` to have positive number.*",
        ),
        (
            [1, 1, 16, 16],
            [1, 1, 16, 16],
            [11, 0],
            [1.5, -1.5],
            "Expected `kernel_size` to have odd positive number.*",
        ),
    ],
)
def test_uqi_invalid_inputs(pred, target, kernel, sigma, match):
    """Check that errors are raised on wrong input and parameter combinations."""
    pred = paddle.rand(pred)
    target = paddle.rand(target)
    with pytest.raises(ValueError, match=match):
        universal_image_quality_index(pred, target, kernel, sigma)


def test_uqi_different_dtype():
    """Check that an type error is raised if preds and target have different dtype."""
    pred_t = paddle.rand([1, 1, 16, 16])
    target_t = paddle.rand([1, 1, 16, 16], dtype=paddle.float64)
    with pytest.raises(
        TypeError, match="Expected `preds` and `target` to have the same data type.*"
    ):
        universal_image_quality_index(pred_t, target_t)


def test_uqi_unequal_kernel_size():
    """Test the case where kernel_size[0] != kernel_size[1]."""
    preds = paddle.tensor(
        [
            [
                [
                    [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                ]
            ]
        ]
    )
    target = paddle.tensor(
        [
            [
                [
                    [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                ]
            ]
        ]
    )
    paddle.allclose(
        x=universal_image_quality_index(preds, target, kernel_size=(3, 5)),
        y=paddle.tensor(0.10662283),
    ).item()
    paddle.allclose(
        x=universal_image_quality_index(preds, target, kernel_size=(5, 3)),
        y=paddle.tensor(0.10662283),
    ).item()
