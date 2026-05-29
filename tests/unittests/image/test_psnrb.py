import sys

import math
from functools import partial

import numpy as np
import paddle
import pytest
from sewar.utils import _compute_bef
from unittests import BATCH_SIZE, NUM_BATCHES
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

from paddlemetrics.functional.image.psnrb import \
    peak_signal_noise_ratio_with_blocked_effect
from paddlemetrics.image import PeakSignalNoiseRatioWithBlockedEffect

seed_all(42)
_input = (
    paddle.rand(NUM_BATCHES, BATCH_SIZE, 1, 16, 16),
    paddle.rand(NUM_BATCHES, BATCH_SIZE, 1, 16, 16),
), (
    paddle.randint(low=0, high=255, shape=(NUM_BATCHES, BATCH_SIZE, 1, 16, 16)),
    paddle.randint(low=0, high=255, shape=(NUM_BATCHES, BATCH_SIZE, 1, 16, 16)),
)


def _reference_psnrb(preds, target, data_range):
    """Reference implementation of PSNRB metric.

    Inspired by
    https://github.com/andrewekhalel/sewar/blob/master/sewar/full_ref.py
    that also supports batched inputs.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        data_range: Range of the data. If not provided, it's determined from the data.

    Returns:
        PSNRB score

    """
    preds = preds.numpy()
    target = target.numpy()
    if isinstance(data_range, tuple):
        preds = np.clip(preds, data_range[0], data_range[1])
        target = np.clip(target, data_range[0], data_range[1])
        dr = data_range[1] - data_range[0]
    else:
        dr = float(data_range)
    imdff = np.double(target) - np.double(preds)
    mse = np.mean(np.square(imdff.flatten()))
    bef = sum([_compute_bef(p.squeeze()) for p in preds])
    mse_b = mse + bef
    return 10 * math.log10(dr**2 / mse_b)


@pytest.mark.parametrize(("preds", "target"), _input)
class TestPSNR(MetricTester):
    """Test class for PSNRB metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_psnr(self, preds, target, ddp):
        """Test that modular PSNRB metric returns the same result as the reference implementation."""
        data_range = 1.0 if preds._max() <= 1.0 else 255.0
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            metric_class=PeakSignalNoiseRatioWithBlockedEffect,
            reference_metric=partial(_reference_psnrb, data_range=data_range),
            metric_args={"data_range": data_range},
        )

    def test_psnr_functional(self, preds, target):
        """Test that functional PSNRB metric returns the same result as the reference implementation."""
        data_range = 1.0 if preds._max() <= 1.0 else 255.0
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=peak_signal_noise_ratio_with_blocked_effect,
            reference_metric=partial(_reference_psnrb, data_range=data_range),
            metric_args={"data_range": data_range},
        )

    def test_psnr_half_cpu(self, preds, target):
        """Test that PSNRB metric works with half precision on cpu."""
        if target._max() - target._min() < 2:
            pytest.xfail("PSNRB metric does not support cpu + half precision")
        data_range = 1.0 if preds._max() <= 1.0 else 255.0
        self.run_precision_test_cpu(
            preds,
            target,
            PeakSignalNoiseRatioWithBlockedEffect,
            peak_signal_noise_ratio_with_blocked_effect,
            metric_args={"data_range": data_range},
        )

    @pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires cuda")
    def test_psnr_half_gpu(self, preds, target):
        """Test that PSNRB metric works with half precision on gpu."""
        data_range = 1.0 if preds._max() <= 1.0 else 255.0
        self.run_precision_test_gpu(
            preds,
            target,
            PeakSignalNoiseRatioWithBlockedEffect,
            peak_signal_noise_ratio_with_blocked_effect,
            metric_args={"data_range": data_range},
        )


def test_error_on_color_images():
    """Test that appropriate error is raised when color images are passed to PSNRB metric."""
    with pytest.raises(ValueError, match="`psnrb` metric expects grayscale images.*"):
        peak_signal_noise_ratio_with_blocked_effect(
            paddle.rand(1, 3, 16, 16), paddle.rand(1, 3, 16, 16), data_range=1.0
        )
