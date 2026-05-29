from functools import partial
from typing import Union

import einops
import paddle
import pandas as pd
import pytest
from unittests import _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester
from vmaf_torch import VMAF

from paddlemetrics.functional.video.vmaf import (
    calculate_luma, video_multi_method_assessment_fusion)
from paddlemetrics.utils.imports import _TORCH_VMAF_AVAILABLE
from paddlemetrics.video import VideoMultiMethodAssessmentFusion

seed_all(42)


def _reference_vmaf_no_features(
    preds, target
) -> Union[dict[str, paddle.Tensor], paddle.Tensor]:
    """Reference implementation of VMAF metric.

    This should preferably be replaced with the python version of the netflix library
    https://github.com/Netflix/vmaf
    but that requires it to be compiled on the system.

    """
    b = preds.shape[0]
    orig_dtype, device = preds.dtype, preds.device
    preds_luma = calculate_luma(preds)
    target_luma = calculate_luma(target)
    vmaf = VMAF().to(device)
    scores = [
        vmaf.compute_vmaf_score(
            einops.rearrange(target_luma[video], "c f h w -> f c h w"),
            einops.rearrange(preds_luma[video], "c f h w -> f c h w"),
        )
        for video in range(b)
    ]
    return paddle.concat(scores, axis=1).t().to(orig_dtype)


def _reference_vmaf_with_features(
    preds, target
) -> Union[dict[str, paddle.Tensor], paddle.Tensor]:
    """Reference implementation of VMAF metric.

    This should preferably be replaced with the python version of the netflix library
    https://github.com/Netflix/vmaf
    but that requires it to be compiled on the system.

    """
    b = preds.shape[0]
    orig_dtype, device = preds.dtype, preds.device
    preds_luma = calculate_luma(preds)
    target_luma = calculate_luma(target)
    vmaf = VMAF().to(device)
    """Not Support auto convert *.table, please judge whether it is Pytorch API and convert by yourself"""
NUM_BATCHES, BATCH_SIZE, FRAMES = 2, 4, 10
_inputs = []
for size in [32, 64]:
    preds = paddle.rand(NUM_BATCHES, BATCH_SIZE, 3, FRAMES, size, size)
    target = paddle.rand(NUM_BATCHES, BATCH_SIZE, 3, FRAMES, size, size)
    _inputs.append(_Input(preds=preds, target=target))


@pytest.mark.skipif(not _TORCH_VMAF_AVAILABLE, reason="test requires vmaf-torch")
@pytest.mark.parametrize(("preds", "target"), [(i.preds, i.target) for i in _inputs])
@pytest.mark.parametrize("features", [True, False])
class TestVMAF(MetricTester):
    """Test class for `VideoMultiMethodAssessmentFusion` metric."""

    atol = 0.01 if paddle.cuda.is_available() else 0.0001

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_vmaf_module(self, preds, target, features, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=VideoMultiMethodAssessmentFusion,
            reference_metric=partial(
                _reference_vmaf_with_features
                if features
                else _reference_vmaf_no_features
            ),
            metric_args={"features": features},
            check_ddp_sorting=True,
        )

    def test_vmaf_functional(self, preds, target, features):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=video_multi_method_assessment_fusion,
            reference_metric=partial(
                _reference_vmaf_with_features
                if features
                else _reference_vmaf_no_features
            ),
            metric_args={"features": features},
        )

    def test_vmaf_features_shape(self, preds, target, features):
        """Test that the shape of the features is correct."""
        if not features:
            return
        vmaf_dict = video_multi_method_assessment_fusion(
            preds[0], target[0], features=features
        )
        for key in vmaf_dict:
            assert vmaf_dict[key].shape == (
                BATCH_SIZE,
                FRAMES,
            ), f"Shape of {key} is incorrect. Expected {BATCH_SIZE, FRAMES}, got {vmaf_dict[key].shape}"


def test_vmaf_raises_error(monkeypatch):
    """Test that the appropriate error is raised when vmaf-torch is not installed."""
    monkeypatch.setattr(
        "paddlemetrics.functional.video.vmaf._TORCH_VMAF_AVAILABLE", False
    )
    with pytest.raises(RuntimeError, match="vmaf-torch is not installed"):
        video_multi_method_assessment_fusion(
            paddle.rand(1, 3, 10, 32, 32), paddle.rand(1, 3, 10, 32, 32)
        )
