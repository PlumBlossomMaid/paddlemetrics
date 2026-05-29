from functools import partial

import paddle
import pytest
from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

from paddlemetrics.functional.image.sam import spectral_angle_mapper
from paddlemetrics.image.sam import SpectralAngleMapper

seed_all(42)
_inputs = []
for size, channel, dtype in [
    (12, 3, paddle.float32),
    (13, 3, paddle.float32),
    (14, 3, paddle.float64),
    (15, 3, paddle.float64),
]:
    preds = paddle.rand(NUM_BATCHES, BATCH_SIZE, channel, size, size, dtype=dtype)
    target = paddle.rand(NUM_BATCHES, BATCH_SIZE, channel, size, size, dtype=dtype)
    _inputs.append(_Input(preds=preds, target=target))


def _reference_sam(
    preds: paddle.Tensor, target: paddle.Tensor, reduction: str = "elementwise_mean"
) -> paddle.Tensor:
    """Baseline implementation of spectral angle mapper."""
    reduction_options = "elementwise_mean", "sum", "none"
    if reduction not in reduction_options:
        raise ValueError(
            f"reduction has to be one of {reduction_options}, got: {reduction}."
        )
    similarity = paddle.nn.functional.cosine_similarity(preds, target)
    sam_score = paddle.clamp(similarity, -1, 1).acos()
    if reduction == "sum":
        return paddle.sum(sam_score)
    if reduction == "elementwise_mean":
        return paddle.mean(sam_score)
    return sam_score


@pytest.mark.parametrize("reduction", ["sum", "elementwise_mean"])
@pytest.mark.parametrize(("preds", "target"), [(i.preds, i.target) for i in _inputs])
class TestSpectralAngleMapper(MetricTester):
    """Test class for `SpectralAngleMapper` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_sam(self, reduction, preds, target, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            metric_class=SpectralAngleMapper,
            reference_metric=partial(_reference_sam, reduction=reduction),
            metric_args={"reduction": reduction},
        )

    def test_sam_functional(self, reduction, preds, target):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=spectral_angle_mapper,
            reference_metric=partial(_reference_sam, reduction=reduction),
            metric_args={"reduction": reduction},
        )

    @pytest.mark.skipif(
        not True,
        reason="Pytoch below 2.1 does not support cpu + half precision used in SAM metric",
    )
    def test_sam_half_cpu(self, reduction, preds, target):
        """Test dtype support of the metric on CPU."""
        self.run_precision_test_cpu(
            preds, target, SpectralAngleMapper, spectral_angle_mapper
        )

    @pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires cuda")
    def test_sam_half_gpu(self, reduction, preds, target):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds, target, SpectralAngleMapper, spectral_angle_mapper
        )


def test_error_on_different_shape(metric_class=SpectralAngleMapper):
    """Test that error is raised if preds and target have different shape."""
    metric = metric_class()
    with pytest.raises(
        RuntimeError,
        match="Predictions and targets are expected to have the same shape.*",
    ):
        metric(paddle.randn([1, 3, 16, 16]), paddle.randn([1, 1, 16, 16]))


def test_error_on_invalid_shape(metric_class=SpectralAngleMapper):
    """Test that error is raised if input is not 4D."""
    metric = metric_class()
    with pytest.raises(
        ValueError, match="Expected `preds` and `target` to have BxCxHxW shape.*"
    ):
        metric(paddle.randn([3, 16, 16]), paddle.randn([3, 16, 16]))


def test_error_on_invalid_type(metric_class=SpectralAngleMapper):
    """Test that error is raised if preds and target have different dtype."""
    metric = metric_class()
    with pytest.raises(
        TypeError, match="Expected `preds` and `target` to have the same data type.*"
    ):
        metric(
            paddle.randn([3, 16, 16]), paddle.randn([3, 16, 16], dtype=paddle.float64)
        )


def test_error_on_grayscale_image(metric_class=SpectralAngleMapper):
    """Test that error is raised if number of channels is not larger than 1."""
    metric = metric_class()
    with pytest.raises(
        ValueError,
        match="Expected channel dimension of `preds` and `target` to be larger than 1.*",
    ):
        metric(paddle.randn([16, 1, 16, 16]), paddle.randn([16, 1, 16, 16]))
