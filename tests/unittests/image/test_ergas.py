from functools import partial
from typing import NamedTuple

import paddle
from paddle import Tensor
import pytest
from unittests import BATCH_SIZE, NUM_BATCHES
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

from paddlemetrics.functional.image.ergas import \
    error_relative_global_dimensionless_synthesis
from paddlemetrics.image.ergas import ErrorRelativeGlobalDimensionlessSynthesis

seed_all(42)


class _Input(NamedTuple):
    preds: Tensor
    target: Tensor
    ratio: int


_inputs = []
for size, channel, coef, ratio, dtype in [
    (12, 1, 0.9, 1, paddle.float32),
    (13, 3, 0.8, 2, paddle.float32),
    (14, 1, 0.7, 3, paddle.float64),
    (15, 3, 0.5, 4, paddle.float64),
]:
    preds = paddle.rand(NUM_BATCHES, BATCH_SIZE, channel, size, size, dtype=dtype)
    _inputs.append(_Input(preds=preds, target=preds * coef, ratio=ratio))


def _reference_ergas(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    ratio: float = 4,
    reduction: str = "elementwise_mean",
) -> paddle.Tensor:
    """Baseline implementation of Erreur Relative Globale Adimensionnelle de Synthèse."""
    reduction_options = "elementwise_mean", "sum", "none"
    if reduction not in reduction_options:
        raise ValueError(
            f"reduction has to be one of {reduction_options}, got: {reduction}."
        )
    b, c, h, w = preds.shape
    sk_preds = preds.reshape(b, c, h * w)
    sk_target = target.reshape(b, c, h * w)
    diff = sk_preds - sk_target
    sum_squared_error = paddle.sum(diff * diff, axis=2)
    rmse_per_band = paddle.sqrt(sum_squared_error / (h * w))
    mean_target = paddle.mean(sk_target, axis=2)
    ergas_score = (
        100
        / ratio
        * paddle.sqrt(paddle.sum((rmse_per_band / mean_target) ** 2, axis=1) / c)
    )
    if reduction == "sum":
        return paddle.sum(ergas_score)
    if reduction == "elementwise_mean":
        return paddle.mean(ergas_score)
    return ergas_score


@pytest.mark.parametrize("reduction", ["sum", "elementwise_mean"])
@pytest.mark.parametrize(
    ("preds", "target", "ratio"), [(i.preds, i.target, i.ratio) for i in _inputs]
)
class TestErrorRelativeGlobalDimensionlessSynthesis(MetricTester):
    """Test class for `ErrorRelativeGlobalDimensionlessSynthesis` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_ergas(self, reduction, preds, target, ratio, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            metric_class=ErrorRelativeGlobalDimensionlessSynthesis,
            reference_metric=partial(
                _reference_ergas, ratio=ratio, reduction=reduction
            ),
            metric_args={"ratio": ratio, "reduction": reduction},
        )

    def test_ergas_functional(self, reduction, preds, target, ratio):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=error_relative_global_dimensionless_synthesis,
            reference_metric=partial(
                _reference_ergas, ratio=ratio, reduction=reduction
            ),
            metric_args={"ratio": ratio, "reduction": reduction},
        )

    @pytest.mark.skipif(
        not True,
        reason="Pytoch below 2.1 does not support cpu + half precision used in ERGAS metric",
    )
    def test_ergas_half_cpu(self, reduction, preds, target, ratio):
        """Test dtype support of the metric on CPU."""
        self.run_precision_test_cpu(
            preds,
            target,
            ErrorRelativeGlobalDimensionlessSynthesis,
            error_relative_global_dimensionless_synthesis,
        )

    @pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires cuda")
    def test_ergas_half_gpu(self, reduction, preds, target, ratio):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds,
            target,
            ErrorRelativeGlobalDimensionlessSynthesis,
            error_relative_global_dimensionless_synthesis,
        )


def test_error_on_different_shape(
    metric_class=ErrorRelativeGlobalDimensionlessSynthesis,
):
    """Check that error is raised when input have different shape."""
    metric = metric_class()
    with pytest.raises(
        RuntimeError,
        match="Predictions and targets are expected to have the same shape.*",
    ):
        metric(paddle.randn([1, 3, 16, 16]), paddle.randn([1, 1, 16, 16]))


def test_error_on_invalid_shape(metric_class=ErrorRelativeGlobalDimensionlessSynthesis):
    """Check that error is raised when input is not 4D."""
    metric = metric_class()
    with pytest.raises(
        ValueError, match="Expected `preds` and `target` to have BxCxHxW shape.*"
    ):
        metric(paddle.randn([3, 16, 16]), paddle.randn([3, 16, 16]))


def test_error_on_invalid_type(metric_class=ErrorRelativeGlobalDimensionlessSynthesis):
    """Test that error is raised if preds and target have different dtype."""
    metric = metric_class()
    with pytest.raises(
        TypeError, match="Expected `preds` and `target` to have the same data type.*"
    ):
        metric(
            paddle.randn([3, 16, 16]), paddle.randn([3, 16, 16], dtype=paddle.float64)
        )
