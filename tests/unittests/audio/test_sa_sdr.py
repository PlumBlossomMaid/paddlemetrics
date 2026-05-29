from functools import partial

import paddle
import pytest
from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

from paddlemetrics.audio import SourceAggregatedSignalDistortionRatio
from paddlemetrics.functional.audio import (
    scale_invariant_signal_distortion_ratio, signal_noise_ratio,
    source_aggregated_signal_distortion_ratio)

seed_all(42)
NUM_SAMPLES = 100
inputs = _Input(
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE, 2, NUM_SAMPLES),
    target=paddle.rand(NUM_BATCHES, BATCH_SIZE, 2, NUM_SAMPLES),
)


def _reference_local_sa_sdr(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    scale_invariant: bool,
    zero_mean: bool,
    reduce_mean: bool = False,
):
    if zero_mean:
        target = target - paddle.mean(target, axis=-1, keepdim=True)
        preds = preds - paddle.mean(preds, axis=-1, keepdim=True)
    preds = preds.reshape(preds.shape[0], preds.shape[1] * preds.shape[2])
    target = target.reshape(target.shape[0], target.shape[1] * target.shape[2])
    if scale_invariant:
        sa_sdr = scale_invariant_signal_distortion_ratio(
            preds=preds, target=target, zero_mean=False
        )
    else:
        sa_sdr = signal_noise_ratio(preds=preds, target=target, zero_mean=zero_mean)
    if reduce_mean:
        return sa_sdr.mean()
    return sa_sdr


@pytest.mark.parametrize(
    ("preds", "target", "scale_invariant", "zero_mean"),
    [
        (inputs.preds, inputs.target, False),
        (inputs.preds, inputs.target),
        (inputs.preds, inputs.target, False, False),
        (inputs.preds, inputs.target, False),
    ],
)
class TestSASDR(MetricTester):
    """Test class for `SourceAggregatedSignalDistortionRatio` metric."""

    atol = 0.01

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_si_sdr(self, preds, target, scale_invariant, zero_mean, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            SourceAggregatedSignalDistortionRatio,
            reference_metric=partial(
                _reference_local_sa_sdr,
                scale_invariant=scale_invariant,
                zero_mean=zero_mean,
                reduce_mean=True,
            ),
            metric_args={"scale_invariant": scale_invariant, "zero_mean": zero_mean},
        )

    def test_sa_sdr_functional(self, preds, target, scale_invariant, zero_mean):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            source_aggregated_signal_distortion_ratio,
            reference_metric=partial(
                _reference_local_sa_sdr,
                scale_invariant=scale_invariant,
                zero_mean=zero_mean,
            ),
            metric_args={"scale_invariant": scale_invariant, "zero_mean": zero_mean},
        )

    def test_sa_sdr_differentiability(self, preds, target, scale_invariant, zero_mean):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=SourceAggregatedSignalDistortionRatio,
            metric_functional=source_aggregated_signal_distortion_ratio,
            metric_args={"scale_invariant": scale_invariant, "zero_mean": zero_mean},
        )

    def test_sa_sdr_half_cpu(self, preds, target, scale_invariant, zero_mean):
        """Test dtype support of the metric on CPU."""
        pytest.xfail("SA-SDR metric does not support cpu + half precision")

    @pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires cuda")
    def test_sa_sdr_half_gpu(self, preds, target, scale_invariant, zero_mean):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=SourceAggregatedSignalDistortionRatio,
            metric_functional=source_aggregated_signal_distortion_ratio,
            metric_args={"scale_invariant": scale_invariant, "zero_mean": zero_mean},
        )


def test_error_on_shape(metric_class=SourceAggregatedSignalDistortionRatio):
    """Test that error is raised on different shapes of input."""
    metric = metric_class()
    with pytest.raises(
        RuntimeError,
        match="Predictions and targets are expected to have the same shape",
    ):
        metric(paddle.randn(100), paddle.randn(50))
    with pytest.raises(
        RuntimeError,
        match="The preds and target should have the shape (..., spk, time)*",
    ):
        metric(paddle.randn(100), paddle.randn(100))
