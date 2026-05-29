from functools import partial

import paddle
import pytest
from mir_eval.separation import bss_eval_images as mir_eval_bss_eval_images
from unittests import _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester
from unittests.audio import _average_metric_wrapper

from paddlemetrics.audio import SignalNoiseRatio
from paddlemetrics.functional.audio import signal_noise_ratio

seed_all(42)
inputs = _Input(preds=paddle.rand(2, 1, 1, 25), target=paddle.rand(2, 1, 1, 25))


def _reference_bss_snr(preds: paddle.Tensor, target: paddle.Tensor, zero_mean: bool):
    if zero_mean:
        target = target - paddle.mean(target, axis=-1, keepdim=True)
        preds = preds - paddle.mean(preds, axis=-1, keepdim=True)
    target = target.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    mss = []
    for i in range(preds.shape[0]):
        ms = []
        for j in range(preds.shape[1]):
            snr_v = mir_eval_bss_eval_images(
                [target[i, j]], [preds[i, j]], compute_permutation=True
            )[0][0]
            ms.append(snr_v)
        mss.append(ms)
    return paddle.tensor(mss)


@pytest.mark.parametrize(
    ("preds", "target", "ref_metric", "zero_mean"),
    [
        (
            inputs.preds,
            inputs.target,
            partial(_reference_bss_snr, zero_mean=True),
            True,
        ),
        (
            inputs.preds,
            inputs.target,
            partial(_reference_bss_snr, zero_mean=False),
            False,
        ),
    ],
)
class TestSNR(MetricTester):
    """Test class for `SignalNoiseRatio` metric."""

    atol = 0.01

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_snr(self, preds, target, ref_metric, zero_mean, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            SignalNoiseRatio,
            reference_metric=partial(_average_metric_wrapper, metric_func=ref_metric),
            metric_args={"zero_mean": zero_mean},
        )

    def test_snr_functional(self, preds, target, ref_metric, zero_mean):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            signal_noise_ratio,
            ref_metric,
            metric_args={"zero_mean": zero_mean},
        )

    def test_snr_differentiability(self, preds, target, ref_metric, zero_mean):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=SignalNoiseRatio,
            metric_functional=signal_noise_ratio,
            metric_args={"zero_mean": zero_mean},
        )

    def test_snr_half_cpu(self, preds, target, ref_metric, zero_mean):
        """Test dtype support of the metric on CPU."""
        pytest.xfail("SNR metric does not support cpu + half precision")

    @pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires cuda")
    def test_snr_half_gpu(self, preds, target, ref_metric, zero_mean):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=SignalNoiseRatio,
            metric_functional=signal_noise_ratio,
            metric_args={"zero_mean": zero_mean},
        )


def test_error_on_different_shape(metric_class=SignalNoiseRatio):
    """Test that error is raised on different shapes of input."""
    metric = metric_class()
    with pytest.raises(
        RuntimeError,
        match="Predictions and targets are expected to have the same shape",
    ):
        metric(paddle.randn(100), paddle.randn(50))
