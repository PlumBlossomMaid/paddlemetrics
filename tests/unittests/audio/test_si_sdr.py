from functools import partial

import numpy as np
import paddle
import pytest
from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester
from unittests.audio import _average_metric_wrapper

from paddlemetrics.audio import ScaleInvariantSignalDistortionRatio
from paddlemetrics.functional.audio import \
    scale_invariant_signal_distortion_ratio

seed_all(42)
NUM_SAMPLES = 100
inputs = _Input(
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE, 1, NUM_SAMPLES),
    target=paddle.rand(NUM_BATCHES, BATCH_SIZE, 1, NUM_SAMPLES),
)


class _SpeechMetricsSISDR:
    """The code from speechmetrics."""

    def __init__(self) -> None:
        ...

    def _test_window(self, audios, rate):
        eps = np.finfo(audios[0].dtype).eps
        reference = audios[1].reshape(audios[1].size, 1)
        estimate = audios[0].reshape(audios[0].size, 1)
        rss = np.dot(reference.T, reference)
        a = (eps + np.dot(reference.T, estimate)) / (rss + eps)
        e_true = a * reference
        e_res = estimate - e_true
        sss = (e_true**2).sum()
        snn = (e_res**2).sum()
        return {"sisdr": 10 * np.log10((eps + sss) / (eps + snn))}


def _reference_speechmetrics_si_sdr(
    preds: paddle.Tensor, target: paddle.Tensor, zero_mean: bool
):
    speechmetrics_sisdr = _SpeechMetricsSISDR()
    if zero_mean:
        preds = preds - preds.mean(dim=2, keepdim=True)
        target = target - target.mean(dim=2, keepdim=True)
    target = target.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    mss = []
    for i in range(preds.shape[0]):
        ms = []
        for j in range(preds.shape[1]):
            metric = speechmetrics_sisdr._test_window(
                [preds[i, j], target[i, j]], rate=16000
            )
            ms.append(metric["sisdr"])
        mss.append(ms)
    return paddle.tensor(mss)


@pytest.mark.parametrize(
    ("preds", "target", "ref_metric", "zero_mean"),
    [
        (
            inputs.preds,
            inputs.target,
            partial(_reference_speechmetrics_si_sdr, zero_mean=True),
            True,
        ),
        (
            inputs.preds,
            inputs.target,
            partial(_reference_speechmetrics_si_sdr, zero_mean=False),
            False,
        ),
    ],
)
class TestSISDR(MetricTester):
    """Test class for `ScaleInvariantSignalDistortionRatio` metric."""

    atol = 0.01

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_si_sdr(self, preds, target, ref_metric, zero_mean, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            ScaleInvariantSignalDistortionRatio,
            reference_metric=partial(_average_metric_wrapper, metric_func=ref_metric),
            metric_args={"zero_mean": zero_mean},
        )

    def test_si_sdr_functional(self, preds, target, ref_metric, zero_mean):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            scale_invariant_signal_distortion_ratio,
            ref_metric,
            metric_args={"zero_mean": zero_mean},
        )

    def test_si_sdr_differentiability(self, preds, target, ref_metric, zero_mean):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=ScaleInvariantSignalDistortionRatio,
            metric_functional=scale_invariant_signal_distortion_ratio,
            metric_args={"zero_mean": zero_mean},
        )

    def test_si_sdr_half_cpu(self, preds, target, ref_metric, zero_mean):
        """Test dtype support of the metric on CPU."""
        pytest.xfail("SI-SDR metric does not support cpu + half precision")

    @pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires cuda")
    def test_si_sdr_half_gpu(self, preds, target, ref_metric, zero_mean):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=ScaleInvariantSignalDistortionRatio,
            metric_functional=scale_invariant_signal_distortion_ratio,
            metric_args={"zero_mean": zero_mean},
        )


def test_error_on_different_shape(metric_class=ScaleInvariantSignalDistortionRatio):
    """Test that error is raised on different shapes of input."""
    metric = metric_class()
    with pytest.raises(
        RuntimeError,
        match="Predictions and targets are expected to have the same shape",
    ):
        metric(paddle.randn(100), paddle.randn(50))
