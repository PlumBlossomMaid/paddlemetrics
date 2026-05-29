from functools import partial

import numpy as np
import paddle
import pytest
from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

from paddlemetrics.audio import ScaleInvariantSignalNoiseRatio
from paddlemetrics.functional.audio import scale_invariant_signal_noise_ratio

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


speechmetrics_sisdr = _SpeechMetricsSISDR()


def _reference_speechmetrics_si_sdr(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    zero_mean: bool = True,
    reduce_mean: bool = False,
):
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
    si_sdr = paddle.tensor(mss)
    if reduce_mean:
        return si_sdr.mean()
    return si_sdr


@pytest.mark.parametrize(
    ("preds", "target", "ref_metric"),
    [(inputs.preds, inputs.target, _reference_speechmetrics_si_sdr)],
)
class TestSISNR(MetricTester):
    """Test class for `ScaleInvariantSignalNoiseRatio` metric."""

    atol = 0.01

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_si_snr(self, preds, target, ref_metric, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            ScaleInvariantSignalNoiseRatio,
            reference_metric=partial(_reference_speechmetrics_si_sdr, reduce_mean=True),
        )

    def test_si_snr_functional(self, preds, target, ref_metric):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds, target, scale_invariant_signal_noise_ratio, ref_metric
        )

    def test_si_snr_differentiability(self, preds, target, ref_metric):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=ScaleInvariantSignalNoiseRatio,
            metric_functional=scale_invariant_signal_noise_ratio,
        )

    def test_si_snr_half_cpu(self, preds, target, ref_metric):
        """Test dtype support of the metric on CPU."""
        pytest.xfail("SI-SNR metric does not support cpu + half precision")

    @pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires cuda")
    def test_si_snr_half_gpu(self, preds, target, ref_metric):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=ScaleInvariantSignalNoiseRatio,
            metric_functional=scale_invariant_signal_noise_ratio,
        )


def test_error_on_different_shape(metric_class=ScaleInvariantSignalNoiseRatio):
    """Test that error is raised on different shapes of input."""
    metric = metric_class()
    with pytest.raises(
        RuntimeError,
        match="Predictions and targets are expected to have the same shape",
    ):
        metric(paddle.randn(100), paddle.randn(50))
