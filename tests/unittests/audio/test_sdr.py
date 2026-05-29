from functools import partial

import numpy as np
import paddle
import pytest
from mir_eval.separation import bss_eval_sources
from scipy.io import wavfile
from unittests import _Input
from unittests._helpers import _IS_LIGHTNING_CI, seed_all
from unittests._helpers.testers import MetricTester
from unittests.audio import (_SAMPLE_AUDIO_SPEECH, _SAMPLE_AUDIO_SPEECH_BAB_DB,
                             _SAMPLE_NUMPY_ISSUE_895)

from paddlemetrics.audio import SignalDistortionRatio
from paddlemetrics.functional import signal_distortion_ratio

seed_all(42)
inputs_1spk = _Input(preds=paddle.rand(2, 1, 1, 500), target=paddle.rand(2, 1, 1, 500))
inputs_2spk = _Input(preds=paddle.rand(2, 1, 2, 500), target=paddle.rand(2, 1, 2, 500))


def _reference_sdr_batch(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    compute_permutation: bool = False,
    reduce_mean: bool = False,
) -> paddle.Tensor:
    target = target.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    mss = []
    for b in range(preds.shape[0]):
        sdr_val_np, _, _, _ = bss_eval_sources(target[b], preds[b], compute_permutation)
        mss.append(sdr_val_np)
    sdr = paddle.tensor(np.array(mss))
    if reduce_mean:
        return sdr.mean()
    return sdr


@pytest.mark.parametrize(
    ("preds", "target"),
    [(inputs_1spk.preds, inputs_1spk.target), (inputs_2spk.preds, inputs_2spk.target)],
)
class TestSDR(MetricTester):
    """Test class for `SignalDistortionRatio` metric."""

    atol = 0.01

    @pytest.mark.parametrize(
        "ddp", [pytest.param(True, marks=[pytest.mark.DDP]), False]
    )
    @pytest.mark.skipif(
        _IS_LIGHTNING_CI, reason="test too slow and unreliable on Lightning CI"
    )
    def test_sdr(self, preds, target, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            SignalDistortionRatio,
            reference_metric=partial(_reference_sdr_batch, reduce_mean=True),
            metric_args={},
        )

    def test_sdr_functional(self, preds, target):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds, target, signal_distortion_ratio, _reference_sdr_batch, metric_args={}
        )

    def test_sdr_differentiability(self, preds, target):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=SignalDistortionRatio,
            metric_args={},
        )

    def test_sdr_half_cpu(self, preds, target):
        """Test dtype support of the metric on CPU."""
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=SignalDistortionRatio,
            metric_functional=signal_distortion_ratio,
            metric_args={},
        )

    @pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires cuda")
    def test_sdr_half_gpu(self, preds, target):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=SignalDistortionRatio,
            metric_functional=signal_distortion_ratio,
            metric_args={},
        )


def test_error_on_different_shape(metric_class=SignalDistortionRatio):
    """Test that error is raised on different shapes of input."""
    metric = metric_class()
    with pytest.raises(
        RuntimeError,
        match="Predictions and targets are expected to have the same shape",
    ):
        metric(paddle.randn(100), paddle.randn(50))


def test_on_real_audio():
    """Test that metric works on real audio signal."""
    _, ref = wavfile.read(_SAMPLE_AUDIO_SPEECH)
    _, deg = wavfile.read(_SAMPLE_AUDIO_SPEECH_BAB_DB)
    sdr = signal_distortion_ratio(paddle.from_numpy(deg), paddle.from_numpy(ref))
    assert paddle.allclose(
        x=sdr.float(), y=paddle.tensor(0.2211), rtol=0.0001, atol=0.0001
    ).item()


def test_too_low_precision():
    """Corner case where the precision of the input is important."""
    data = np.load(_SAMPLE_NUMPY_ISSUE_895)
    preds = paddle.tensor(data["preds"])
    target = paddle.tensor(data["target"])
    sdr_tm = signal_distortion_ratio(preds, target).double()
    sdr_bss, _, _, _ = bss_eval_sources(target.numpy(), preds.numpy(), False)
    assert paddle.allclose(
        x=sdr_tm.mean(), y=paddle.tensor(sdr_bss).mean(), rtol=0.0001, atol=0.01
    ).item()
