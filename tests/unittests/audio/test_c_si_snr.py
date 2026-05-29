import paddle
import pytest
from scipy.io import wavfile
from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester
from unittests.audio import _SAMPLE_AUDIO_SPEECH, _SAMPLE_AUDIO_SPEECH_BAB_DB

from paddlemetrics.audio import ComplexScaleInvariantSignalNoiseRatio
from paddlemetrics.functional.audio import \
    complex_scale_invariant_signal_noise_ratio

seed_all(42)
inputs = _Input(
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE, 129, 20, 2),
    target=paddle.rand(NUM_BATCHES, BATCH_SIZE, 129, 20, 2),
)


@pytest.mark.parametrize(
    ("preds", "target", "ref_metric", "zero_mean"),
    [
        (inputs.preds, inputs.target, None),
        (inputs.preds, inputs.target, None, False),
    ],
)
class TestComplexSISNR(MetricTester):
    """Test class for `ComplexScaleInvariantSignalNoiseRatio` metric."""

    atol = 0.01

    def test_c_si_snr_differentiability(self, preds, target, ref_metric, zero_mean):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=ComplexScaleInvariantSignalNoiseRatio,
            metric_functional=complex_scale_invariant_signal_noise_ratio,
            metric_args={"zero_mean": zero_mean},
        )

    def test_c_si_sdr_half_cpu(self, preds, target, ref_metric, zero_mean):
        """Test dtype support of the metric on CPU."""
        pytest.xfail("C-SI-SDR metric does not support cpu + half precision")

    def test_c_si_sdr_half_gpu(self, preds, target, ref_metric, zero_mean):
        """Test dtype support of the metric on GPU."""
        pytest.xfail("C-SI-SDR metric does not support gpu + half precision")


def test_on_real_audio():
    """Test that metric works as expected on real audio signals."""
    rate, ref = wavfile.read(_SAMPLE_AUDIO_SPEECH)
    rate, deg = wavfile.read(_SAMPLE_AUDIO_SPEECH_BAB_DB)
    ref = paddle.tensor(ref, dtype=paddle.float32)
    deg = paddle.tensor(deg, dtype=paddle.float32)
    ref_stft = paddle.signal.stft(x=ref, n_fft=256, hop_length=128)
    deg_stft = paddle.signal.stft(x=deg, n_fft=256, hop_length=128)
    v = complex_scale_invariant_signal_noise_ratio(deg_stft, ref_stft, zero_mean=False)
    assert paddle.allclose(
        x=v, y=paddle.tensor(0.03019072115421295, dtype=v.dtype), atol=0.0001
    ).item(), v
    v = complex_scale_invariant_signal_noise_ratio(deg_stft, ref_stft, zero_mean=True)
    assert paddle.allclose(
        x=v, y=paddle.tensor(0.030391741544008255, dtype=v.dtype), atol=0.0001
    ).item(), v


def test_error_on_incorrect_shape(metric_class=ComplexScaleInvariantSignalNoiseRatio):
    """Test that error is raised on incorrect shapes of input."""
    metric = metric_class()
    with pytest.raises(
        RuntimeError,
        match="Predictions and targets are expected to have the shape (..., frequency, time, 2)*",
    ):
        metric(paddle.randn(100), paddle.randn(50))


def test_error_on_different_shape(metric_class=ComplexScaleInvariantSignalNoiseRatio):
    """Test that error is raised on different shapes of input."""
    metric = metric_class()
    with pytest.raises(
        RuntimeError,
        match="Predictions and targets are expected to have the same shape*",
    ):
        metric(paddle.randn(129, 100, 2), paddle.randn(129, 101, 2))
