from functools import partial

import paddle
import pytest
from pesq import pesq as pesq_backend
from scipy.io import wavfile
from unittests import _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester
from unittests.audio import (_SAMPLE_AUDIO_SPEECH, _SAMPLE_AUDIO_SPEECH_BAB_DB,
                             _average_metric_wrapper)

from paddlemetrics.audio import PerceptualEvaluationSpeechQuality
from paddlemetrics.functional.audio import perceptual_evaluation_speech_quality

seed_all(42)
inputs_8k = _Input(preds=paddle.rand(2, 3, 2100), target=paddle.rand(2, 3, 2100))
inputs_16k = _Input(preds=paddle.rand(2, 3, 4100), target=paddle.rand(2, 3, 4100))


def _reference_pesq_batch(
    preds: paddle.Tensor, target: paddle.Tensor, fs: int, mode: str
):
    """Comparison function."""
    target = target.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    mss = []
    for b in range(preds.shape[0]):
        pesq_val = pesq_backend(fs, target[b, ...], preds[b, ...], mode)
        mss.append(pesq_val)
    return paddle.tensor(mss)


@pytest.mark.parametrize(
    ("preds", "target", "ref_metric", "fs", "mode"),
    [
        (
            inputs_8k.preds,
            inputs_8k.target,
            partial(_reference_pesq_batch, fs=8000, mode="nb"),
            8000,
            "nb",
        ),
        (
            inputs_16k.preds,
            inputs_16k.target,
            partial(_reference_pesq_batch, fs=16000, mode="nb"),
            16000,
            "nb",
        ),
        (
            inputs_16k.preds,
            inputs_16k.target,
            partial(_reference_pesq_batch, fs=16000, mode="wb"),
            16000,
            "wb",
        ),
    ],
)
class TestPESQ(MetricTester):
    """Test class for `PerceptualEvaluationSpeechQuality` metric."""

    atol = 0.01

    @pytest.mark.parametrize("num_processes", [1, 2])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_pesq(self, preds, target, ref_metric, fs, mode, num_processes, ddp):
        """Test class implementation of metric."""
        if num_processes != 1 and ddp:
            pytest.skip("Multiprocessing and ddp does not work together")
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            PerceptualEvaluationSpeechQuality,
            reference_metric=partial(_average_metric_wrapper, metric_func=ref_metric),
            metric_args={"fs": fs, "mode": mode, "n_processes": num_processes},
        )

    @pytest.mark.parametrize("num_processes", [1, 2])
    def test_pesq_functional(self, preds, target, ref_metric, fs, mode, num_processes):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            perceptual_evaluation_speech_quality,
            ref_metric,
            metric_args={"fs": fs, "mode": mode, "n_processes": num_processes},
        )

    def test_pesq_differentiability(self, preds, target, ref_metric, fs, mode):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=PerceptualEvaluationSpeechQuality,
            metric_functional=perceptual_evaluation_speech_quality,
            metric_args={"fs": fs, "mode": mode},
        )

    def test_pesq_half_cpu(self, preds, target, ref_metric, fs, mode):
        """Test dtype support of the metric on CPU."""
        pytest.xfail("PESQ metric does not support cpu + half precision")

    @pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires cuda")
    def test_pesq_half_gpu(self, preds, target, ref_metric, fs, mode):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=PerceptualEvaluationSpeechQuality,
            metric_functional=partial(
                perceptual_evaluation_speech_quality, fs=fs, mode=mode
            ),
            metric_args={"fs": fs, "mode": mode},
        )


def test_error_on_different_shape(metric_class=PerceptualEvaluationSpeechQuality):
    """Test that an error is raised on different shapes of input."""
    metric = metric_class(16000, "nb")
    with pytest.raises(
        RuntimeError,
        match="Predictions and targets are expected to have the same shape",
    ):
        metric(paddle.randn(100), paddle.randn(50))


def test_on_real_audio():
    """Test that metric works as expected on real audio signals."""
    rate, ref = wavfile.read(_SAMPLE_AUDIO_SPEECH)
    rate, deg = wavfile.read(_SAMPLE_AUDIO_SPEECH_BAB_DB)
    pesq_score = perceptual_evaluation_speech_quality(
        paddle.from_numpy(deg), paddle.from_numpy(ref), rate, "wb"
    )
    assert paddle.allclose(
        x=pesq_score, y=paddle.tensor(1.0832337141036987), atol=0.0001
    ).item()
    pesq_score = perceptual_evaluation_speech_quality(
        paddle.from_numpy(deg), paddle.from_numpy(ref), rate, "nb"
    )
    assert paddle.allclose(
        x=pesq_score, y=paddle.tensor(1.6072081327438354), atol=0.0001
    ).item()
