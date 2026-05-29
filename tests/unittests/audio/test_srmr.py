from functools import partial
from typing import Any

import paddle
import pytest
from srmrpy import srmr as srmrpy_srmr
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

from paddlemetrics.audio.srmr import SpeechReverberationModulationEnergyRatio
from paddlemetrics.functional.audio.srmr import \
    speech_reverberation_modulation_energy_ratio

seed_all(42)
preds = paddle.rand(2, 2, 8000)


def _reference_srmr_batch(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    fs: int,
    fast: bool,
    norm: bool,
    reduce_mean: bool = False,
    **kwargs: dict[str, Any]
):
    shape = preds.shape
    preds = preds.reshape(1, -1) if len(shape) == 1 else preds.reshape(-1, shape[-1])
    n_batch, time = preds.shape
    preds = preds.detach().cpu().numpy()
    score = []
    for b in range(preds.shape[0]):
        val, _ = srmrpy_srmr(
            preds[b, ...], fs=fs, fast=fast, norm=norm, max_cf=128 if not norm else 30
        )
        score.append(val)
    score = paddle.tensor(score)
    srmr = score.reshape(*shape[:-1])
    if reduce_mean:
        return srmr.mean()
    return srmr


def _speech_reverberation_modulation_energy_ratio_cheat(
    preds, target, **kwargs: dict[str, Any]
):
    return speech_reverberation_modulation_energy_ratio(preds, **kwargs)


class _SpeechReverberationModulationEnergyRatioCheat(
    SpeechReverberationModulationEnergyRatio
):
    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        super().update(preds=preds)


@pytest.mark.parametrize(
    ("preds", "fs", "fast", "norm"),
    [
        (preds, 8000, False, False),
        (preds, 8000, False),
        (preds, 8000, False),
        (preds, 8000),
        (preds, 16000, False, False),
        (preds, 16000, False),
        (preds, 16000, False),
        (preds, 16000),
    ],
)
class TestSRMR(MetricTester):
    """Test class for `SpeechReverberationModulationEnergyRatio` metric."""

    atol = 0.05

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_srmr(self, preds, fs, fast, norm, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds=preds,
            target=preds,
            metric_class=_SpeechReverberationModulationEnergyRatioCheat,
            reference_metric=partial(
                _reference_srmr_batch, fs=fs, fast=fast, norm=norm, reduce_mean=True
            ),
            metric_args={"fs": fs, "fast": fast, "norm": norm},
        )

    def test_srmr_functional(self, preds, fs, fast, norm):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=preds,
            metric_functional=_speech_reverberation_modulation_energy_ratio_cheat,
            reference_metric=partial(
                _reference_srmr_batch, fs=fs, fast=fast, norm=norm
            ),
            metric_args={"fs": fs, "fast": fast, "norm": norm},
        )

    def test_srmr_differentiability(self, preds, fs, fast, norm):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        if fast is True:
            pytest.xfail("SRMR metric is not differentiable when `fast=True`")
        else:
            pytest.xfail(
                "differentiable test for SRMR metric is skipped as it is too slow"
            )

    def test_srmr_half_cpu(self, preds, fs, fast, norm):
        """Test dtype support of the metric on CPU."""
        pytest.xfail("SRMR metric does not support cpu + half precision")

    @pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires cuda")
    def test_srmr_half_gpu(self, preds, fs, fast, norm):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds=preds,
            target=preds,
            metric_module=_SpeechReverberationModulationEnergyRatioCheat,
            metric_functional=_speech_reverberation_modulation_energy_ratio_cheat,
            metric_args={"fs": fs, "fast": fast, "norm": norm},
        )
