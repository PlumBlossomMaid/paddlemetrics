from functools import partial
from typing import Any

import paddle
import pytest
from unittests._helpers.testers import MetricTester

from paddlemetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment
from paddlemetrics.functional.audio.nisqa import \
    non_intrusive_speech_quality_assessment

inputs = [
    {
        "preds": paddle.rand(2, 2, 16000),
        "fs": 16000,
        "reference": paddle.tensor(
            [
                [
                    [0.8105150461, 1.8459059, 2.478022337, 1.0402423143, 1.5687377453],
                    [
                        0.8629049063,
                        1.7767801285,
                        2.3915612698,
                        1.0460783243,
                        1.6212222576,
                    ],
                ],
                [
                    [
                        0.8608418703,
                        1.9113740921,
                        2.5213730335,
                        1.0900889635,
                        1.6314117908,
                    ],
                    [
                        0.8071692586,
                        1.7834275961,
                        2.4235677719,
                        1.0236976147,
                        1.5617829561,
                    ],
                ],
            ]
        ),
    },
    {
        "preds": paddle.rand(2, 2, 48000),
        "fs": 48000,
        "reference": paddle.tensor(
            [
                [
                    [
                        0.7670641541,
                        1.163433075,
                        2.605681181,
                        1.4002652168,
                        1.5218108892,
                    ],
                    [
                        0.7974857688,
                        1.184592247,
                        2.6476621628,
                        1.4282002449,
                        1.5324314833,
                    ],
                ],
                [
                    [
                        0.81146878,
                        1.1764185429,
                        2.6281285286,
                        1.4396891594,
                        1.5460423231,
                    ],
                    [
                        0.6779640913,
                        1.1818346977,
                        2.510627985,
                        1.2842310667,
                        1.401417613,
                    ],
                ],
            ]
        ),
    },
    {
        "preds": paddle.stack(
            [
                paddle.stack(
                    [
                        paddle.sin(2 * 3.14159 * 440 / 16000 * paddle.arange(16000)),
                        paddle.sin(2 * 3.14159 * 1000 / 16000 * paddle.arange(16000)),
                    ]
                ),
                paddle.stack(
                    [
                        paddle.sign(
                            paddle.sin(2 * 3.14159 * 200 / 16000 * paddle.arange(16000))
                        ),
                        (1 + 2 * 200 / 16000 * paddle.arange(16000)) % 2 - 1,
                    ]
                ),
            ]
        ),
        "fs": 16000,
        "reference": paddle.tensor(
            [
                [
                    [
                        1.1243989468,
                        2.123770237,
                        3.6184809208,
                        1.2584471703,
                        1.8518198729,
                    ],
                    [
                        1.276180625,
                        1.8802671432,
                        3.3731021881,
                        1.2554246187,
                        1.6879540682,
                    ],
                ],
                [
                    [0.925907433, 2.7644648552, 3.1585879326, 1.41639328, 1.5672523975],
                    [
                        0.8493731022,
                        2.6398222446,
                        3.0776870251,
                        1.1348335743,
                        1.6034533978,
                    ],
                ],
            ]
        ),
    },
    {
        "preds": paddle.stack(
            [
                paddle.stack(
                    [
                        paddle.sin(2 * 3.14159 * 440 / 48000 * paddle.arange(48000)),
                        paddle.sin(2 * 3.14159 * 1000 / 48000 * paddle.arange(48000)),
                    ]
                ),
                paddle.stack(
                    [
                        paddle.sign(
                            paddle.sin(2 * 3.14159 * 200 / 48000 * paddle.arange(48000))
                        ),
                        (1 + 2 * 200 / 48000 * paddle.arange(48000)) % 2 - 1,
                    ]
                ),
            ]
        ),
        "fs": 48000,
        "reference": paddle.tensor(
            [
                [
                    [
                        1.1263639927,
                        2.1246092319,
                        3.6191856861,
                        1.2572505474,
                        1.8531025648,
                    ],
                    [
                        1.2741736174,
                        1.8896869421,
                        3.3755991459,
                        1.2591584921,
                        1.6720581055,
                    ],
                ],
                [
                    [
                        0.8731431961,
                        1.6447117329,
                        2.8125579357,
                        1.619717598,
                        1.2627843618,
                    ],
                    [
                        1.2543514967,
                        2.0644433498,
                        3.1744530201,
                        1.8767380714,
                        1.9447042942,
                    ],
                ],
            ]
        ),
    },
]


def _reference_metric_batch(preds, target, mean):
    def _reference_metric(preds):
        for pred, ref in zip(
            *[
                [x for i in inputs for x in i[which].reshape(-1, i[which].shape[-1])]
                for which in ["preds", "reference"]
            ]
        ):
            if paddle.equal(preds, pred):
                return ref
        raise NotImplementedError

    out = paddle.stack(
        [_reference_metric(pred) for pred in preds.reshape(-1, preds.shape[-1])]
    )
    return out.mean(dim=0) if mean else out.reshape(*preds.shape[:-1], 5)


def _nisqa_cheat(preds, target, **kwargs: dict[str, Any]):
    return non_intrusive_speech_quality_assessment(preds, **kwargs)


class _NISQACheat(NonIntrusiveSpeechQualityAssessment):
    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        super().update(preds=preds)


@pytest.mark.parametrize(
    ("preds", "fs", "reference"),
    [(i["preds"], i["fs"], i["reference"]) for i in inputs],
)
class TestNISQA(MetricTester):
    """Test class for `NonIntrusiveSpeechQualityAssessment` metric."""

    atol = 0.0001

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_nisqa(
        self,
        preds: paddle.Tensor,
        reference: paddle.Tensor,
        fs: int,
        ddp: bool,
        device=None,
    ):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds=preds,
            target=preds,
            metric_class=_NISQACheat,
            reference_metric=partial(_reference_metric_batch, mean=True),
            metric_args={"fs": fs},
        )

    def test_nisqa_functional(
        self, preds: paddle.Tensor, reference: paddle.Tensor, fs: int, device="cpu"
    ):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=preds,
            metric_functional=_nisqa_cheat,
            reference_metric=partial(_reference_metric_batch, mean=False),
            metric_args={"fs": fs},
        )


@pytest.mark.parametrize("shape", [(3000,), (2, 3000), (1, 2, 3000), (2, 3, 1, 3000)])
def test_shape(shape: tuple[int]):
    """Test output shape."""
    preds = paddle.rand(*shape)
    out = non_intrusive_speech_quality_assessment(preds, 16000)
    assert out.shape == (*shape[:-1], 5)
    metric = NonIntrusiveSpeechQualityAssessment(16000)
    out = metric(preds)
    assert out.shape == (5,)


def test_batched_vs_unbatched():
    """Test batched versus unbatched processing."""
    preds = paddle.rand(2, 2, 16000)
    out_batched = non_intrusive_speech_quality_assessment(preds, 16000)
    out_unbatched = paddle.stack(
        [
            non_intrusive_speech_quality_assessment(x, 16000)
            for x in preds.reshape(-1, 16000)
        ]
    ).reshape(2, 2, 5)
    assert paddle.allclose(x=out_batched, y=out_unbatched).item()


def test_error_on_short_input():
    """Test error on short input."""
    preds = paddle.rand(3000)
    non_intrusive_speech_quality_assessment(preds, 16000)
    with pytest.raises(RuntimeError, match="Input signal is too short."):
        non_intrusive_speech_quality_assessment(preds, 48000)
    preds = paddle.rand(2000)
    with pytest.raises(RuntimeError, match="Input signal is too short."):
        non_intrusive_speech_quality_assessment(preds, 16000)
    with pytest.raises(RuntimeError, match="Input signal is too short."):
        non_intrusive_speech_quality_assessment(preds, 48000)


def test_error_on_long_input():
    """Test error on long input."""
    preds = paddle.rand(834240)
    with pytest.raises(
        RuntimeError,
        match="Maximum number of mel spectrogram windows exceeded. Use shorter audio.",
    ):
        non_intrusive_speech_quality_assessment(preds, 16000)
    non_intrusive_speech_quality_assessment(preds, 48000)
    preds = paddle.rand(2502720)
    with pytest.raises(
        RuntimeError,
        match="Maximum number of mel spectrogram windows exceeded. Use shorter audio.",
    ):
        non_intrusive_speech_quality_assessment(preds, 16000)
    with pytest.raises(
        RuntimeError,
        match="Maximum number of mel spectrogram windows exceeded. Use shorter audio.",
    ):
        non_intrusive_speech_quality_assessment(preds, 48000)
