from functools import partial

import numpy as np
import paddle
import pytest
from skimage.metrics import \
    peak_signal_noise_ratio as skimage_peak_signal_noise_ratio
from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

from paddlemetrics.functional import peak_signal_noise_ratio
from paddlemetrics.image import PeakSignalNoiseRatio

seed_all(42)
_input_size = NUM_BATCHES, BATCH_SIZE, 32, 32
_inputs = [
    _Input(
        preds=paddle.randint(
            low=0, high=n_cls_pred, shape=_input_size, dtype=paddle.float32
        ),
        target=paddle.randint(
            low=0, high=n_cls_target, shape=_input_size, dtype=paddle.float32
        ),
    )
    for n_cls_pred, n_cls_target in [(10, 10), (5, 10), (10, 5)]
]


def _to_sk_peak_signal_noise_ratio_inputs(value, dim):
    value = value.numpy()
    batches = value[None] if value.ndim == len(_input_size) - 1 else value
    if dim is None:
        return [batches]
    num_dims = np.size(dim)
    if not num_dims:
        return batches
    inputs = []
    for batch in batches:
        batch = np.moveaxis(batch, dim, np.arange(-num_dims, 0))
        psnr_input_shape = batch.shape[-num_dims:]
        inputs.extend(batch.reshape(-1, *psnr_input_shape))
    return inputs


def _reference_skimage_psnr(preds, target, data_range, reduction, dim):
    if isinstance(data_range, tuple):
        preds = preds.clamp(min=data_range[0], max=data_range[1])
        target = target.clamp(min=data_range[0], max=data_range[1])
        data_range = data_range[1] - data_range[0]
    sk_preds_lists = _to_sk_peak_signal_noise_ratio_inputs(preds, axis=dim)
    sk_target_lists = _to_sk_peak_signal_noise_ratio_inputs(target, axis=dim)
    np_reduce_map = {"elementwise_mean": np.mean, "none": np.array, "sum": np.sum}
    return np_reduce_map[reduction](
        [
            skimage_peak_signal_noise_ratio(sk_target, sk_preds, data_range=data_range)
            for sk_target, sk_preds in zip(sk_target_lists, sk_preds_lists)
        ]
    )


def _reference_sklearn_psnr_log(preds, target, data_range, reduction, dim):
    return _reference_skimage_psnr(preds, target, data_range, reduction, dim) * np.log(
        10
    )


@pytest.mark.parametrize(
    ("preds", "target", "data_range", "reduction", "dim"),
    [
        (_inputs[0].preds, _inputs[0].target, 10, "elementwise_mean", None),
        (_inputs[1].preds, _inputs[1].target, 10, "elementwise_mean", None),
        (_inputs[2].preds, _inputs[2].target, 5, "elementwise_mean", None),
        (_inputs[2].preds, _inputs[2].target, 5, "elementwise_mean", 1),
        (_inputs[2].preds, _inputs[2].target, 5, "elementwise_mean", (1, 2)),
        (_inputs[2].preds, _inputs[2].target, 5, "sum", (1, 2)),
        (_inputs[0].preds, _inputs[0].target, (0.0, 1.0), "elementwise_mean", None),
    ],
)
@pytest.mark.parametrize(
    ("base", "ref_metric"),
    [(10.0, _reference_skimage_psnr), (2.718281828459045, _reference_sklearn_psnr_log)],
)
class TestPSNR(MetricTester):
    """Test class for `PeakSignalNoiseRatio` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_psnr(
        self, preds, target, data_range, base, reduction, dim, ref_metric, ddp
    ):
        """Test class implementation of metric."""
        _args = {
            "data_range": data_range,
            "base": base,
            "reduction": reduction,
            "dim": dim,
        }
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            metric_class=PeakSignalNoiseRatio,
            reference_metric=partial(
                ref_metric, data_range=data_range, reduction=reduction, axis=dim
            ),
            metric_args=_args,
        )

    def test_psnr_functional(
        self, preds, target, ref_metric, data_range, base, reduction, dim
    ):
        """Test functional implementation of metric."""
        _args = {
            "data_range": data_range,
            "base": base,
            "reduction": reduction,
            "dim": dim,
        }
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=peak_signal_noise_ratio,
            reference_metric=partial(
                ref_metric, data_range=data_range, reduction=reduction, axis=dim
            ),
            metric_args=_args,
        )

    @pytest.mark.skipif(
        not True,
        reason="Pytoch below 2.1 does not support cpu + half precision used in PSNR metric",
    )
    def test_psnr_half_cpu(
        self, preds, target, data_range, reduction, dim, base, ref_metric
    ):
        """Test dtype support of the metric on CPU."""
        self.run_precision_test_cpu(
            preds,
            target,
            PeakSignalNoiseRatio,
            peak_signal_noise_ratio,
            {
                "data_range": data_range,
                "base": base,
                "reduction": reduction,
                "dim": dim,
            },
        )

    @pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires cuda")
    def test_psnr_half_gpu(
        self, preds, target, data_range, reduction, dim, base, ref_metric
    ):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds,
            target,
            PeakSignalNoiseRatio,
            peak_signal_noise_ratio,
            {
                "data_range": data_range,
                "base": base,
                "reduction": reduction,
                "dim": dim,
            },
        )


@pytest.mark.parametrize("reduction", ["none", "sum"])
def test_reduction_for_dim_none(reduction):
    """Test that warnings are raised when then reduction parameter is combined with no dim provided arg."""
    match = f"The `reduction={reduction}` will not have any effect when `dim` is None."
    with pytest.warns(UserWarning, match=match):
        PeakSignalNoiseRatio(data_range=10.0, reduction=reduction, axis=None)
    with pytest.warns(UserWarning, match=match):
        peak_signal_noise_ratio(
            _inputs[0].preds,
            _inputs[0].target,
            data_range=10.0,
            reduction=reduction, axis=None,
        )


def test_psnr_uint_dtype():
    """Check that automatic casting to float is done for uint dtype.

    See issue: https://github.com/Lightning-AI/paddlemetrics/issues/2787

    """
    preds = paddle.randint(low=0, high=255, shape=_input_size, dtype=paddle.uint8)
    target = paddle.randint(low=0, high=255, shape=_input_size, dtype=paddle.uint8)
    psnr = peak_signal_noise_ratio(preds, target, data_range=255.0)
    prnr2 = peak_signal_noise_ratio(preds.float(), target.float(), data_range=255.0)
    assert paddle.allclose(x=psnr, y=prnr2).item()
