from functools import partial
from typing import Callable

import numpy as np
import paddle
import pytest
from scipy.optimize import linear_sum_assignment
from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester
from unittests.audio import _average_metric_wrapper

from paddlemetrics.audio import PermutationInvariantTraining
from paddlemetrics.functional.audio import (
    permutation_invariant_training, scale_invariant_signal_distortion_ratio,
    signal_noise_ratio)
from paddlemetrics.functional.audio.pit import (
    _find_best_perm_by_exhaustive_method,
    _find_best_perm_by_linear_sum_assignment)

seed_all(42)
TIME_FRAME = 10
inputs1 = _Input(
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE, 3, TIME_FRAME),
    target=paddle.rand(NUM_BATCHES, BATCH_SIZE, 3, TIME_FRAME),
)
inputs2 = _Input(
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE, 2, TIME_FRAME),
    target=paddle.rand(NUM_BATCHES, BATCH_SIZE, 2, TIME_FRAME),
)


def _reference_scipy_pit(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    metric_func: Callable,
    eval_func: str,
    zero_mean: bool = False,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Naive implementation of `Permutation Invariant Training` based on Scipy.

    Args:
        preds: predictions, shape[batch, spk, time]
        target: targets, shape[batch, spk, time]
        metric_func: which metric
        eval_func: min or max
        zero_mean: whether to zero mean the input

    Returns:
        best_metric: shape [batch]
        best_perm: shape [batch, spk]

    """
    if zero_mean:
        target = target - paddle.mean(target, axis=-1, keepdim=True)
        preds = preds - paddle.mean(preds, axis=-1, keepdim=True)
    batch_size, spk_num = target.shape[0:2]
    metric_mtx = paddle.empty((batch_size, spk_num, spk_num), device=target.place)
    for t in range(spk_num):
        for e in range(spk_num):
            metric_mtx[:, t, e] = metric_func(preds[:, e, ...], target[:, t, ...])
    metric_mtx = metric_mtx.detach().cpu().numpy()
    best_metrics = []
    best_perms = []
    for b in range(batch_size):
        row_idx, col_idx = linear_sum_assignment(metric_mtx[b, ...], eval_func == "max")
        best_metrics.append(metric_mtx[b, row_idx, col_idx].mean())
        best_perms.append(col_idx)
    return paddle.from_numpy(np.stack(best_metrics)), paddle.from_numpy(
        np.stack(best_perms)
    )


def _reference_scipy_pit_snr(
    preds: paddle.Tensor, target: paddle.Tensor, zero_mean: bool = False
) -> tuple[paddle.Tensor, paddle.Tensor]:
    return _reference_scipy_pit(
        preds=preds,
        target=target,
        metric_func=signal_noise_ratio,
        eval_func="max",
        zero_mean=zero_mean,
    )


def _reference_scipy_pit_si_sdr(
    preds: paddle.Tensor, target: paddle.Tensor, zero_mean: bool = False
) -> tuple[paddle.Tensor, paddle.Tensor]:
    return _reference_scipy_pit(
        preds=preds,
        target=target,
        metric_func=scale_invariant_signal_distortion_ratio,
        eval_func="max",
        zero_mean=zero_mean,
    )


@pytest.mark.parametrize(
    ("preds", "target", "ref_metric", "metric_func", "mode", "eval_func"),
    [
        (
            inputs1.preds,
            inputs1.target,
            _reference_scipy_pit_snr,
            signal_noise_ratio,
            "speaker-wise",
            "max",
        ),
        (
            inputs1.preds,
            inputs1.target,
            _reference_scipy_pit_si_sdr,
            scale_invariant_signal_distortion_ratio,
            "speaker-wise",
            "max",
        ),
        (
            inputs2.preds,
            inputs2.target,
            _reference_scipy_pit_snr,
            signal_noise_ratio,
            "speaker-wise",
            "max",
        ),
        (
            inputs2.preds,
            inputs2.target,
            _reference_scipy_pit_si_sdr,
            scale_invariant_signal_distortion_ratio,
            "speaker-wise",
            "max",
        ),
        (
            inputs1.preds,
            inputs1.target,
            _reference_scipy_pit_snr,
            signal_noise_ratio,
            "permutation-wise",
            "max",
        ),
        (
            inputs1.preds,
            inputs1.target,
            _reference_scipy_pit_si_sdr,
            scale_invariant_signal_distortion_ratio,
            "permutation-wise",
            "max",
        ),
        (
            inputs2.preds,
            inputs2.target,
            _reference_scipy_pit_snr,
            signal_noise_ratio,
            "permutation-wise",
            "max",
        ),
        (
            inputs2.preds,
            inputs2.target,
            _reference_scipy_pit_si_sdr,
            scale_invariant_signal_distortion_ratio,
            "permutation-wise",
            "max",
        ),
    ],
)
class TestPIT(MetricTester):
    """Test class for `PermutationInvariantTraining` metric."""

    atol = 0.01

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_pit(self, preds, target, ref_metric, metric_func, mode, eval_func, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            PermutationInvariantTraining,
            reference_metric=partial(
                _average_metric_wrapper, metric_func=ref_metric, res_index=0
            ),
            metric_args={
                "metric_func": metric_func,
                "mode": mode,
                "eval_func": eval_func,
            },
        )

    @pytest.mark.parametrize("zero_mean", [True, False])
    def test_pit_functional(
        self, preds, target, ref_metric, metric_func, mode, eval_func, zero_mean
    ):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=permutation_invariant_training,
            reference_metric=partial(ref_metric, zero_mean=zero_mean),
            metric_args={
                "metric_func": metric_func,
                "mode": mode,
                "eval_func": eval_func,
                "zero_mean": zero_mean,
            },
        )

    def test_pit_differentiability(
        self, preds, target, ref_metric, metric_func, mode, eval_func
    ):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""

        def pit_diff(preds, target, metric_func, mode, eval_func):
            return permutation_invariant_training(
                preds, target, metric_func, mode, eval_func
            )[0]

        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=PermutationInvariantTraining,
            metric_functional=pit_diff,
            metric_args={
                "metric_func": metric_func,
                "mode": mode,
                "eval_func": eval_func,
            },
        )

    def test_pit_half_cpu(
        self, preds, target, ref_metric, metric_func, mode, eval_func
    ):
        """Test dtype support of the metric on CPU."""
        pytest.xfail("PIT metric does not support cpu + half precision")

    @pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires cuda")
    def test_pit_half_gpu(
        self, preds, target, ref_metric, metric_func, mode, eval_func
    ):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=PermutationInvariantTraining,
            metric_functional=partial(
                permutation_invariant_training,
                metric_func=metric_func,
                eval_func=eval_func,
            ),
            metric_args={
                "metric_func": metric_func,
                "mode": mode,
                "eval_func": eval_func,
            },
        )


def test_error_on_different_shape() -> None:
    """Test that error is raised on different shapes of input."""
    metric = PermutationInvariantTraining(signal_noise_ratio)
    with pytest.raises(
        RuntimeError,
        match="Predictions and targets are expected to have the same shape at the batch and speaker dimensions",
    ):
        metric(paddle.randn(3, 3, 10), paddle.randn(3, 2, 10))


def test_error_on_wrong_eval_func() -> None:
    """Test that error is raised on wrong `eval_func` argument."""
    metric = PermutationInvariantTraining(signal_noise_ratio, eval_func="xxx")
    with pytest.raises(ValueError, match='eval_func can only be "max" or "min"'):
        metric(paddle.randn(3, 3, 10), paddle.randn(3, 3, 10))


def test_error_on_wrong_mode() -> None:
    """Test that error is raised on wrong `mode` argument."""
    metric = PermutationInvariantTraining(signal_noise_ratio, mode="xxx")
    with pytest.raises(
        ValueError, match='mode can only be "speaker-wise" or "permutation-wise"*'
    ):
        metric(paddle.randn(3, 3, 10), paddle.randn(3, 3, 10))


def test_error_on_wrong_shape() -> None:
    """Test that error is raised on wrong input shape."""
    metric = PermutationInvariantTraining(signal_noise_ratio)
    with pytest.raises(ValueError, match="Inputs must be of shape *"):
        metric(paddle.randn(3), paddle.randn(3))


def test_consistency_of_two_implementations() -> None:
    """Test that both backend functions for computing metric (depending on torch version) returns the same result."""
    shapes_test = [(5, 2, 2), (4, 3, 3), (4, 4, 4), (3, 5, 5)]
    for shp in shapes_test:
        metric_mtx = paddle.randn(size=shp)
        bm1, bp1 = _find_best_perm_by_linear_sum_assignment(
            metric_mtx, paddle.max
        )
        bm2, bp2 = _find_best_perm_by_exhaustive_method(metric_mtx, paddle.max)
        assert paddle.allclose(x=bm1, y=bm2).item()
        assert (bp1 == bp2).all()
