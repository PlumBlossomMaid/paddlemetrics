from functools import partial

import numpy as np
import paddle
import pytest
from sewar.full_ref import scc as sewar_scc
from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

from paddlemetrics.functional.image import spatial_correlation_coefficient
from paddlemetrics.image import SpatialCorrelationCoefficient

seed_all(42)
_inputs = [
    _Input(
        preds=paddle.randn(NUM_BATCHES, BATCH_SIZE, channels, 32, 32),
        target=paddle.randn(NUM_BATCHES, BATCH_SIZE, channels, 32, 32),
    )
    for channels in [1, 3]
]
_kernels = [paddle.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])]


def _reference_sewar_scc(preds, target, hp_filter, window_size, reduction):
    """Wrapper around reference implementation of scc from sewar."""
    preds = paddle.moveaxis(x=preds, source=1, destination=-1)
    target = paddle.moveaxis(x=target, source=1, destination=-1)
    preds = preds.cpu().numpy()
    target = target.cpu().numpy()
    scc = [
        sewar_scc(GT=target[batch], P=preds[batch], win=hp_filter, ws=window_size)
        for batch in range(preds.shape[0])
    ]
    if reduction == "mean":
        return np.mean(scc)
    if reduction == "none":
        return scc
    return None


def _reference_sewar_scc_simple(preds, target):
    """Reference implementation of SCC from sewar."""
    hp_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    return _reference_sewar_scc(
        preds, target, hp_filter, window_size=8, reduction="mean"
    )


@pytest.mark.parametrize(("preds", "target"), [(i.preds, i.target) for i in _inputs])
class TestSpatialCorrelationCoefficient(MetricTester):
    """Tests for SpatialCorrelationCoefficient metric."""

    atol = 1e-08

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_scc(self, preds, target, ddp):
        """Test SpatialCorrelationCoefficient class usage."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            metric_class=SpatialCorrelationCoefficient,
            reference_metric=_reference_sewar_scc_simple,
        )

    @pytest.mark.parametrize("hp_filter", _kernels)
    @pytest.mark.parametrize("window_size", [8, 11])
    @pytest.mark.parametrize("reduction", ["mean", "none"])
    def test_scc_functional(self, preds, target, hp_filter, window_size, reduction):
        """Test SpatialCorrelationCoefficient functional usage."""
        kwargs = {
            "hp_filter": hp_filter,
            "window_size": window_size,
            "reduction": reduction,
        }
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=spatial_correlation_coefficient,
            reference_metric=partial(_reference_sewar_scc, **kwargs),
            metric_args=kwargs,
        )
