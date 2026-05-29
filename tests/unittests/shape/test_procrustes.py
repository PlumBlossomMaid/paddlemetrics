from functools import partial

import numpy as np
import paddle
import pytest
from scipy.spatial import procrustes as scipy_procrustes
from unittests import BATCH_SIZE, EXTRA_DIM, NUM_BATCHES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

from paddlemetrics.functional.shape.procrustes import procrustes_disparity
from paddlemetrics.shape.procrustes import ProcrustesDisparity

seed_all(42)
NUM_TARGETS = 5
_inputs = _Input(
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE, 50, EXTRA_DIM),
    target=paddle.rand(NUM_BATCHES, BATCH_SIZE, 50, EXTRA_DIM),
)


def _reference_procrustes(point_cloud1, point_cloud2, reduction=None):
    point_cloud1 = point_cloud1.numpy()
    point_cloud2 = point_cloud2.numpy()
    if reduction is None:
        return np.array(
            [scipy_procrustes(d1, d2)[2] for d1, d2 in zip(point_cloud1, point_cloud2)]
        )
    disparity = 0
    for d1, d2 in zip(point_cloud1, point_cloud2):
        disparity += scipy_procrustes(d1, d2)[2]
    if reduction == "mean":
        return disparity / len(point_cloud1)
    return disparity


@pytest.mark.parametrize(
    ("point_cloud1", "point_cloud2"), [(_inputs.preds, _inputs.target)]
)
class TestProcrustesDisparity(MetricTester):
    """Test class for `ProcrustesDisparity` metric."""

    @pytest.mark.parametrize("reduction", ["sum", "mean"])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_procrustes_disparity(self, reduction, point_cloud1, point_cloud2, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            point_cloud1,
            point_cloud2,
            ProcrustesDisparity,
            partial(_reference_procrustes, reduction=reduction),
            metric_args={"reduction": reduction},
        )

    def test_procrustes_disparity_functional(self, point_cloud1, point_cloud2):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            point_cloud1, point_cloud2, procrustes_disparity, _reference_procrustes
        )


def test_error_on_different_shape():
    """Test that error is raised on different shapes of input."""
    metric = ProcrustesDisparity()
    with pytest.raises(
        RuntimeError,
        match="Predictions and targets are expected to have the same shape",
    ):
        metric(paddle.randn(10, 100, 2), paddle.randn(10, 50, 2))
    with pytest.raises(
        RuntimeError,
        match="Predictions and targets are expected to have the same shape",
    ):
        procrustes_disparity(paddle.randn(10, 100, 2), paddle.randn(10, 50, 2))


def test_error_on_non_3d_input():
    """Test that error is raised if input is not 3-dimensional."""
    metric = ProcrustesDisparity()
    with pytest.raises(
        ValueError, match="Expected both datasets to be 3D tensors of shape"
    ):
        metric(paddle.randn(100), paddle.randn(100))
    with pytest.raises(
        ValueError, match="Expected both datasets to be 3D tensors of shape"
    ):
        procrustes_disparity(paddle.randn(100), paddle.randn(100))
