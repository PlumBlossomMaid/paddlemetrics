from functools import partial
from typing import NamedTuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import paddle
from paddle import Tensor
import pytest
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

from paddlemetrics.functional.multimodal.lve import lip_vertex_error
from paddlemetrics.multimodal.lve import LipVertexError

seed_all(42)


class _InputVertices(NamedTuple):
    vertices_pred: Tensor
    vertices_gt: Tensor


def _generate_vertices(batch_size: int = 1) -> _InputVertices:
    """Generate random vertices for testing."""
    return _InputVertices(
        vertices_pred=paddle.randn(batch_size, 10, 100, 3),
        vertices_gt=paddle.randn(batch_size, 10, 100, 3),
    )


def _reference_lip_vertex_error(vertices_pred, vertices_gt, mouth_map):
    """Reference implementation for Lip Vertex Error.

    This uses a numpy implementation for validation.

    """
    min_frames = min(vertices_pred.shape[0], vertices_gt.shape[0])
    vertices_pred = vertices_pred[:min_frames].numpy()
    vertices_gt = vertices_gt[:min_frames].numpy()
    l2_dis_mouth_max = np.array(
        [np.square(vertices_gt[:, v, :] - vertices_pred[:, v, :]) for v in mouth_map]
    )
    l2_dis_mouth_max = np.transpose(l2_dis_mouth_max, (1, 0, 2))
    l2_dis_mouth_max = np.sum(l2_dis_mouth_max, axis=2)
    l2_dis_mouth_max = np.max(l2_dis_mouth_max, axis=1)
    lve = np.mean(l2_dis_mouth_max)
    return paddle.tensor(lve)


class TestLipVertexError(MetricTester):
    """Test class for `LipVertexError` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_lip_vertex_error(self, ddp):
        """Test class implementation of metric."""
        mouth_map = [0, 1, 2, 3, 4]
        vertices_pred, vertices_gt = _generate_vertices(batch_size=4)
        self.run_class_metric_test(
            ddp=ddp,
            preds=vertices_pred,
            target=vertices_gt,
            metric_class=LipVertexError,
            reference_metric=partial(_reference_lip_vertex_error, mouth_map=mouth_map),
            metric_args={"mouth_map": mouth_map},
        )

    def test_lip_vertex_error_functional(self):
        """Test functional implementation of metric."""
        mouth_map = [0, 1, 2, 3, 4]
        vertices_pred, vertices_gt = _generate_vertices(batch_size=4)
        self.run_functional_metric_test(
            preds=vertices_pred,
            target=vertices_gt,
            metric_functional=lip_vertex_error,
            reference_metric=partial(_reference_lip_vertex_error, mouth_map=mouth_map),
            metric_args={"mouth_map": mouth_map},
        )

    def test_lip_vertex_error_differentiability(self):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        mouth_map = [0, 1, 2, 3, 4]
        vertices_pred, vertices_gt = _generate_vertices(batch_size=4)
        self.run_differentiability_test(
            preds=vertices_pred,
            target=vertices_gt,
            metric_module=LipVertexError,
            metric_functional=lip_vertex_error,
            metric_args={"mouth_map": mouth_map},
        )

    def test_error_on_wrong_dimensions(self):
        """Test that an error is raised if vertices tensors have wrong dimensions."""
        metric = LipVertexError(mouth_map=[0, 1, 2, 3, 4])
        with pytest.raises(
            ValueError,
            match="Expected both vertices_pred and vertices_gt to have 3 dimensions.*",
        ):
            metric(paddle.randn(10, 100), paddle.randn(10, 100, 3))

    def test_error_on_mismatched_dimensions(self):
        """Test that an error is raised if vertex dimensions don't match."""
        metric = LipVertexError(mouth_map=[0, 1, 2, 3, 4])
        with pytest.raises(
            ValueError,
            match="Expected vertices_pred and vertices_gt to have same vertex.*",
        ):
            metric(paddle.randn(10, 80, 3), paddle.randn(10, 100, 3))

    def test_error_on_empty_mouth_map(self):
        """Test that an error is raised if mouth_map is empty."""
        with pytest.raises(ValueError, match="mouth_map cannot be empty."):
            LipVertexError(mouth_map=[])

    def test_error_on_invalid_mouth_indices(self):
        """Test that an error is raised if mouth_map contains invalid indices."""
        metric = LipVertexError(mouth_map=[98, 99, 100])
        with pytest.raises(
            ValueError, match="mouth_map contains invalid vertex indices.*"
        ):
            metric(paddle.randn(10, 50, 3), paddle.randn(10, 50, 3))

    def test_different_sequence_lengths(self):
        """Test that the metric handles different sequence lengths correctly."""
        metric = LipVertexError(mouth_map=[0, 1, 2])
        pred = paddle.randn(10, 50, 3)
        target = paddle.randn(8, 50, 3)
        metric(pred, target)

    def test_plot_method(self):
        """Test the plot method of LipVertexError."""
        metric = LipVertexError(mouth_map=[0, 1, 2, 3, 4])
        vertices_pred, vertices_gt = _generate_vertices()
        metric.update(vertices_pred[0], vertices_gt[0])
        fig, ax = metric.plot()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
