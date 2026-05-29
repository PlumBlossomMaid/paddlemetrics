from typing import Union

import paddle

from paddlemetrics.utils.checks import _check_same_shape
from paddlemetrics.utils.prints import rank_zero_warn


def procrustes_disparity(
    point_cloud1: paddle.Tensor, point_cloud2: paddle.Tensor, return_all: bool = False
) -> Union[paddle.Tensor, tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]]:
    """Runs procrustrus analysis on a batch of data points.

    Works similar ``scipy.spatial.procrustes`` but for batches of data points.

    Args:
        point_cloud1: The first set of data points
        point_cloud2: The second set of data points
        return_all: If True, returns the scale and rotation matrices along with the disparity

    """
    _check_same_shape(point_cloud1, point_cloud2)
    if point_cloud1.ndim != 3:
        raise ValueError(
            f"Expected both datasets to be 3D tensors of shape (N, M, D), where N is the batch size, M is the number of data points and D is the dimensionality of the data points, but got {point_cloud1.ndim} dimensions."
        )
    point_cloud1 = point_cloud1 - point_cloud1.mean(dim=1, keepdim=True)
    point_cloud2 = point_cloud2 - point_cloud2.mean(dim=1, keepdim=True)
    point_cloud1 /= paddle.linalg.norm(point_cloud1, axis=[1, 2], keepdim=True)
    point_cloud2 /= paddle.linalg.norm(point_cloud2, axis=[1, 2], keepdim=True)
    try:
        u, w, v = paddle.linalg.svd(
            x=paddle.matmul(point_cloud2.transpose(1, 2), point_cloud1).transpose(1, 2),
            full_matrices=False,
        )
    except Exception as ex:
        rank_zero_warn(
            f"SVD calculation in procrustes_disparity failed with exception {ex}. Returning 0 disparity and identity scale/rotation.",
            UserWarning,
        )
        return (
            paddle.tensor(0.0),
            paddle.ones(point_cloud1.shape[0]),
            paddle.eye(point_cloud1.shape[2]),
        )
    rotation = paddle.matmul(u, v)
    scale = w.sum(1, keepdim=True)
    point_cloud2 = scale[:, None] * paddle.matmul(
        point_cloud2, rotation.transpose(1, 2)
    )
    disparity = (point_cloud1 - point_cloud2).square().sum(dim=[1, 2])
    if return_all:
        return disparity, scale, rotation
    return disparity
