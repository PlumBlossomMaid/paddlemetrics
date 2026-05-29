from typing import List

import paddle


def lip_vertex_error(
    vertices_pred: paddle.Tensor,
    vertices_gt: paddle.Tensor,
    mouth_map: List[int],
    validate_args: bool = True,
) -> paddle.Tensor:
    """Compute Lip Vertex Error (LVE) for 3D talking head evaluation.

    The Lip Vertex Error (LVE) metric evaluates the quality of lip synchronization in 3D facial animations by measuring
    the maximum Euclidean distance (L2 error) between corresponding lip vertices of the generated and ground truth
    meshes for each frame. The metric is defined as:

    .. math::
        \\text{LVE} = \\frac{1}{N} \\sum_{i=1}^{N} \\max_{v \\in \\text{lip}} \\|x_{i,v} - \\hat{x}_{i,v}\\|_2^2

    where :math:`N` is the number of frames, :math:`x_{i,v}` represents the 3D coordinates of vertex :math:`v` in the
    lip region of the ground truth frame :math:`i`, and :math:`\\hat{x}_{i,v}` represents the corresponding vertex in
    the predicted frame. The metric computes the maximum squared L2 distance between corresponding lip vertices for each
    frame and averages across all frames. A lower LVE value indicates better lip synchronization quality.

    Args:
        vertices_pred: Predicted vertices tensor of shape (T, V, 3) where T is number of frames,
            V is number of vertices, and 3 represents XYZ coordinates
        vertices_gt: Ground truth vertices tensor of shape (T', V, 3) where T' can be different from T
        mouth_map: List of vertex indices corresponding to the mouth region
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        paddle.Tensor: Scalar tensor containing the mean LVE value across all frames

    Raises:
        ValueError:
            If the number of dimensions of `vertices_pred` or `vertices_gt` is not 3.
            If vertex dimensions (V) or coordinate dimensions (3) don't match
            If ``mouth_map`` is empty or contains invalid indices

    Example:
        >>> import paddle
        >>> from paddlemetrics.functional.multimodal import lip_vertex_error
        >>> vertices_pred = paddle.randn(10, 100, 3, generator=paddle.seed(42))
        >>> vertices_gt = paddle.randn(10, 100, 3, generator=paddle.seed(43))
        >>> mouth_map = [0, 1, 2, 3, 4]
        >>> lip_vertex_error(vertices_pred, vertices_gt, mouth_map)
        tensor(12.7688)

    """
    if validate_args:
        if vertices_pred.ndim != 3 or vertices_gt.ndim != 3:
            raise ValueError(
                f"Expected both vertices_pred and vertices_gt to have 3 dimensions but got {vertices_pred.ndim} and {vertices_gt.ndim} dimensions respectively."
            )
        if vertices_pred.shape[1:] != vertices_gt.shape[1:]:
            raise ValueError(
                f"Expected vertices_pred and vertices_gt to have same vertex and coordinate dimensions but got shapes {vertices_pred.shape} and {vertices_gt.shape}."
            )
        if not mouth_map:
            raise ValueError("mouth_map cannot be empty.")
        if max(mouth_map) >= vertices_pred.shape[1]:
            raise ValueError(
                f"mouth_map contains invalid vertex indices. Max index {max(mouth_map)} is larger than number of vertices {vertices_pred.shape[1]}."
            )
    min_frames = min(vertices_pred.shape[0], vertices_gt.shape[0])
    vertices_pred = vertices_pred[:min_frames]
    vertices_gt = vertices_gt[:min_frames]
    diff = vertices_gt[:, mouth_map, :] - vertices_pred[:, mouth_map, :]
    sq_dist = paddle.sum(diff**2, axis=-1)
    max_per_frame = paddle.max(sq_dist, axis=1).values
    return paddle.mean(max_per_frame)
