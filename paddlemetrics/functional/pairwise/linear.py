from typing import Optional

import paddle
from paddle import Tensor
from typing_extensions import Literal

from paddlemetrics.functional.pairwise.helpers import (_check_input,
                                                      _reduce_distance_matrix)
from paddlemetrics.utils.compute import _safe_matmul


def _pairwise_linear_similarity_update(
    x: paddle.Tensor,
    y: Optional[paddle.Tensor] = None,
    zero_diagonal: Optional[bool] = None,
) -> paddle.Tensor:
    """Calculate the pairwise linear similarity matrix.

    Args:
        x: tensor of shape ``[N,d]``
        y: tensor of shape ``[M,d]``
        zero_diagonal: determines if the diagonal of the distance matrix should be set to zero

    """
    x, y, zero_diagonal = _check_input(x, y, zero_diagonal)
    distance = _safe_matmul(x, y)
    if zero_diagonal:
        distance.fill_diagonal_(value=0)
    return distance


def pairwise_linear_similarity(
    x: paddle.Tensor,
    y: Optional[paddle.Tensor] = None,
    reduction: Literal["mean", "sum", "none", None] = None,
    zero_diagonal: Optional[bool] = None,
) -> paddle.Tensor:
    """Calculate pairwise linear similarity.

    .. math::
        s_{lin}(x,y) = <x,y> = \\sum_{d=1}^D x_d \\cdot y_d

    If both :math:`x` and :math:`y` are passed in, the calculation will be performed pairwise between
    the rows of :math:`x` and :math:`y`.
    If only :math:`x` is passed in, the calculation will be performed between the rows of :math:`x`.

    Args:
        x: Tensor with shape ``[N, d]``
        y: Tensor with shape ``[M, d]``, optional
        reduction: reduction to apply along the last dimension. Choose between `'mean'`, `'sum'`
            (applied along column dimension) or  `'none'`, `None` for no reduction
        zero_diagonal: if the diagonal of the distance matrix should be set to 0. If only `x` is given
            this defaults to `True` else if `y` is also given it defaults to `False`

    Returns:
        A ``[N,N]`` matrix of distances if only ``x`` is given, else a ``[N,M]`` matrix

    Example:
        >>> import paddle
        >>> from paddlemetrics.functional.pairwise import pairwise_linear_similarity
        >>> x = paddle.to_tensor([[2, 3], [3, 5], [5, 8]], dtype=paddle.float32)
        >>> y = paddle.to_tensor([[1, 0], [2, 1]], dtype=paddle.float32)
        >>> pairwise_linear_similarity(x, y)
        tensor([[ 2.,  7.],
                [ 3., 11.],
                [ 5., 18.]])
        >>> pairwise_linear_similarity(x)
        tensor([[ 0., 21., 34.],
                [21.,  0., 55.],
                [34., 55.,  0.]])

    """
    distance = _pairwise_linear_similarity_update(x, y, zero_diagonal)
    return _reduce_distance_matrix(distance, reduction)
