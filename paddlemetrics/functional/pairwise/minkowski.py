from typing import Optional

import paddle
from paddle import Tensor
from typing_extensions import Literal

from paddlemetrics.functional.pairwise.helpers import (_check_input,
                                                      _reduce_distance_matrix)
from paddlemetrics.utils.exceptions import PaddleMetricsUserError


def _pairwise_minkowski_distance_update(
    x: paddle.Tensor,
    y: Optional[paddle.Tensor] = None,
    exponent: float = 2,
    zero_diagonal: Optional[bool] = None,
) -> paddle.Tensor:
    """Calculate the pairwise minkowski distance matrix.

    Args:
        x: tensor of shape ``[N,d]``
        y: tensor of shape ``[M,d]``
        exponent: int or float larger than 1, exponent to which the difference between preds and target is to be raised
        zero_diagonal: determines if the diagonal of the distance matrix should be set to zero

    """
    x, y, zero_diagonal = _check_input(x, y, zero_diagonal)
    if not (isinstance(exponent, (float, int)) and exponent >= 1):
        raise PaddleMetricsUserError(
            f"Argument ``p`` must be a float or int greater than 1, but got {exponent}"
        )
    _orig_dtype = x.dtype
    x = x.to(paddle.float64)
    y = y.to(paddle.float64)
    distance = (
        (x.unsqueeze(1) - y.unsqueeze(0))
        .abs()
        .pow(exponent)
        .sum(-1)
        .pow(1.0 / exponent)
    )
    if zero_diagonal:
        distance.fill_diagonal_(value=0)
    return distance.to(_orig_dtype)


def pairwise_minkowski_distance(
    x: paddle.Tensor,
    y: Optional[paddle.Tensor] = None,
    exponent: float = 2,
    reduction: Literal["mean", "sum", "none", None] = None,
    zero_diagonal: Optional[bool] = None,
) -> paddle.Tensor:
    """Calculate pairwise minkowski distances.

    .. math::
        d_{minkowski}(x,y,p) = ||x - y||_p = \\sqrt[p]{\\sum_{d=1}^D (x_d - y_d)^p}

    If both :math:`x` and :math:`y` are passed in, the calculation will be performed pairwise between the rows of
    :math:`x` and :math:`y`. If only :math:`x` is passed in, the calculation will be performed between the rows
    of :math:`x`.

    Args:
        x: Tensor with shape ``[N, d]``
        y: Tensor with shape ``[M, d]``, optional
        exponent: int or float larger than 1, exponent to which the difference between preds and target is to be raised
        reduction: reduction to apply along the last dimension. Choose between `'mean'`, `'sum'`
            (applied along column dimension) or  `'none'`, `None` for no reduction
        zero_diagonal: if the diagonal of the distance matrix should be set to 0. If only `x` is given
            this defaults to `True` else if `y` is also given it defaults to `False`

    Returns:
        A ``[N,N]`` matrix of distances if only ``x`` is given, else a ``[N,M]`` matrix

    Example:
        >>> import paddle
        >>> from paddlemetrics.functional.pairwise import pairwise_minkowski_distance
        >>> x = paddle.to_tensor([[2, 3], [3, 5], [5, 8]], dtype=paddle.float32)
        >>> y = paddle.to_tensor([[1, 0], [2, 1]], dtype=paddle.float32)
        >>> pairwise_minkowski_distance(x, y, exponent=4)
        tensor([[3.0092, 2.0000],
                [5.0317, 4.0039],
                [8.1222, 7.0583]])
        >>> pairwise_minkowski_distance(x, exponent=4)
        tensor([[0.0000, 2.0305, 5.1547],
                [2.0305, 0.0000, 3.1383],
                [5.1547, 3.1383, 0.0000]])

    """
    distance = _pairwise_minkowski_distance_update(x, y, exponent, zero_diagonal)
    return _reduce_distance_matrix(distance, reduction)
