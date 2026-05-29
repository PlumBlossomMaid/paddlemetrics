from typing import Optional

import paddle


def _check_input(
    x: paddle.Tensor,
    y: Optional[paddle.Tensor] = None,
    zero_diagonal: Optional[bool] = None,
) -> tuple[paddle.Tensor, paddle.Tensor, bool]:
    """Check that input has the right dimensionality and sets the ``zero_diagonal`` argument if user has not set it.

    Args:
        x: tensor of shape ``[N,d]``
        y: if provided, a tensor of shape ``[M,d]``
        zero_diagonal: determines if the diagonal of the distance matrix should be set to zero

    """
    if x.ndim != 2:
        raise ValueError(
            f"Expected argument `x` to be a 2D tensor of shape `[N, d]` but got {x.shape}"
        )
    if y is not None:
        if y.ndim != 2 or y.shape[1] != x.shape[1]:
            raise ValueError(
                "Expected argument `y` to be a 2D tensor of shape `[M, d]` where `d` should be same as the last dimension of `x`"
            )
        zero_diagonal = False if zero_diagonal is None else zero_diagonal
    else:
        y = x.clone()
        zero_diagonal = True if zero_diagonal is None else zero_diagonal
    return x, y, zero_diagonal


def _reduce_distance_matrix(
    distmat: paddle.Tensor, reduction: Optional[str] = None
) -> paddle.Tensor:
    """Reduction of distance matrix.

    Args:
        distmat: a ``[N,M]`` matrix
        reduction: string determining how to reduce along last dimension

    """
    if reduction == "mean":
        return distmat.mean(dim=-1)
    if reduction == "sum":
        return distmat.sum(dim=-1)
    if reduction is None or reduction == "none":
        return distmat
    raise ValueError(
        f"Expected reduction to be one of `['mean', 'sum', None]` but got {reduction}"
    )
