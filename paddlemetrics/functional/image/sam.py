import paddle
from paddle import Tensor
from typing_extensions import Literal

from paddlemetrics.utils.checks import _check_same_shape
from paddlemetrics.utils.distributed import reduce


def _sam_update(
    preds: paddle.Tensor, target: paddle.Tensor
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Update and returns variables required to compute Spectral Angle Mapper.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor

    """
    if preds.dtype != target.dtype:
        raise TypeError(
            f"Expected `preds` and `target` to have the same data type. Got preds: {preds.dtype} and target: {target.dtype}."
        )
    _check_same_shape(preds, target)
    if len(preds.shape) != 4:
        raise ValueError(
            f"Expected `preds` and `target` to have BxCxHxW shape. Got preds: {preds.shape} and target: {target.shape}."
        )
    if preds.shape[1] <= 1 or target.shape[1] <= 1:
        raise ValueError(
            f"Expected channel dimension of `preds` and `target` to be larger than 1. Got preds: {preds.shape[1]} and target: {target.shape[1]}."
        )
    return preds, target


def _sam_compute(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
) -> paddle.Tensor:
    """Compute Spectral Angle Mapper.

    Args:
        preds: estimated image
        target: ground truth image
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

    Example:
        >>> from paddle import rand
        >>> preds = rand([16, 3, 16, 16])
        >>> target = rand([16, 3, 16, 16])
        >>> preds, target = _sam_update(preds, target)
        >>> _sam_compute(preds, target)
        tensor(0.5914)

    """
    dot_product = (preds * target).sum(dim=1)
    preds_norm = preds.norm(dim=1)
    target_norm = target.norm(dim=1)
    sam_score = paddle.clamp(dot_product / (preds_norm * target_norm), -1, 1).acos()
    return reduce(sam_score, reduction)


def spectral_angle_mapper(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
) -> paddle.Tensor:
    """Universal Spectral Angle Mapper.

    Args:
        preds: estimated image
        target: ground truth image
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

    Return:
        Tensor with Spectral Angle Mapper score

    Raises:
        TypeError:
            If ``preds`` and ``target`` don't have the same data type.
        ValueError:
            If ``preds`` and ``target`` don't have ``BxCxHxW shape``.

    Example:
        >>> from paddle import rand
        >>> from paddlemetrics.functional.image import spectral_angle_mapper
        >>> preds = rand([16, 3, 16, 16],)
        >>> target = rand([16, 3, 16, 16])
        >>> spectral_angle_mapper(preds, target)
        tensor(0.5914)

    References:
        [1] Roberta H. Yuhas, Alexander F. H. Goetz and Joe W. Boardman, "Discrimination among semi-arid
        landscape endmembers using the Spectral Angle Mapper (SAM) algorithm" in PL, Summaries of the Third Annual JPL
        Airborne Geoscience Workshop, vol. 1, June 1, 1992.

    """
    preds, target = _sam_update(preds, target)
    return _sam_compute(preds, target, reduction)
