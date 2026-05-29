from typing import Optional, Union

import paddle
from paddle import Tensor
from typing_extensions import Literal


def _total_variation_update(img: paddle.Tensor) -> tuple[paddle.Tensor, int]:
    """Compute total variation statistics on current batch."""
    if img.ndim != 4:
        raise RuntimeError(
            f"Expected input `img` to be an 4D tensor, but got {img.shape}"
        )
    diff1 = img[..., 1:, :] - img[..., :-1, :]
    diff2 = img[..., :, 1:] - img[..., :, :-1]
    res1 = diff1.abs().sum([1, 2, 3])
    res2 = diff2.abs().sum([1, 2, 3])
    score = res1 + res2
    return score, img.shape[0]


def _total_variation_compute(
    score: paddle.Tensor,
    num_elements: Union[int, paddle.Tensor],
    reduction: Optional[Literal["mean", "sum", "none"]],
) -> paddle.Tensor:
    """Compute final total variation score."""
    if reduction == "mean":
        return score.sum() / num_elements
    if reduction == "sum":
        return score.sum()
    if reduction is None or reduction == "none":
        return score
    raise ValueError(
        "Expected argument `reduction` to either be 'sum', 'mean', 'none' or None"
    )


def total_variation(
    img: paddle.Tensor, reduction: Optional[Literal["mean", "sum", "none"]] = "sum"
) -> paddle.Tensor:
    """Compute total variation loss.

    Args:
        img: A `Tensor` of shape `(N, C, H, W)` consisting of images
        reduction: a method to reduce metric score over samples.

            - ``'mean'``: takes the mean over samples
            - ``'sum'``: takes the sum over samples
            - ``None`` or ``'none'``: return the score per sample

    Returns:
        A loss scalar value containing the total variation

    Raises:
        ValueError:
            If ``reduction`` is not one of ``'sum'``, ``'mean'``, ``'none'`` or ``None``
        RuntimeError:
            If ``img`` is not 4D tensor

    Example:
        >>> from paddle import rand
        >>> from paddlemetrics.functional.image import total_variation
        >>> img = rand(5, 3, 28, 28)
        >>> total_variation(img)
        tensor(7546.8018)

    """
    score, num_elements = _total_variation_update(img)
    return _total_variation_compute(score, num_elements, reduction)
