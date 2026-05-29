import paddle
from paddle import Tensor
from typing_extensions import Literal

from paddlemetrics.functional.image.uqi import universal_image_quality_index
from paddlemetrics.utils.distributed import reduce


def _spectral_distortion_index_update(
    preds: paddle.Tensor, target: paddle.Tensor
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Update and returns variables required to compute Spectral Distortion Index.

    Args:
        preds: Low resolution multispectral image
        target: High resolution fused image

    """
    if preds.dtype != target.dtype:
        raise TypeError(
            f"Expected `ms` and `fused` to have the same data type. Got ms: {preds.dtype} and fused: {target.dtype}."
        )
    if len(preds.shape) != 4:
        raise ValueError(
            f"Expected `preds` and `target` to have BxCxHxW shape. Got preds: {preds.shape} and target: {target.shape}."
        )
    if preds.shape[:2] != target.shape[:2]:
        raise ValueError(
            f"Expected `preds` and `target` to have same batch and channel sizes.Got preds: {preds.shape} and target: {target.shape}."
        )
    return preds, target


def _spectral_distortion_index_compute(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    p: int = 1,
    reduction: Literal["elementwise_mean", "sum", "none"] = "elementwise_mean",
) -> paddle.Tensor:
    """Compute Spectral Distortion Index (SpectralDistortionIndex_).

    Args:
        preds: Low resolution multispectral image
        target: High resolution fused image
        p: a parameter to emphasize large spectral difference
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Example:
        >>> from paddle import rand
        >>> preds = rand([16, 3, 16, 16])
        >>> target = rand([16, 3, 16, 16])
        >>> preds, target = _spectral_distortion_index_update(preds, target)
        >>> _spectral_distortion_index_compute(preds, target)
        tensor(0.0234)

    """
    length = preds.shape[1]
    m1 = paddle.zeros((length, length), device=preds.place)
    m2 = paddle.zeros((length, length), device=preds.place)
    for k in range(length):
        num = length - (k + 1)
        if num == 0:
            continue
        stack1 = target[:, k : k + 1, :, :].repeat(num, 1, 1, 1)
        stack2 = paddle.concat(
            [target[:, r : r + 1, :, :] for r in range(k + 1, length)], axis=0
        )
        score = [
            s.mean()
            for s in universal_image_quality_index(
                stack1, stack2, reduction="none"
            ).split(preds.shape[0])
        ]
        m1[k, k + 1 :] = paddle.stack(score, 0)
        stack1 = preds[:, k : k + 1, :, :].repeat(num, 1, 1, 1)
        stack2 = paddle.concat(
            [preds[:, r : r + 1, :, :] for r in range(k + 1, length)], axis=0
        )
        score = [
            s.mean()
            for s in universal_image_quality_index(
                stack1, stack2, reduction="none"
            ).split(preds.shape[0])
        ]
        m2[k, k + 1 :] = paddle.stack(score, 0)
    m1 = m1 + m1.T
    m2 = m2 + m2.T
    diff = paddle.pow(paddle.abs(m1 - m2), p)
    if length == 1:
        output = paddle.pow(diff, 1.0 / p)
    else:
        output = paddle.pow(1.0 / (length * (length - 1)) * paddle.sum(diff), 1.0 / p)
    return reduce(output, reduction)


def spectral_distortion_index(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    p: int = 1,
    reduction: Literal["elementwise_mean", "sum", "none"] = "elementwise_mean",
) -> paddle.Tensor:
    """Calculate `Spectral Distortion Index`_ (SpectralDistortionIndex_) also known as D_lambda.

    Metric is used to compare the spectral distortion between two images.

    Args:
        preds: Low resolution multispectral image
        target: High resolution fused image
        p: Large spectral differences
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Return:
        Tensor with SpectralDistortionIndex score

    Raises:
        TypeError:
            If ``preds`` and ``target`` don't have the same data type.
        ValueError:
            If ``preds`` and ``target`` don't have ``BxCxHxW shape``.
        ValueError:
            If ``p`` is not a positive integer.

    Example:
        >>> from paddle import rand
        >>> from paddlemetrics.functional.image import spectral_distortion_index
        >>> preds = rand([16, 3, 16, 16])
        >>> target = rand([16, 3, 16, 16])
        >>> spectral_distortion_index(preds, target)
        tensor(0.0234)

    """
    if not isinstance(p, int) or p <= 0:
        raise ValueError(f"Expected `p` to be a positive integer. Got p: {p}.")
    preds, target = _spectral_distortion_index_update(preds, target)
    return _spectral_distortion_index_compute(preds, target, p, reduction)
