from typing import Optional

import paddle
from paddle import Tensor
from typing_extensions import Literal

from paddlemetrics.functional.image.d_lambda import spectral_distortion_index
from paddlemetrics.functional.image.d_s import spatial_distortion_index
from paddlemetrics.utils.imports import _TORCHVISION_AVAILABLE

if not _TORCHVISION_AVAILABLE:
    __doctest_skip__ = ["quality_with_no_reference"]


def quality_with_no_reference(
    preds: paddle.Tensor,
    ms: paddle.Tensor,
    pan: paddle.Tensor,
    pan_lr: Optional[paddle.Tensor] = None,
    alpha: float = 1,
    beta: float = 1,
    norm_order: int = 1,
    window_size: int = 7,
    reduction: Literal["elementwise_mean", "sum", "none"] = "elementwise_mean",
) -> paddle.Tensor:
    """Calculate `Quality with No Reference`_ (QualityWithNoReference_) also known as QNR.

    Metric is used to compare the joint spectral and spatial distortion between two images.

    Args:
        preds: High resolution multispectral image.
        ms: Low resolution multispectral image.
        pan: High resolution panchromatic image.
        pan_lr: Low resolution panchromatic image.
        alpha: Relevance of spectral distortion.
        beta: Relevance of spatial distortion.
        norm_order: Order of the norm applied on the difference.
        window_size: Window size of the filter applied to degrade the high resolution panchromatic image.
        reduction: A method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Return:
        Tensor with QualityWithNoReference score

    Raises:
        ValueError:
            If ``alpha`` or ``beta`` is not a non-negative real number.

    Example:
        >>> from paddle import rand
        >>> from paddlemetrics.functional.image import quality_with_no_reference
        >>> preds = rand([16, 3, 32, 32])
        >>> ms = rand([16, 3, 16, 16])
        >>> pan = rand([16, 3, 32, 32])
        >>> quality_with_no_reference(preds, ms, pan)
        tensor(0.9694)

    """
    if not isinstance(alpha, (int, float)) or alpha < 0:
        raise ValueError(
            f"Expected `alpha` to be a non-negative real number. Got alpha: {alpha}."
        )
    if not isinstance(beta, (int, float)) or beta < 0:
        raise ValueError(
            f"Expected `beta` to be a non-negative real number. Got beta: {beta}."
        )
    d_lambda = spectral_distortion_index(preds, ms, norm_order, reduction)
    d_s = spatial_distortion_index(
        preds, ms, pan, pan_lr, norm_order, window_size, reduction
    )
    return (1 - d_lambda) ** alpha * (1 - d_s) ** beta
