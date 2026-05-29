import math
from typing import Optional, Union

import paddle
from paddle import Tensor
from typing_extensions import Literal

from paddlemetrics.utils.checks import _check_same_shape
from paddlemetrics.utils.distributed import reduce


def _scc_update(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    hp_filter: paddle.Tensor,
    window_size: int,
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Update and returns variables required to compute Spatial Correlation Coefficient.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        hp_filter: High-pass filter tensor
        window_size: Local window size integer

    Return:
        Tuple of (preds, target, hp_filter) tensors

    Raises:
        ValueError:
            If ``preds`` and ``target`` have different number of channels
            If ``preds`` and ``target`` have different shapes
            If ``preds`` and ``target`` have invalid shapes
            If ``window_size`` is not a positive integer
            If ``window_size`` is greater than the size of the image

    """
    if preds.dtype != target.dtype:
        target = target.to(preds.dtype)
    _check_same_shape(preds, target)
    if preds.ndim not in (3, 4):
        raise ValueError(
            f"Expected `preds` and `target` to have batch of colored images with BxCxHxW shape  or batch of grayscale images of BxHxW shape. Got preds: {preds.shape} and target: {target.shape}."
        )
    if len(preds.shape) == 3:
        preds = preds.unsqueeze(1)
        target = target.unsqueeze(1)
    if not window_size > 0:
        raise ValueError(
            f"Expected `window_size` to be a positive integer. Got {window_size}."
        )
    if window_size > preds.size(2) or window_size > preds.size(3):
        raise ValueError(
            f"Expected `window_size` to be less than or equal to the size of the image. Got window_size: {window_size} and image size: {preds.size(2)}x{preds.size(3)}."
        )
    preds = preds.to(paddle.float32)
    target = target.to(paddle.float32)
    hp_filter = hp_filter[None, None, :].to(dtype=preds.dtype, device=preds.place)
    return preds, target, hp_filter


def _symmetric_reflect_pad_2d(
    input_img: paddle.Tensor, pad: Union[int, tuple[int, ...]]
) -> paddle.Tensor:
    """Applies symmetric padding to the 2D image tensor input using ``reflect`` mode (d c b a | a b c d | d c b a)."""
    if isinstance(paddle.nn.functional.pad, int):
        pad = pad, pad, pad, pad
    if len(pad) != 4:
        raise ValueError(f"Expected padding to have length 4, but got {len(pad)}")
    left_pad = input_img[:, :, :, 0 : paddle.nn.functional.pad[0]].flip(axis=[3])
    right_pad = input_img[:, :, :, -paddle.nn.functional.pad[1] :].flip(axis=[3])
    padded = paddle.concat([left_pad, input_img, right_pad], axis=3)
    top_pad = padded[:, :, 0 : paddle.nn.functional.pad[2], :].flip(axis=[2])
    bottom_pad = padded[:, :, -paddle.nn.functional.pad[3] :, :].flip(axis=[2])
    return paddle.concat([top_pad, padded, bottom_pad], axis=2)


def _signal_convolve_2d(
    input_img: paddle.Tensor, kernel: paddle.Tensor
) -> paddle.Tensor:
    """Applies 2D signal convolution to the input tensor with the given kernel."""
    left_padding = math.floor((kernel.size(3) - 1) / 2)
    right_padding = math.ceil((kernel.size(3) - 1) / 2)
    top_padding = math.floor((kernel.size(2) - 1) / 2)
    bottom_padding = math.ceil((kernel.size(2) - 1) / 2)
    padded = _symmetric_reflect_pad_2d(
        input_img, pad=(left_padding, right_padding, top_padding, bottom_padding)
    )
    kernel = kernel.flip(axis=[2, 3])
    return paddle.nn.functional.conv2d(padded, kernel, stride=1, padding=0)


def _hp_2d_laplacian(input_img: paddle.Tensor, kernel: paddle.Tensor) -> paddle.Tensor:
    """Applies 2-D Laplace filter to the input tensor with the given high pass filter."""
    return _signal_convolve_2d(input_img, kernel) * 2.0


def _local_variance_covariance(
    preds: paddle.Tensor, target: paddle.Tensor, window: paddle.Tensor
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Computes local variance and covariance of the input tensors."""
    left_padding = math.ceil((window.size(3) - 1) / 2)
    right_padding = math.floor((window.size(3) - 1) / 2)
    preds = paddle.nn.functional.pad(
        preds, (left_padding, right_padding, left_padding, right_padding)
    )
    target = paddle.nn.functional.pad(
        target, (left_padding, right_padding, left_padding, right_padding)
    )
    preds_mean = paddle.nn.functional.conv2d(preds, window, stride=1, padding=0)
    target_mean = paddle.nn.functional.conv2d(target, window, stride=1, padding=0)
    preds_var = (
        paddle.nn.functional.conv2d(preds**2, window, stride=1, padding=0)
        - preds_mean**2
    )
    target_var = (
        paddle.nn.functional.conv2d(target**2, window, stride=1, padding=0)
        - target_mean**2
    )
    target_preds_cov = (
        paddle.nn.functional.conv2d(target * preds, window, stride=1, padding=0)
        - target_mean * preds_mean
    )
    return preds_var, target_var, target_preds_cov


def _scc_per_channel_compute(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    hp_filter: paddle.Tensor,
    window_size: int,
) -> paddle.Tensor:
    """Computes per channel Spatial Correlation Coefficient.

    Args:
        preds: estimated image of Bx1xHxW shape.
        target: ground truth image of Bx1xHxW shape.
        hp_filter: 2D high-pass filter.
        window_size: size of window for local mean calculation.

    Return:
        Tensor with Spatial Correlation Coefficient score

    """
    dtype = preds.dtype
    device = preds.device
    window = (
        paddle.ones(size=(1, 1, window_size, window_size), dtype=dtype, device=device)
        / window_size**2
    )
    preds_hp = _hp_2d_laplacian(preds, hp_filter)
    target_hp = _hp_2d_laplacian(target, hp_filter)
    preds_var, target_var, target_preds_cov = _local_variance_covariance(
        preds_hp, target_hp, window
    )
    preds_var[preds_var < 0] = 0
    target_var[target_var < 0] = 0
    den = paddle.sqrt(target_var) * paddle.sqrt(preds_var)
    idx = den == 0
    den[den == 0] = 1
    scc = target_preds_cov / den
    scc[idx] = 0
    return scc


def spatial_correlation_coefficient(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    hp_filter: Optional[paddle.Tensor] = None,
    window_size: int = 8,
    reduction: Optional[Literal["mean", "none", None]] = "mean",
) -> paddle.Tensor:
    """Compute Spatial Correlation Coefficient (SCC_).

    Args:
        preds: predicted images of shape ``(N,C,H,W)`` or ``(N,H,W)``.
        target: ground truth images of shape ``(N,C,H,W)`` or ``(N,H,W)``.
        hp_filter: High-pass filter tensor. default: tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
        window_size: Local window size integer. default: 8,
        reduction: Reduction method for output tensor. If ``None`` or ``"none"``,
                   returns a tensor with the per sample results. default: ``"mean"``.

    Return:
        Tensor with scc score

    Example:
        >>> from paddle import randn
        >>> from paddlemetrics.functional.image import spatial_correlation_coefficient as scc
        >>> x = randn(5, 3, 16, 16)
        >>> scc(x, x)
        tensor(1.)
        >>> x = randn(5, 16, 16)
        >>> scc(x, x)
        tensor(1.)
        >>> x = randn(5, 3, 16, 16)
        >>> y = randn(5, 3, 16, 16)
        >>> scc(x, y, reduction="none")
        tensor([0.0223, 0.0256, 0.0616, 0.0159, 0.0170])

    """
    if hp_filter is None:
        hp_filter = paddle.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    if reduction is None:
        reduction = "none"
    if reduction not in ("mean", "none"):
        raise ValueError(
            f"Expected reduction to be 'mean' or 'none', but got {reduction}"
        )
    preds, target, hp_filter = _scc_update(preds, target, hp_filter, window_size)
    per_channel = [
        _scc_per_channel_compute(
            preds[:, i, :, :].unsqueeze(1),
            target[:, i, :, :].unsqueeze(1),
            hp_filter,
            window_size,
        )
        for i in range(preds.size(1))
    ]
    if reduction == "none":
        return paddle.mean(paddle.concat(per_channel, axis=1), axis=[1, 2, 3])
    if reduction == "mean":
        return reduce(paddle.concat(per_channel, axis=1), reduction="elementwise_mean")
    return None
