import math
from typing import Union

import paddle
from paddle import Tensor


def _compute_bef(x: paddle.Tensor, block_size: int = 8) -> paddle.Tensor:
    """Compute block effect.

    Args:
        x: input image
        block_size: integer indication the block size

    Returns:
        Computed block effect

    Raises:
        ValueError:
            If the image is not a grayscale image

    """
    _, channels, height, width = x.shape
    if channels > 1:
        raise ValueError(
            f"`psnrb` metric expects grayscale images, but got images with {channels} channels."
        )
    h = paddle.arange(width - 1)
    h_b = paddle.tensor(range(block_size - 1, width - 1, block_size))
    h_bc = paddle.tensor(list(set(h.tolist()).symmetric_difference(h_b.tolist())))
    v = paddle.arange(height - 1)
    v_b = paddle.tensor(range(block_size - 1, height - 1, block_size))
    v_bc = paddle.tensor(list(set(v.tolist()).symmetric_difference(v_b.tolist())))
    d_b = (x[:, :, :, h_b] - x[:, :, :, h_b + 1]).pow(2.0).sum()
    d_bc = (x[:, :, :, h_bc] - x[:, :, :, h_bc + 1]).pow(2.0).sum()
    d_b += (x[:, :, v_b, :] - x[:, :, v_b + 1, :]).pow(2.0).sum()
    d_bc += (x[:, :, v_bc, :] - x[:, :, v_bc + 1, :]).pow(2.0).sum()
    n_hb = height * (width / block_size) - 1
    n_hbc = height * (width - 1) - n_hb
    n_vb = width * (height / block_size) - 1
    n_vbc = width * (height - 1) - n_vb
    d_b /= n_hb + n_vb
    d_bc /= n_hbc + n_vbc
    t = math.log2(block_size) / math.log2(min(height, width)) if d_b > d_bc else 0
    return t * (d_b - d_bc)


def _psnrb_compute(
    sum_squared_error: paddle.Tensor,
    bef: paddle.Tensor,
    num_obs: paddle.Tensor,
    data_range: paddle.Tensor,
) -> paddle.Tensor:
    """Computes peak signal-to-noise ratio.

    Args:
        sum_squared_error: Sum of square of errors over all observations
        bef: block effect
        num_obs: Number of predictions or observations
        data_range: the range of the data.

    """
    sum_squared_error = sum_squared_error / num_obs + bef
    return 10 * paddle.log10(x=data_range**2 / sum_squared_error)


def _psnrb_update(
    preds: paddle.Tensor, target: paddle.Tensor, block_size: int = 8
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Updates and returns variables required to compute peak signal-to-noise ratio.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        block_size: Integer indication the block size

    """
    sum_squared_error = paddle.sum(paddle.pow(preds - target, 2))
    num_obs = paddle.tensor(target.size, device=target.place)
    bef = _compute_bef(preds, block_size=block_size)
    return sum_squared_error, bef, num_obs


def peak_signal_noise_ratio_with_blocked_effect(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    data_range: Union[float, tuple[float, float]],
    block_size: int = 8,
) -> paddle.Tensor:
    """Computes `Peak Signal to Noise Ratio With Blocked Effect` (PSNRB) metrics.

    .. math::
        \\text{PSNRB}(I, J) = 10 * \\log_{10} \\left(\\frac{\\max(I)^2}{\\text{MSE}(I, J)-\\text{B}(I, J)}\\right)

    Where :math:`\\text{MSE}` denotes the `mean-squared-error`_ function.

    Args:
        preds: estimated signal
        target: ground truth signal
        data_range: the range of the data. If a tuple is provided then the range is calculated as the difference and
            input is clamped between the values.
        block_size: integer indication the block size

    Return:
        Tensor with PSNRB score

    Example:
        >>> from paddle import rand
        >>> from paddlemetrics.functional.image import peak_signal_noise_ratio_with_blocked_effect
        >>> preds = rand(1, 1, 28, 28)
        >>> target = rand(1, 1, 28, 28)
        >>> peak_signal_noise_ratio_with_blocked_effect(preds, target, data_range=1.0)
        tensor(7.8402)

    """
    if isinstance(data_range, tuple):
        preds = paddle.clamp(preds, min=data_range[0], max=data_range[1])
        target = paddle.clamp(target, min=data_range[0], max=data_range[1])
        data_range_val = paddle.tensor(data_range[1] - data_range[0])
    else:
        data_range_val = paddle.tensor(float(data_range))
    sum_squared_error, bef, num_obs = _psnrb_update(
        preds, target, block_size=block_size
    )
    return _psnrb_compute(sum_squared_error, bef, num_obs, data_range_val)
