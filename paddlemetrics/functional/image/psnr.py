from typing import Optional, Union

import paddle
from paddle import Tensor
from typing_extensions import Literal

from paddlemetrics.utils import rank_zero_warn, reduce


def _psnr_compute(
    sum_squared_error: paddle.Tensor,
    num_obs: paddle.Tensor,
    data_range: paddle.Tensor,
    base: float = 10.0,
    reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
) -> paddle.Tensor:
    """Compute peak signal-to-noise ratio.

    Args:
        sum_squared_error: Sum of square of errors over all observations
        num_obs: Number of predictions or observations
        data_range: the range of the data. If None, it is determined from the data (max - min).
           ``data_range`` must be given when ``dim`` is not None.
        base: a base of a logarithm to use
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

    Example:
        >>> preds = paddle.to_tensor([[0.0, 1.0], [2.0, 3.0]])
        >>> target = paddle.to_tensor([[3.0, 2.0], [1.0, 0.0]])
        >>> data_range = target.max() - target.min()
        >>> sum_squared_error, num_obs = _psnr_update(preds, target)
        >>> _psnr_compute(sum_squared_error, num_obs, data_range)
        tensor(2.5527)

    """
    psnr_base_e = 2 * paddle.log(data_range) - paddle.log(sum_squared_error / num_obs)
    psnr_vals = psnr_base_e * (10 / paddle.log(paddle.tensor(base)))
    return reduce(psnr_vals, reduction=reduction)


def _psnr_update(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    dim: Optional[Union[int, tuple[int, ...]]] = None,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Update and return variables required to compute peak signal-to-noise ratio.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        dim: Dimensions to reduce PSNR scores over provided as either an integer or a list of integers.
            Default is None meaning scores will be reduced across all dimensions.

    """
    if not preds.is_floating_point():
        preds = preds.to(paddle.float32)
    if not target.is_floating_point():
        target = target.to(paddle.float32)
    if dim is None:
        sum_squared_error = paddle.sum(paddle.pow(preds - target, 2))
        num_obs = paddle.tensor(target.size, device=target.place)
        return sum_squared_error, num_obs
    diff = preds - target
    sum_squared_error = paddle.sum(diff * diff, axis=dim)
    dim_list = [dim] if isinstance(dim, int) else list(dim)
    if not dim_list:
        num_obs = paddle.tensor(target.size, device=target.place)
    else:
        num_obs = paddle.tensor(target.size(), device=target.place)[dim_list].prod()
        num_obs = num_obs.expand_as(sum_squared_error)
    return sum_squared_error, num_obs


def peak_signal_noise_ratio(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    data_range: Union[float, tuple[float, float]],
    base: float = 10.0,
    reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
    dim: Optional[Union[int, tuple[int, ...]]] = None,
) -> paddle.Tensor:
    """Compute the peak signal-to-noise ratio.

    Args:
        preds: estimated signal
        target: groun truth signal
        data_range:
            the range of the data. If a tuple is provided then the range is calculated as the difference and
            input is clamped between the values.
        base: a base of a logarithm to use
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or None``: no reduction will be applied

        dim:
            Dimensions to reduce PSNR scores over provided as either an integer or a list of integers. Default is
            None meaning scores will be reduced across all dimensions.

    Return:
        Tensor with PSNR score

    Example:
        >>> from paddlemetrics.functional.image import peak_signal_noise_ratio
        >>> pred = paddle.to_tensor([[0.0, 1.0], [2.0, 3.0]])
        >>> target = paddle.to_tensor([[3.0, 2.0], [1.0, 0.0]])
        >>> peak_signal_noise_ratio(pred, target, data_range=3.0)
        tensor(2.5527)

    .. attention::
        Half precision is only support on GPU for this metric.

    """
    if dim is None and reduction != "elementwise_mean":
        rank_zero_warn(
            f"The `reduction={reduction}` will not have any effect when `dim` is None."
        )
    if isinstance(data_range, tuple):
        preds = paddle.clamp(preds, min=data_range[0], max=data_range[1])
        target = paddle.clamp(target, min=data_range[0], max=data_range[1])
        data_range_val = paddle.tensor(data_range[1] - data_range[0])
    else:
        data_range_val = paddle.tensor(float(data_range))
    sum_squared_error, num_obs = _psnr_update(preds, target, axis=dim)
    return _psnr_compute(
        sum_squared_error, num_obs, data_range_val, base=base, reduction=reduction
    )
