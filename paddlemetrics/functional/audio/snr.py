import paddle

from paddlemetrics.functional.audio.sdr import \
    scale_invariant_signal_distortion_ratio
from paddlemetrics.utils.checks import _check_same_shape


def signal_noise_ratio(
    preds: paddle.Tensor, target: paddle.Tensor, zero_mean: bool = False
) -> paddle.Tensor:
    """Calculate `Signal-to-noise ratio`_ (SNR_) meric for evaluating quality of audio.

    .. math::
        \\text{SNR} = \\frac{P_{signal}}{P_{noise}}

    where  :math:`P` denotes the power of each signal. The SNR metric compares the level of the desired signal to
    the level of background noise. Therefore, a high value of SNR means that the audio is clear.

    Args:
        preds: float tensor with shape ``(...,time)``
        target: float tensor with shape ``(...,time)``
        zero_mean: if to zero mean target and preds or not

    Returns:
        Float tensor with shape ``(...,)`` of SNR values per sample

    Raises:
        RuntimeError:
            If ``preds`` and ``target`` does not have the same shape

    Example:
        >>> from paddlemetrics.functional.audio import signal_noise_ratio
        >>> target = paddle.to_tensor([3.0, -0.5, 2.0, 7.0])
        >>> preds = paddle.to_tensor([2.5, 0.0, 2.0, 8.0])
        >>> signal_noise_ratio(preds, target)
        tensor(16.1805)

    """
    _check_same_shape(preds, target)
    eps = paddle.finfo(preds.dtype).eps
    if zero_mean:
        target = target - paddle.mean(target, axis=-1, keepdim=True)
        preds = preds - paddle.mean(preds, axis=-1, keepdim=True)
    noise = target - preds
    snr_value = (paddle.sum(target**2, axis=-1) + eps) / (
        paddle.sum(noise**2, axis=-1) + eps
    )
    return 10 * paddle.log10(x=snr_value)


def scale_invariant_signal_noise_ratio(
    preds: paddle.Tensor, target: paddle.Tensor
) -> paddle.Tensor:
    """`Scale-invariant signal-to-noise ratio`_ (SI-SNR).

    Args:
        preds: float tensor with shape ``(...,time)``
        target: float tensor with shape ``(...,time)``

    Returns:
         Float tensor with shape ``(...,)`` of SI-SNR values per sample

    Raises:
        RuntimeError:
            If ``preds`` and ``target`` does not have the same shape

    Example:
        >>> import paddle
        >>> from paddlemetrics.functional.audio import scale_invariant_signal_noise_ratio
        >>> target = paddle.to_tensor([3.0, -0.5, 2.0, 7.0])
        >>> preds = paddle.to_tensor([2.5, 0.0, 2.0, 8.0])
        >>> scale_invariant_signal_noise_ratio(preds, target)
        tensor(15.0918)

    """
    return scale_invariant_signal_distortion_ratio(
        preds=preds, target=target, zero_mean=True
    )


def complex_scale_invariant_signal_noise_ratio(
    preds: paddle.Tensor, target: paddle.Tensor, zero_mean: bool = False
) -> paddle.Tensor:
    """`Complex scale-invariant signal-to-noise ratio`_ (C-SI-SNR).

    Args:
        preds: real float tensor with shape ``(...,frequency,time,2)`` or complex float tensor with
            shape ``(..., frequency,time)``
        target: real float tensor with shape ``(...,frequency,time,2)`` or complex float tensor with
            shape ``(..., frequency,time)``
        zero_mean: When set to True, the mean of all signals is subtracted prior to computation of the metrics

    Returns:
         Float tensor with shape ``(...,)`` of C-SI-SNR values per sample

    Raises:
        RuntimeError:
            If ``preds`` is not the shape (...,frequency,time,2) (after being converted to real if it is complex).
            If ``preds`` and ``target`` does not have the same shape.

    Example:
        >>> from paddle import randn
        >>> from paddlemetrics.functional.audio import complex_scale_invariant_signal_noise_ratio
        >>> preds = randn([1, 257, 100, 2])
        >>> target = randn([1, 257, 100, 2])
        >>> complex_scale_invariant_signal_noise_ratio(preds, target)
        tensor([-38.8832])

    """
    if preds.is_complex():
        preds = paddle.as_real(preds)
    if target.is_complex():
        target = paddle.as_real(target)
    if (preds.ndim < 3 or preds.shape[-1] != 2) or (
        target.ndim < 3 or target.shape[-1] != 2
    ):
        raise RuntimeError(
            f"Predictions and targets are expected to have the shape (..., frequency, time, 2), but got {preds.shape} and {target.shape}."
        )
    preds = preds.reshape([*preds.shape[:-3], -1])
    target = target.reshape([*target.shape[:-3], -1])
    return scale_invariant_signal_distortion_ratio(
        preds=preds, target=target, zero_mean=zero_mean
    )