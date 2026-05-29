import math
from typing import Optional

import paddle
from paddle import Tensor

from paddlemetrics.utils import rank_zero_warn
from paddlemetrics.utils.checks import _check_same_shape
from paddlemetrics.utils.imports import _FAST_BSS_EVAL_AVAILABLE


def _symmetric_toeplitz(vector: paddle.Tensor) -> paddle.Tensor:
    """Construct a symmetric Toeplitz matrix using one vector.

    Args:
        vector: shape [..., L]

    Example:
        >>> from paddle import to_tensor
        >>> from paddlemetrics.functional.audio.sdr import _symmetric_toeplitz
        >>> v = to_tensor([0, 1, 2, 3, 4])
        >>> _symmetric_toeplitz(v)
        Tensor([[0, 1, 2, 3, 4],
                [1, 0, 1, 2, 3],
                [2, 1, 0, 1, 2],
                [3, 2, 1, 0, 1],
                [4, 3, 2, 1, 0]])

    Returns:
        a symmetric Toeplitz matrix of shape [..., L, L]

    """
    vec_exp = paddle.concat([paddle.flip(x=vector, axis=[-1]), vector[..., 1:]], axis=-1)
    v_len = vector.shape[-1]
    return paddle.as_strided(
        x=vec_exp,
        shape=(*vec_exp.shape[:-1], v_len, v_len),
        stride=(*vec_exp.strides[:-1], 1, 1),
    ).flip(axis=[-1])


def _compute_autocorr_crosscorr(
    target: paddle.Tensor, preds: paddle.Tensor, corr_len: int
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Compute the auto correlation of `target` and the cross correlation of `target` and `preds`.

    This calculation is done using the fast Fourier transform (FFT). Let's denotes the symmetric Toeplitz metric of the
    auto correlation of `target` as `R`, the cross correlation as 'b', then solving the equation `Rh=b` could have `h`
    as the coordinate of `preds` in the column space of the `corr_len` shifts of `target`.

    Args:
        target: the target (reference) signal of shape [..., time]
        preds: the preds (estimated) signal of shape [..., time]
        corr_len: the length of the auto correlation and cross correlation

    Returns:
        the auto correlation of `target` of shape [..., corr_len]
        the cross correlation of `target` and `preds` of shape [..., corr_len]

    """
    n_fft = 2 ** math.ceil(math.log2(preds.shape[-1] + target.shape[-1] - 1))
    t_fft = paddle.fft.rfft(target, n=n_fft, axis=-1)
    r_0 = paddle.fft.irfft(t_fft.real() ** 2 + t_fft.imag() ** 2, n=n_fft)[
        ..., :corr_len
    ]
    p_fft = paddle.fft.rfft(preds, n=n_fft, axis=-1)
    b = paddle.fft.irfft(t_fft.conj() * p_fft, n=n_fft, axis=-1)[..., :corr_len]
    return r_0, b


def signal_distortion_ratio(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    use_cg_iter: Optional[int] = None,
    filter_length: int = 512,
    zero_mean: bool = False,
    load_diag: Optional[float] = None,
) -> paddle.Tensor:
    """Calculate Signal to Distortion Ratio (SDR) metric. See `SDR ref1`_ and `SDR ref2`_ for details on the metric.

    .. note::
        The metric currently does not seem to work with Pytorch v1.11 and specific GPU hardware.

    Args:
        preds: float tensor with shape ``(...,time)``
        target: float tensor with shape ``(...,time)``
        use_cg_iter:
            If provided, conjugate gradient descent is used to solve for the distortion
            filter coefficients instead of direct Gaussian elimination, which requires that
            ``fast-bss-eval`` is installed and pytorch version >= 1.8.
            This can speed up the computation of the metrics in case the filters
            are long. Using a value of 10 here has been shown to provide
            good accuracy in most cases and is sufficient when using this
            loss to train neural separation networks.
        filter_length: The length of the distortion filter allowed
        zero_mean: When set to True, the mean of all signals is subtracted prior to computation of the metrics
        load_diag:
            If provided, this small value is added to the diagonal coefficients of
            the system metrics when solving for the filter coefficients.
            This can help stabilize the metric in the case where some reference signals may sometimes be zero

    Returns:
        Float tensor with shape ``(...,)`` of SDR values per sample

    Raises:
        RuntimeError:
            If ``preds`` and ``target`` does not have the same shape

    Example:
        >>> from paddle import randn
        >>> from paddlemetrics.functional.audio import signal_distortion_ratio
        >>> preds = randn([8000])
        >>> target = randn([8000])
        >>> signal_distortion_ratio(preds, target)
        tensor(-11.9930)
        >>> # use with permutation_invariant_training
        >>> from paddlemetrics.functional.audio import permutation_invariant_training
        >>> preds = randn([4, 2, 8000])  # [batch, spk, time]
        >>> target = randn([4, 2, 8000])
        >>> best_metric, best_perm = permutation_invariant_training(preds, target, signal_distortion_ratio)
        >>> best_metric
        tensor([-11.7748, -11.7948, -11.7160, -11.6254])
        >>> best_perm
        tensor([[1, 0],
                [1, 0],
                [1, 0],
                [0, 1]])

    """
    _check_same_shape(preds, target)
    preds_dtype = preds.dtype
    preds = preds.astype('float64')
    target = target.astype('float64')
    if zero_mean:
        preds = preds - preds.mean(axis=-1, keepdim=True)
        target = target - target.mean(axis=-1, keepdim=True)
    target = target / paddle.clip(
        paddle.linalg.norm(target, axis=-1, keepdim=True), min=1e-06
    )
    preds = preds / paddle.clip(
        paddle.linalg.norm(preds, axis=-1, keepdim=True), min=1e-06
    )
    r_0, b = _compute_autocorr_crosscorr(target, preds, corr_len=filter_length)
    if load_diag is not None:
        r_0[..., 0] += load_diag
    if use_cg_iter is not None and _FAST_BSS_EVAL_AVAILABLE:
        from fast_bss_eval.paddle.cgd import toeplitz_conjugate_gradient

        sol = toeplitz_conjugate_gradient(r_0, b, n_iter=use_cg_iter)
    else:
        if use_cg_iter is not None and not _FAST_BSS_EVAL_AVAILABLE:
            rank_zero_warn(
                "The `use_cg_iter` parameter of `SDR` requires that `fast-bss-eval` is installed. To make this this warning disappear, you could install `fast-bss-eval` using `pip install fast-bss-eval` or set `use_cg_iter=None`. For this time, the solver provided by Paddle is used.",
                UserWarning,
            )
        r = _symmetric_toeplitz(r_0)
        sol = paddle.linalg.solve(r, b)
    coh = paddle.einsum("...l,...l->...", b, sol)
    ratio = coh / (1 - coh)
    val = 10.0 * paddle.log10(x=ratio)
    if preds_dtype == paddle.float64:
        return val
    return val.astype(preds_dtype)


def scale_invariant_signal_distortion_ratio(
    preds: paddle.Tensor, target: paddle.Tensor, zero_mean: bool = False
) -> paddle.Tensor:
    """`Scale-invariant signal-to-distortion ratio`_ (SI-SDR).

    The SI-SDR value is in general considered an overall measure of how good a source sound.

    Args:
        preds: float tensor with shape ``(...,time)``
        target: float tensor with shape ``(...,time)``
        zero_mean: If to zero mean target and preds or not

    Returns:
        Float tensor with shape ``(...,)`` of SDR values per sample

    Raises:
        RuntimeError:
            If ``preds`` and ``target`` does not have the same shape

    Example:
        >>> from paddlemetrics.functional.audio import scale_invariant_signal_distortion_ratio
        >>> target = paddle.to_tensor([3.0, -0.5, 2.0, 7.0])
        >>> preds = paddle.to_tensor([2.5, 0.0, 2.0, 8.0])
        >>> scale_invariant_signal_distortion_ratio(preds, target)
        tensor(18.4030)

    """
    _check_same_shape(preds, target)
    eps = paddle.finfo(preds.dtype).eps
    if zero_mean:
        target = target - paddle.mean(target, axis=-1, keepdim=True)
        preds = preds - paddle.mean(preds, axis=-1, keepdim=True)
    alpha = (paddle.sum(preds * target, axis=-1, keepdim=True) + eps) / (
        paddle.sum(target**2, axis=-1, keepdim=True) + eps
    )
    target_scaled = alpha * target
    noise = target_scaled - preds
    val = (paddle.sum(target_scaled**2, axis=-1) + eps) / (
        paddle.sum(noise**2, axis=-1) + eps
    )
    return 10 * paddle.log10(x=val)


def source_aggregated_signal_distortion_ratio(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    scale_invariant: bool = True,
    zero_mean: bool = False,
) -> paddle.Tensor:
    """`Source-aggregated signal-to-distortion ratio`_ (SA-SDR).

    The SA-SDR is proposed to provide a stable gradient for meeting style source separation, where
    one-speaker and multiple-speaker scenes coexist.

    Args:
        preds: float tensor with shape ``(..., spk, time)``
        target: float tensor with shape ``(..., spk, time)``
        scale_invariant: if True, scale the targets of different speakers with the same alpha
        zero_mean: If to zero mean target and preds or not

    Returns:
        SA-SDR with shape ``(...)``

    Example:
        >>> from paddle import randn
        >>> from paddlemetrics.functional.audio import source_aggregated_signal_distortion_ratio
        >>> preds = randn([2, 8000])  # [..., spk, time]
        >>> target = randn([2, 8000])
        >>> source_aggregated_signal_distortion_ratio(preds, target)
        tensor(-50.8171)
        >>> # use with permutation_invariant_training
        >>> from paddlemetrics.functional.audio import permutation_invariant_training
        >>> preds = randn([4, 2, 8000])  # [batch, spk, time]
        >>> target = randn([4, 2, 8000])
        >>> best_metric, best_perm = permutation_invariant_training(preds, target,
        ...     source_aggregated_signal_distortion_ratio, mode="permutation-wise")
        >>> best_metric
        tensor([-42.6290, -44.3500, -34.7503, -54.1828])
        >>> best_perm
        tensor([[0, 1],
                [1, 0],
                [0, 1],
                [1, 0]])

    """
    _check_same_shape(preds, target)
    if preds.ndim < 2:
        raise RuntimeError(
            f"The preds and target should have the shape (..., spk, time), but {preds.shape} found"
        )
    eps = paddle.finfo(preds.dtype).eps
    if zero_mean:
        target = target - paddle.mean(target, axis=-1, keepdim=True)
        preds = preds - paddle.mean(preds, axis=-1, keepdim=True)
    if scale_invariant:
        alpha = (
            (preds * target).sum(axis=-1, keepdim=True).sum(axis=-2, keepdim=True) + eps
        ) / ((target**2).sum(axis=-1, keepdim=True).sum(axis=-2, keepdim=True) + eps)
        target = alpha * target
    distortion = target - preds
    val = ((target**2).sum(axis=-1).sum(axis=-1) + eps) / (
        (distortion**2).sum(axis=-1).sum(axis=-1) + eps
    )
    return 10 * paddle.log10(x=val)