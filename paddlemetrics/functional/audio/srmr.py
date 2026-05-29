from functools import lru_cache
from math import ceil, pi
from typing import Optional

import paddle
from paddle import Tensor

from paddlemetrics.utils import rank_zero_warn
from paddlemetrics.utils.imports import _GAMMATONE_AVAILABLE
                                            
from paddlemetrics.utils.filtering import lfilter

if not _GAMMATONE_AVAILABLE:
    __doctest_skip__ = ["speech_reverberation_modulation_energy_ratio"]


@lru_cache(maxsize=100)
def _calc_erbs(
    low_freq: float, fs: int, n_filters: int, device: paddle.device
) -> paddle.Tensor:
    from gammatone.filters import centre_freqs

    ear_q = 9.26449
    min_bw = 24.7
    order = 1
    erbs = (
        (centre_freqs(fs, n_filters, low_freq) / ear_q) ** order + min_bw**order
    ) ** (1 / order)
    return paddle.to_tensor(erbs)


@lru_cache(maxsize=100)
def _make_erb_filters(
    fs: int, num_freqs: int, cutoff: float, device: paddle.device
) -> paddle.Tensor:
    from gammatone.filters import centre_freqs, make_erb_filters

    cfs = centre_freqs(fs, num_freqs, cutoff)
    fcoefs = make_erb_filters(fs, cfs)
    return paddle.to_tensor(fcoefs)


@lru_cache(maxsize=100)
def _compute_modulation_filterbank_and_cutoffs(
    min_cf: float, max_cf: float, n: int, fs: float, q: int, device: paddle.device
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    spacing_factor = (max_cf / min_cf) ** (1.0 / (n - 1))
    cfs = paddle.zeros([n], dtype=paddle.float64)
    cfs[0] = min_cf
    for k in range(1, n):
        cfs[k] = cfs[k - 1] * spacing_factor

    def _make_modulation_filter(w0: paddle.Tensor, q: int) -> paddle.Tensor:
        w0 = paddle.tan(x=w0 / 2)
        b0 = w0 / q
        b = paddle.to_tensor([b0, 0, -b0], dtype=paddle.float64)
        a = paddle.to_tensor(
            [1 + b0 + w0**2, 2 * w0**2 - 2, 1 - b0 + w0**2], dtype=paddle.float64
        )
        return paddle.stack([b, a], axis=0)

    mfb = paddle.stack(
        [_make_modulation_filter(w0, q) for w0 in 2 * pi * cfs / fs], axis=0
    )

    def _calc_cutoffs(
        cfs: paddle.Tensor, fs: float, q: int
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        w0 = 2 * pi * cfs / fs
        b0 = paddle.tan(x=w0 / 2) / q
        ll = cfs - b0 * fs / (2 * pi)
        rr = cfs + b0 * fs / (2 * pi)
        return ll, rr

    cfs = cfs
    mfb = mfb
    ll, rr = _calc_cutoffs(cfs, fs, q)
    return cfs, mfb, ll, rr


def _hilbert(x: paddle.Tensor, n: Optional[int] = None) -> paddle.Tensor:
    if x.is_complex():
        raise ValueError("x must be real.")
    if n is None:
        n = x.shape[-1]
        if n % 16:
            n = ceil(n / 16) * 16
    if n <= 0:
        raise ValueError("N must be positive.")
    x_fft = paddle.fft.fft(x, n=n, axis=-1)
    h = paddle.zeros([n], dtype=x.dtype, stop_gradient=True)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1 : n // 2] = 2
    else:
        h[0] = 1
        h[1 : (n + 1) // 2] = 2
    y = paddle.fft.ifft(x_fft * h, axis=-1)
    return y[..., : x.shape[-1]]


def _erb_filterbank(wave: paddle.Tensor, coefs: paddle.Tensor) -> paddle.Tensor:
    """Translated from gammatone package.

    Args:
        wave: shape [B, time]
        coefs: shape [N, 10]

    Returns:
        Tensor: shape [B, N, time]

    """
    num_batch, time = wave.shape
    wave = wave.astype(coefs.dtype).reshape([num_batch, 1, time])
    wave = wave.expand([-1, coefs.shape[0], -1])
    gain = coefs[:, 9]
    as1 = coefs[:, (0, 1, 5)]
    as2 = coefs[:, (0, 2, 5)]
    as3 = coefs[:, (0, 3, 5)]
    as4 = coefs[:, (0, 4, 5)]
    bs = coefs[:, 6:9]
    y1 = lfilter(wave, bs, as1, batching=True)
    y2 = lfilter(y1, bs, as2, batching=True)
    y3 = lfilter(y2, bs, as3, batching=True)
    y4 = lfilter(y3, bs, as4, batching=True)
    return y4 / gain.reshape([1, -1, 1])


def _normalize_energy(energy: paddle.Tensor, drange: float = 30.0) -> paddle.Tensor:
    """Normalize energy to a dynamic range of 30 dB.

    Args:
        energy: shape [B, N_filters, 8, n_frames]
        drange: dynamic range in dB

    """
    peak_energy = (
        paddle.mean(energy, axis=1, keepdim=True).max(keepdim=True, axis=2),
        paddle.mean(energy, axis=1, keepdim=True).argmax(keepdim=True, axis=2),
    )[0].values
    peak_energy = (
        peak_energy.max(keepdim=True, axis=3),
        peak_energy.argmax(keepdim=True, axis=3),
    )[0].values
    min_energy = peak_energy * 10.0 ** (-drange / 10.0)
    energy = paddle.where(energy < min_energy, min_energy, energy)
    return paddle.where(energy > peak_energy, peak_energy, energy)


def _cal_srmr_score(
    bw: paddle.Tensor, avg_energy: paddle.Tensor, cutoffs: paddle.Tensor
) -> paddle.Tensor:
    """Calculate srmr score."""
    if cutoffs[4] <= bw and cutoffs[5] > bw:
        kstar = 5
    elif cutoffs[5] <= bw and cutoffs[6] > bw:
        kstar = 6
    elif cutoffs[6] <= bw and cutoffs[7] > bw:
        kstar = 7
    elif cutoffs[7] <= bw:
        kstar = 8
    else:
        raise ValueError("Something wrong with the cutoffs compared to bw values.")
    return paddle.sum(avg_energy[:, :4]) / paddle.sum(avg_energy[:, 4:kstar])


def speech_reverberation_modulation_energy_ratio(
    preds: paddle.Tensor,
    fs: int,
    n_cochlear_filters: int = 23,
    low_freq: float = 125,
    min_cf: float = 4,
    max_cf: Optional[float] = None,
    norm: bool = False,
    fast: bool = False,
) -> paddle.Tensor:
    """Calculate `Speech-to-Reverberation Modulation Energy Ratio`_ (SRMR).

    SRMR is a non-intrusive metric for speech quality and intelligibility based on
    a modulation spectral representation of the speech signal.
    This code is translated from SRMRToolbox and `SRMRpy`_.

    Args:
        preds: shape ``(..., time)``
        fs: the sampling rate
        n_cochlear_filters: Number of filters in the acoustic filterbank
        low_freq: determines the frequency cutoff for the corresponding gammatone filterbank.
        min_cf: Center frequency in Hz of the first modulation filter.
        max_cf: Center frequency in Hz of the last modulation filter. If None is given,
            then 30 Hz will be used for `norm==False`, otherwise 128 Hz will be used.
        norm: Use modulation spectrum energy normalization
        fast: Use the faster version based on the gammatonegram.
            Note: this argument is inherited from `SRMRpy`_. As the translated code is based to pytorch,
            setting `fast=True` may slow down the speed for calculating this metric on GPU.

    .. hint::
        Usingsing this metrics requires you to have ``gammatone`` and ``paddleaudio`` installed.
        Either install as ``pip install paddlemetrics[audio]`` or ``pip install paddleaudio``
        and ``pip install git+https://github.com/detly/gammatone``.

    .. attention::
        This implementation is experimental, and might not be consistent with the matlab
        implementation SRMRToolbox, especially the fast implementation.
        The slow versions, a) ``fast=False, norm=False, max_cf=128``, b) ``fast=False, norm=True, max_cf=30``,
        have a relatively small inconsistency.

    Returns:
        Scalar tensor with srmr value with shape ``(...)``

    Raises:
        ModuleNotFoundError:
            If ``gammatone`` or ``paddleaudio`` package is not installed

    Example:
        >>> from paddle import randn
        >>> from paddlemetrics.functional.audio import speech_reverberation_modulation_energy_ratio
        >>> preds = randn([8000])
        >>> speech_reverberation_modulation_energy_ratio(preds, 8000)
        tensor([0.3191], dtype=paddle.float64)

    """
    if not _GAMMATONE_AVAILABLE:
        raise ModuleNotFoundError(
            "speech_reverberation_modulation_energy_ratio requires you to have `gammatone` and `paddleaudio` installed. Either install as ``pip install paddlemetrics[audio]`` or ``pip install paddleaudio`` and ``pip install git+https://github.com/detly/gammatone``"
        )
    from gammatone.fftweight import fft_gtgram

    _srmr_arg_validate(
        fs=fs,
        n_cochlear_filters=n_cochlear_filters,
        low_freq=low_freq,
        min_cf=min_cf,
        max_cf=max_cf,
        norm=norm,
        fast=fast,
    )
    shape = preds.shape
    preds = preds.reshape([1, -1]) if len(shape) == 1 else preds.reshape([-1, shape[-1]])
    num_batch, time = preds.shape
    if not paddle.is_floating_point(preds):
        preds = preds.astype(paddle.float64) / paddle.finfo(preds.dtype).max
    max_vals = (
        preds.abs().max(keepdim=True, axis=-1),
        preds.abs().argmax(keepdim=True, axis=-1),
    )[0].values
    val_norm = paddle.where(
        max_vals > 1,
        max_vals,
        paddle.to_tensor(1.0, dtype=max_vals.dtype),
    )
    preds = preds / val_norm
    w_length_s = 0.256
    w_inc_s = 0.064
    if fast:
        rank_zero_warn("`fast=True` may slow down the speed of SRMR metric on GPU.")
        mfs = 400.0
        temp = []
        preds_np = preds.detach().cpu().numpy()
        for b in range(num_batch):
            gt_env_b = fft_gtgram(
                preds_np[b], fs, 0.01, 0.0025, n_cochlear_filters, low_freq
            )
            temp.append(paddle.to_tensor(gt_env_b))
        gt_env = paddle.stack(temp, axis=0)
    else:
        fcoefs = _make_erb_filters(
            fs, n_cochlear_filters, low_freq, device=preds.place
        )
        gt_env = paddle.abs(_hilbert(_erb_filterbank(preds, fcoefs)))
        mfs = fs
    w_length = ceil(w_length_s * mfs)
    w_inc = ceil(w_inc_s * mfs)
    if max_cf is None:
        max_cf = 30 if norm else 128
    _, mf, cutoffs, _ = _compute_modulation_filterbank_and_cutoffs(
        min_cf, max_cf, n=8, fs=mfs, q=2, device=preds.place
    )
    num_frames = int(1 + (time - w_length) // w_inc)
    w = paddle.hamming_window(w_length + 1, dtype=paddle.float64)[:-1]
    mod_out = lfilter(
        gt_env.unsqueeze(-2).expand([-1, -1, mf.shape[0], -1]),
        mf[:, 1, :],
        mf[:, 0, :],
        clamp=False,
        batching=True,
    )
    padding = [0, max(ceil(time / w_inc) * w_inc - time, w_length - time)]
    mod_out_pad = paddle.nn.functional.pad(
        mod_out, pad=padding, mode="constant", value=0
    )
    mod_out_frame = mod_out_pad.unfold(axis=-1, size=w_length, step=w_inc)
    energy = ((mod_out_frame[..., :num_frames, :] * w) ** 2).sum(axis=-1)
    if norm:
        energy = _normalize_energy(energy)
    erbs = paddle.flip(
        x=_calc_erbs(low_freq, fs, n_cochlear_filters, device=preds.place), axis=0
    )
    avg_energy = paddle.mean(energy, axis=-1)
    total_energy = paddle.sum(avg_energy.reshape([num_batch, -1]), axis=-1)
    ac_energy = paddle.sum(avg_energy, axis=2)
    ac_perc = ac_energy * 100 / total_energy.reshape([-1, 1])
    ac_perc_cumsum = ac_perc.flip(axis=-1).cumsum(axis=-1)
    k90perc_idx = paddle.nonzero((ac_perc_cumsum > 90).cumsum(axis=-1) == 1)[:, 1]
    bw = erbs[k90perc_idx]
    temp = []
    for b in range(num_batch):
        score = _cal_srmr_score(bw[b], avg_energy[b], cutoffs=cutoffs)
        temp.append(score)
    score = paddle.stack(temp)
    return score.reshape([*shape[:-1]]) if len(shape) > 1 else score


def _srmr_arg_validate(
    fs: int,
    n_cochlear_filters: int = 23,
    low_freq: float = 125,
    min_cf: float = 4,
    max_cf: Optional[float] = 128,
    norm: bool = False,
    fast: bool = False,
) -> None:
    """Validate the arguments for speech_reverberation_modulation_energy_ratio.

    Args:
        fs: the sampling rate
        n_cochlear_filters: Number of filters in the acoustic filterbank
        low_freq: determines the frequency cutoff for the corresponding gammatone filterbank.
        min_cf: Center frequency in Hz of the first modulation filter.
        max_cf: Center frequency in Hz of the last modulation filter. If None is given,
        norm: Use modulation spectrum energy normalization
        fast: Use the faster version based on the gammatonegram.

    """
    if not (isinstance(fs, int) and fs > 0):
        raise ValueError(
            f"Expected argument `fs` to be an int larger than 0, but got {fs}"
        )
    if not (isinstance(n_cochlear_filters, int) and n_cochlear_filters > 0):
        raise ValueError(
            f"Expected argument `n_cochlear_filters` to be an int larger than 0, but got {n_cochlear_filters}"
        )
    if not (isinstance(low_freq, (float, int)) and low_freq > 0):
        raise ValueError(
            f"Expected argument `low_freq` to be a float larger than 0, but got {low_freq}"
        )
    if not (isinstance(min_cf, (float, int)) and min_cf > 0):
        raise ValueError(
            f"Expected argument `min_cf` to be a float larger than 0, but got {min_cf}"
        )
    if max_cf is not None and not (isinstance(max_cf, (float, int)) and max_cf > 0):
        raise ValueError(
            f"Expected argument `max_cf` to be a float larger than 0, but got {max_cf}"
        )
    if not isinstance(norm, bool):
        raise ValueError("Expected argument `norm` to be a bool value")
    if not isinstance(fast, bool):
        raise ValueError("Expected argument `fast` to be a bool value")