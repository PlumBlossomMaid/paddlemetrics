from itertools import permutations
from typing import Any, Callable

import numpy as np
import paddle
from paddle import Tensor
from typing_extensions import Literal

from paddlemetrics.utils import rank_zero_warn
from paddlemetrics.utils.imports import _SCIPY_AVAILABLE

_ps_dict: dict = {}


def _gen_permutations(spk_num: int, device: paddle.place) -> paddle.Tensor:
    key = str(spk_num) + str(device)
    if key not in _ps_dict:
        ps = paddle.to_tensor(list(permutations(range(spk_num))))
        _ps_dict[key] = ps
    else:
        ps = _ps_dict[key]
    return ps


def _find_best_perm_by_linear_sum_assignment(
    metric_mtx: paddle.Tensor, eval_func: Callable
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Solves the linear sum assignment problem.

    This implementation uses scipy and input is therefore transferred to cpu during calculations.

    Args:
        metric_mtx: the metric matrix, shape [batch_size, spk_num, spk_num]
        eval_func: the function to reduce the metric values of different the permutations

    Returns:
        best_metric: shape ``[batch]``
        best_perm: shape ``[batch, spk]``

    """
    from scipy.optimize import linear_sum_assignment

    mmtx = metric_mtx.detach().cpu()
    best_perm = paddle.to_tensor(
        np.array(
            [
                linear_sum_assignment(pwm, eval_func == paddle.max)[1]
                for pwm in mmtx
            ]
        )
    )
    best_perm = best_perm
    best_metric = paddle.gather(metric_mtx, 2, best_perm[:, :, None]).mean(axis=[-1, -2])
    return best_metric, best_perm


def _find_best_perm_by_exhaustive_method(
    metric_mtx: paddle.Tensor, eval_func: Callable
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Solves the linear sum assignment problem using exhaustive method.

    This is done by exhaustively calculating the metric values of all possible permutations, and returns the best metric
    values and the corresponding permutations.

    Args:
        metric_mtx: the metric matrix, shape ``[batch_size, spk_num, spk_num]``
        eval_func: the function to reduce the metric values of different the permutations

    Returns:
        best_metric: shape ``[batch]``
        best_perm: shape ``[batch, spk]``

    """
    batch_size, spk_num = metric_mtx.shape[:2]
    ps = _gen_permutations(spk_num=spk_num, device=metric_mtx.place)
    perm_num = ps.shape[0]
    bps = ps.T[None, ...].expand([batch_size, spk_num, perm_num])
    metric_of_ps_details = paddle.gather(metric_mtx, 2, bps)
    metric_of_ps = metric_of_ps_details.mean(axis=1)
    best_metric, best_indexes = eval_func(metric_of_ps, axis=1)
    best_indexes = best_indexes.detach()
    best_perm = ps[best_indexes, :]
    return best_metric, best_perm


def permutation_invariant_training(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    metric_func: Callable,
    mode: Literal["speaker-wise", "permutation-wise"] = "speaker-wise",
    eval_func: Literal["max", "min"] = "max",
    **kwargs: Any,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Calculate `Permutation invariant training`_ (PIT).

    This metric can evaluate models for speaker independent multi-talker speech separation in a permutation
    invariant way.

    Args:
        preds: float tensor with shape ``(batch_size,num_speakers,...)``
        target: float tensor with shape ``(batch_size,num_speakers,...)``
        metric_func: a metric function accept a batch of target and estimate.
            if `mode`==`'speaker-wise'`, then ``metric_func(preds[:, i, ...], target[:, j, ...])`` is called
            and expected to return a batch of metric tensors ``(batch,)``;

            if `mode`==`'permutation-wise'`, then ``metric_func(preds[:, p, ...], target[:, :, ...])`` is called,
            where `p` is one possible permutation, e.g. [0,1] or [1,0] for 2-speaker case, and expected to return
            a batch of metric tensors ``(batch,)``;

        mode: can be `'speaker-wise'` or `'permutation-wise'`.
        eval_func: the function to find the best permutation, can be ``'min'`` or ``'max'``,
            i.e. the smaller the better or the larger the better.
        kwargs: Additional args for metric_func

    Returns:
        Tuple of two float tensors. First tensor with shape ``(batch,)`` contains the best metric value for each sample
        and second tensor with shape ``(batch,)`` contains the best permutation.

    Example:
        >>> from paddlemetrics.functional.audio import scale_invariant_signal_distortion_ratio
        >>> # [batch, spk, time]
        >>> preds = paddle.to_tensor([[[-0.0579,  0.3560, -0.9604], [-0.1719,  0.3205,  0.2951]]])
        >>> target = paddle.to_tensor([[[ 1.0958, -0.1648,  0.5228], [-0.4100,  1.1942, -0.5103]]])
        >>> best_metric, best_perm = permutation_invariant_training(
        ...     preds, target, scale_invariant_signal_distortion_ratio,
        ...     mode="speaker-wise", eval_func="max")
        >>> best_metric
        tensor([-5.1091])
        >>> best_perm
        tensor([[0, 1]])
        >>> pit_permutate(preds, best_perm)
        tensor([[[-0.0579,  0.3560, -0.9604],
                 [-0.1719,  0.3205,  0.2951]]])

    """
    if preds.shape[0:2] != target.shape[0:2]:
        raise RuntimeError(
            "Predictions and targets are expected to have the same shape at the batch and speaker dimensions"
        )
    if eval_func not in ["max", "min"]:
        raise ValueError(f'eval_func can only be "max" or "min" but got {eval_func}')
    if mode not in ["speaker-wise", "permutation-wise"]:
        raise ValueError(
            f'mode can only be "speaker-wise" or "permutation-wise" but got {mode}'
        )
    if target.ndim < 2:
        raise ValueError(
            f"Inputs must be of shape [batch, spk, ...], got {target.shape} and {preds.shape} instead"
        )
    eval_op = paddle.max if eval_func == "max" else paddle.min
    batch_size, spk_num = target.shape[0:2]
    if mode == "permutation-wise":
        perms = _gen_permutations(spk_num=spk_num, device=preds.place)
        perm_num = perms.shape[0]
        ppreds = paddle.index_select(preds, axis=1, index=perms.reshape([-1])).reshape(
            [batch_size * perm_num, *preds.shape[1:]]
        )
        ptarget = target.repeat_interleave(repeats=perm_num, axis=0)
        metric_of_ps = metric_func(ppreds, ptarget, **kwargs)
        metric_of_ps = paddle.mean(
            metric_of_ps.reshape([batch_size, len(perms), -1]), axis=-1
        )
        best_metric, best_indexes = eval_op(metric_of_ps, axis=1)
        best_indexes = best_indexes.detach()
        best_perm = perms[best_indexes, :]
        return best_metric, best_perm
    first_ele = metric_func(preds[:, 0, ...], target[:, 0, ...], **kwargs)
    metric_mtx = paddle.empty(
        [batch_size, spk_num, spk_num], dtype=first_ele.dtype
    )
    metric_mtx[:, 0, 0] = first_ele
    for target_idx in range(spk_num):
        for preds_idx in range(spk_num):
            if target_idx == 0 and preds_idx == 0:
                continue
            metric_mtx[:, target_idx, preds_idx] = metric_func(
                preds[:, preds_idx, ...], target[:, target_idx, ...], **kwargs
            )
    if spk_num < 3 or not _SCIPY_AVAILABLE:
        if spk_num >= 3 and not _SCIPY_AVAILABLE:
            rank_zero_warn(
                f"In pit metric for speaker-num {spk_num}>3, we recommend installing scipy for better performance"
            )
        best_metric, best_perm = _find_best_perm_by_exhaustive_method(
            metric_mtx, eval_op
        )
    else:
        best_metric, best_perm = _find_best_perm_by_linear_sum_assignment(
            metric_mtx, eval_op
        )
    return best_metric, best_perm


def pit_permutate(preds: paddle.Tensor, perm: paddle.Tensor) -> paddle.Tensor:
    """Permutate estimate according to perm.

    Args:
        preds: the estimates you want to permutate, shape [batch, spk, ...]
        perm: the permutation returned from permutation_invariant_training, shape [batch, spk]

    Returns:
        Tensor: the permutated version of estimate

    """
    return paddle.stack(
        [paddle.index_select(pred, 0, p) for pred, p in zip(preds, perm)]
    )