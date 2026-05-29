from typing import Optional

import paddle

from paddlemetrics.utils.checks import _check_retrieval_functional_inputs


def retrieval_reciprocal_rank(
    preds: paddle.Tensor, target: paddle.Tensor, top_k: Optional[int] = None
) -> paddle.Tensor:
    """Compute reciprocal rank (for information retrieval). See `Mean Reciprocal Rank`_.

    ``preds`` and ``target`` should be of the same shape and live on the same device. If no ``target`` is ``True``,
    0 is returned. ``target`` must be either `bool` or `integers` and ``preds`` must be ``float``,
    otherwise an error is raised.

    Args:
        preds: estimated probabilities of each document to be relevant.
        target: ground truth about each document being relevant or not.
        top_k: consider only the top k elements (default: ``None``, which considers them all)

    Return:
        a single-value tensor with the reciprocal rank (RR) of the predictions ``preds`` wrt the labels ``target``.

    Raises:
        ValueError:
            If ``top_k`` is not ``None`` or an integer larger than 0.

    Example:
        >>> from paddlemetrics.functional.retrieval import retrieval_reciprocal_rank
        >>> preds = paddle.to_tensor([0.2, 0.3, 0.5])
        >>> target = paddle.to_tensor([False, False])
        >>> retrieval_reciprocal_rank(preds, target)
        tensor(0.5000)

    """
    preds, target = _check_retrieval_functional_inputs(preds, target)
    top_k = top_k or preds.shape[-1]
    if not isinstance(top_k, int) and top_k <= 0:
        raise ValueError(
            f"Argument ``top_k`` has to be a positive integer or None, but got {top_k}."
        )
    target = paddle.where(preds > 0, target, paddle.zeros_like(target))
    target = target[preds.topk(min(top_k, preds.shape[-1]), sorted=True, axis=-1)[1]]
    if not target.sum():
        return paddle.tensor(0.0, device=preds.place)
    position = paddle.nonzero(target).view(-1)
    return 1.0 / (position[0] + 1.0)
