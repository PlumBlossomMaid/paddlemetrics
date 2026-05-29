from typing import Optional

import paddle
from paddle import Tensor

from paddlemetrics.utils.checks import _check_retrieval_functional_inputs
from paddlemetrics.utils.data import _cumsum


def retrieval_precision_recall_curve(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    max_k: Optional[int] = None,
    adaptive_k: bool = False,
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Compute precision-recall pairs for different k (from 1 to `max_k`).

    In a ranked retrieval context, appropriate sets of retrieved documents are naturally given by
    the top k retrieved documents.

    Recall is the fraction of relevant documents retrieved among all the relevant documents.
    Precision is the fraction of relevant documents among all the retrieved documents.

    For each such set, precision and recall values can be plotted to give a recall-precision
    curve.

    ``preds`` and ``target`` should be of the same shape and live on the same device. If no ``target`` is ``True``,
    ``0`` is returned. ``target`` must be either `bool` or `integers` and ``preds`` must be ``float``,
    otherwise an error is raised.

    Args:
        preds: estimated probabilities of each document to be relevant.
        target: ground truth about each document being relevant or not.
        max_k: Calculate recall and precision for all possible top k from 1 to max_k
               (default: `None`, which considers all possible top k)
        adaptive_k: adjust `max_k` to `min(max_k, number of documents)` for each query

    Returns:
        Tensor with the precision values for each k (at ``top_k``) from 1 to `max_k`
        Tensor with the recall values for each k (at ``top_k``) from 1 to `max_k`
        Tensor with all possibles k

    Raises:
        ValueError:
            If ``max_k`` is not `None` or an integer larger than 0.
        ValueError:
            If ``adaptive_k`` is not boolean.

    Example:
        >>> from paddle import tensor
        >>> from  paddlemetrics.functional import retrieval_precision_recall_curve
        >>> preds = tensor([0.2, 0.3, 0.5])
        >>> target = tensor([True, False])
        >>> precisions, recalls, top_k = retrieval_precision_recall_curve(preds, target, max_k=2)
        >>> precisions
        tensor([1.0000, 0.5000])
        >>> recalls
        tensor([0.5000, 0.5000])
        >>> top_k
        tensor([1, 2])

    """
    preds, target = _check_retrieval_functional_inputs(preds, target)
    if not isinstance(adaptive_k, bool):
        raise ValueError("`adaptive_k` has to be a boolean")
    if max_k is None:
        max_k = preds.shape[-1]
    if not (isinstance(max_k, int) and max_k > 0):
        raise ValueError("`max_k` has to be a positive integer or None")
    if adaptive_k and max_k > preds.shape[-1]:
        topk = paddle.arange(1, preds.shape[-1] + 1, device=preds.place)
        topk = paddle.nn.functional.pad(
            topk, (0, max_k - preds.shape[-1]), "constant", float(preds.shape[-1])
        )
    else:
        topk = paddle.arange(1, max_k + 1, device=preds.place)
    if not target.sum():
        return (
            paddle.zeros(max_k, device=preds.place),
            paddle.zeros(max_k, device=preds.place),
            topk,
        )
    relevant = target[preds.topk(min(max_k, preds.shape[-1]), axis=-1)[1]].float()
    relevant = _cumsum(
        paddle.nn.functional.pad(
            relevant, (0, max(0, max_k - len(relevant))), "constant", 0.0
        ), axis=0,
    )
    recall = relevant / target.sum()
    precision = relevant / topk
    return precision, recall, topk
