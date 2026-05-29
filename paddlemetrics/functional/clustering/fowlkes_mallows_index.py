import paddle

from paddlemetrics.functional.clustering.utils import (
    calculate_contingency_matrix, check_cluster_labels)


def _fowlkes_mallows_index_update(
    preds: paddle.Tensor, target: paddle.Tensor
) -> tuple[paddle.Tensor, int]:
    """Return contingency matrix required to compute the Fowlkes-Mallows index.

    Args:
        preds: predicted class labels
        target: ground truth class labels

    Returns:
        contingency: contingency matrix

    """
    check_cluster_labels(preds, target)
    return calculate_contingency_matrix(preds, target), preds.size(0)


def _fowlkes_mallows_index_compute(contingency: paddle.Tensor, n: int) -> paddle.Tensor:
    """Compute the Fowlkes-Mallows index based on the contingency matrix.

    Args:
        contingency: contingency matrix
        n: number of samples

    Returns:
        fowlkes_mallows: Fowlkes-Mallows index

    """
    tk = paddle.sum(contingency**2) - n
    if paddle.allclose(x=tk, y=paddle.tensor(0)).item():
        return paddle.tensor(0.0, device=contingency.place)
    pk = paddle.sum(contingency.sum(dim=0) ** 2) - n
    qk = paddle.sum(contingency.sum(dim=1) ** 2) - n
    return paddle.sqrt(tk / pk) * paddle.sqrt(tk / qk)


def fowlkes_mallows_index(preds: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
    """Compute Fowlkes-Mallows index between two clusterings.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels

    Returns:
        Scalar tensor with Fowlkes-Mallows index

    Example:
        >>> import paddle
        >>> from paddlemetrics.functional.clustering import fowlkes_mallows_index
        >>> preds = paddle.to_tensor([2, 2, 0, 1, 0])
        >>> target = paddle.to_tensor([2, 2, 1, 1, 0])
        >>> fowlkes_mallows_index(preds, target)
        tensor(0.5000)

    """
    contingency, n = _fowlkes_mallows_index_update(preds, target)
    return _fowlkes_mallows_index_compute(contingency, n)
