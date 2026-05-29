import paddle

from paddlemetrics.functional.clustering.utils import (
    calculate_contingency_matrix, check_cluster_labels)


def _mutual_info_score_update(
    preds: paddle.Tensor, target: paddle.Tensor
) -> paddle.Tensor:
    """Update and return variables required to compute the mutual information score.

    Args:
        preds: predicted class labels
        target: ground truth class labels

    Returns:
        contingency: contingency matrix

    """
    check_cluster_labels(preds, target)
    return calculate_contingency_matrix(preds, target)


def _mutual_info_score_compute(contingency: paddle.Tensor) -> paddle.Tensor:
    """Compute the mutual information score based on the contingency matrix.

    Args:
        contingency: contingency matrix

    Returns:
        mutual_info: mutual information score

    """
    n = contingency.sum()
    u = contingency.sum(dim=1)
    v = contingency.sum(dim=0)
    if u.size() == 1 or v.size() == 1:
        return paddle.tensor(0.0)
    nzu, nzv = paddle.nonzero(contingency, as_tuple=True)
    contingency = contingency[nzu, nzv]
    log_outer = paddle.log(u[nzu]) + paddle.log(v[nzv])
    mutual_info = (
        contingency / n * (paddle.log(n) + paddle.log(contingency) - log_outer)
    )
    return mutual_info.sum()


def mutual_info_score(preds: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
    """Compute mutual information between two clusterings.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels

    Example:
        >>> from paddlemetrics.functional.clustering import mutual_info_score
        >>> target = paddle.to_tensor([0, 3, 2, 2, 1])
        >>> preds = paddle.to_tensor([1, 3, 2, 0, 1])
        >>> mutual_info_score(preds, target)
        tensor(1.0549)

    """
    contingency = _mutual_info_score_update(preds, target)
    return _mutual_info_score_compute(contingency)
