import paddle

from paddlemetrics.functional.clustering.utils import (
    calculate_contingency_matrix, calculate_pair_cluster_confusion_matrix,
    check_cluster_labels)


def _rand_score_update(preds: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
    """Update and return variables required to compute the rand score.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels

    Returns:
        contingency: contingency matrix

    """
    check_cluster_labels(preds, target)
    return calculate_contingency_matrix(preds, target)


def _rand_score_compute(contingency: paddle.Tensor) -> paddle.Tensor:
    """Compute the rand score based on the contingency matrix.

    Args:
        contingency: contingency matrix

    Returns:
        rand_score: rand score

    """
    pair_matrix = calculate_pair_cluster_confusion_matrix(contingency=contingency)
    numerator = pair_matrix.diagonal().sum()
    denominator = pair_matrix.sum()
    if numerator == denominator or denominator == 0:
        return paddle.ones_like(numerator, dtype=paddle.float32)
    return numerator / denominator


def rand_score(preds: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
    """Compute the Rand score between two clusterings.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels

    Returns:
        scalar tensor with the rand score

    Example:
        >>> from paddlemetrics.functional.clustering import rand_score
        >>> import paddle
        >>> rand_score(paddle.to_tensor([0, 0, 1, 1]), paddle.to_tensor([1, 1, 0, 0]))
        tensor(1.)
        >>> rand_score(paddle.to_tensor([0, 0, 1, 2]), paddle.to_tensor([0, 0, 1, 1]))
        tensor(0.8333)

    """
    contingency = _rand_score_update(preds, target)
    return _rand_score_compute(contingency)
