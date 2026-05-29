import paddle

from paddlemetrics.functional.clustering.utils import (
    calculate_contingency_matrix, calculate_pair_cluster_confusion_matrix,
    check_cluster_labels)


def _adjusted_rand_score_update(
    preds: paddle.Tensor, target: paddle.Tensor
) -> paddle.Tensor:
    """Update and return variables required to compute the rand score.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels

    Returns:
        contingency: contingency matrix

    """
    check_cluster_labels(preds, target)
    return calculate_contingency_matrix(preds, target)


def _adjusted_rand_score_compute(contingency: paddle.Tensor) -> paddle.Tensor:
    """Compute the rand score based on the contingency matrix.

    Args:
        contingency: contingency matrix

    Returns:
        rand_score: rand score

    """
    (tn, fp), (fn, tp) = calculate_pair_cluster_confusion_matrix(
        contingency=contingency
    )
    if fn == 0 and fp == 0:
        return paddle.ones_like(tn, dtype=paddle.float32)
    return 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))


def adjusted_rand_score(preds: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
    """Compute the Adjusted Rand score between two clusterings.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels

    Returns:
        Scalar tensor with adjusted rand score

    Example:
        >>> from paddlemetrics.functional.clustering import adjusted_rand_score
        >>> import paddle
        >>> adjusted_rand_score(paddle.to_tensor([0, 0, 1, 1]), paddle.to_tensor([0, 0, 1, 1]))
        tensor(1.)
        >>> adjusted_rand_score(paddle.to_tensor([0, 0, 1, 2]), paddle.to_tensor([0, 0, 1, 1]))
        tensor(0.5714)

    """
    contingency = _adjusted_rand_score_update(preds, target)
    return _adjusted_rand_score_compute(contingency)
