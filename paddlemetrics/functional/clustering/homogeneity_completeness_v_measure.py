import paddle

from paddlemetrics.functional.clustering.mutual_info_score import \
    mutual_info_score
from paddlemetrics.functional.clustering.utils import (calculate_entropy,
                                                      check_cluster_labels)


def _homogeneity_score_compute(
    preds: paddle.Tensor, target: paddle.Tensor
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Computes the homogeneity score of a clustering given the predicted and target cluster labels."""
    check_cluster_labels(preds, target)
    if len(target) == 0:
        zero = paddle.tensor(0.0, dtype=paddle.float32, device=preds.place)
        return zero.clone(), zero.clone(), zero.clone(), zero.clone()
    entropy_target = calculate_entropy(target)
    entropy_preds = calculate_entropy(preds)
    mutual_info = mutual_info_score(preds, target)
    homogeneity = (
        mutual_info / entropy_target
        if entropy_target
        else paddle.ones_like(entropy_target)
    )
    return homogeneity, mutual_info, entropy_preds, entropy_target


def _completeness_score_compute(
    preds: paddle.Tensor, target: paddle.Tensor
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Computes the completeness score of a clustering given the predicted and target cluster labels."""
    homogeneity, mutual_info, entropy_preds, _ = _homogeneity_score_compute(
        preds, target
    )
    completeness = (
        mutual_info / entropy_preds
        if entropy_preds
        else paddle.ones_like(entropy_preds)
    )
    return completeness, homogeneity


def homogeneity_score(preds: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
    """Compute the Homogeneity score between two clusterings.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels

    Returns:
        scalar tensor with the rand score

    Example:
        >>> from paddlemetrics.functional.clustering import homogeneity_score
        >>> import paddle
        >>> homogeneity_score(paddle.to_tensor([0, 0, 1, 1]), paddle.to_tensor([1, 1, 0, 0]))
        tensor(1.)
        >>> homogeneity_score(paddle.to_tensor([0, 0, 1, 2]), paddle.to_tensor([0, 0, 1, 1]))
        tensor(1.)

    """
    homogeneity, _, _, _ = _homogeneity_score_compute(preds, target)
    return homogeneity


def completeness_score(preds: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
    """Compute the Completeness score between two clusterings.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels

    Returns:
        scalar tensor with the rand score

    Example:
        >>> from paddlemetrics.functional.clustering import completeness_score
        >>> import paddle
        >>> completeness_score(paddle.to_tensor([0, 0, 1, 1]), paddle.to_tensor([1, 1, 0, 0]))
        tensor(1.)
        >>> completeness_score(paddle.to_tensor([0, 0, 1, 2]), paddle.to_tensor([0, 0, 1, 1]))
        tensor(0.6667)

    """
    completeness, _ = _completeness_score_compute(preds, target)
    return completeness


def v_measure_score(
    preds: paddle.Tensor, target: paddle.Tensor, beta: float = 1.0
) -> paddle.Tensor:
    """Compute the V-measure score between two clusterings.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels
        beta: weight of the harmonic mean between homogeneity and completeness

    Returns:
        scalar tensor with the rand score

    Example:
        >>> from paddlemetrics.functional.clustering import v_measure_score
        >>> import paddle
        >>> v_measure_score(paddle.to_tensor([0, 0, 1, 1]), paddle.to_tensor([1, 1, 0, 0]))
        tensor(1.)
        >>> v_measure_score(paddle.to_tensor([0, 0, 1, 2]), paddle.to_tensor([0, 0, 1, 1]))
        tensor(0.8000)

    """
    completeness, homogeneity = _completeness_score_compute(preds, target)
    if homogeneity + completeness == 0.0:
        return paddle.ones_like(homogeneity)
    return (1 + beta) * homogeneity * completeness / (beta * homogeneity + completeness)
