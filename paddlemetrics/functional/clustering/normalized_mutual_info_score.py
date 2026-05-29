from typing import Literal

import paddle

from paddlemetrics.functional.clustering.mutual_info_score import \
    mutual_info_score
from paddlemetrics.functional.clustering.utils import (
    _validate_average_method_arg, calculate_entropy,
    calculate_generalized_mean, check_cluster_labels)


def normalized_mutual_info_score(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    average_method: Literal["min", "geometric", "arithmetic", "max"] = "arithmetic",
) -> paddle.Tensor:
    """Compute normalized mutual information between two clusterings.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels
        average_method: normalizer computation method

    Returns:
        Scalar tensor with normalized mutual info score between 0.0 and 1.0

    Example:
        >>> from paddlemetrics.functional.clustering import normalized_mutual_info_score
        >>> target = paddle.to_tensor([0, 3, 2, 2, 1])
        >>> preds = paddle.to_tensor([1, 3, 2, 0, 1])
        >>> normalized_mutual_info_score(preds, target, "arithmetic")
        tensor(0.7919)

    """
    check_cluster_labels(preds, target)
    _validate_average_method_arg(average_method)
    mutual_info = mutual_info_score(preds, target)
    if paddle.allclose(
        x=mutual_info, y=paddle.tensor(0.0), atol=paddle.finfo().eps
    ).item():
        return mutual_info
    normalizer = calculate_generalized_mean(
        paddle.stack([calculate_entropy(preds), calculate_entropy(target)]),
        average_method,
    )
    return mutual_info / normalizer
