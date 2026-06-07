from __future__ import annotations

import paddle
from scipy.optimize import linear_sum_assignment

from paddlemetrics.functional.classification import multiclass_confusion_matrix
from paddlemetrics.functional.clustering.utils import check_cluster_labels


def _cluster_accuracy_compute(confmat: paddle.Tensor) -> paddle.Tensor:
    """Computes the clustering accuracy from a confusion matrix using the Hungarian algorithm."""
    cost = -(confmat.numpy())
    row_ind, col_ind = linear_sum_assignment(cost)
    tps = confmat[paddle.to_tensor(row_ind), paddle.to_tensor(col_ind)].sum()
    return tps / confmat.sum()


def cluster_accuracy(preds: paddle.Tensor, target: paddle.Tensor, num_classes: int) -> paddle.Tensor:
    """Computes the clustering accuracy between the predicted and target clusters.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels
        num_classes: number of classes

    Returns:
        Scalar tensor with clustering accuracy between 0.0 and 1.0

    Example:
        >>> from paddlemetrics.functional.clustering import cluster_accuracy
        >>> preds = paddle.to_tensor([0, 0, 1, 1])
        >>> target = paddle.to_tensor([1, 1, 0, 0])
        >>> cluster_accuracy(preds, target, 2)
        tensor(1.000)

    """
    check_cluster_labels(preds, target)
    confmat = multiclass_confusion_matrix(preds, target, num_classes=num_classes)
    return _cluster_accuracy_compute(confmat)
