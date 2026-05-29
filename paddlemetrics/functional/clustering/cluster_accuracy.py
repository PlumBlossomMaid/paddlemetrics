import sys

import paddle

from paddlemetrics.functional.classification import multiclass_confusion_matrix
from paddlemetrics.functional.clustering.utils import check_cluster_labels
from paddlemetrics.utils.imports import _TORCH_LINEAR_ASSIGNMENT_AVAILABLE

if not _TORCH_LINEAR_ASSIGNMENT_AVAILABLE:
    __doctest_skip__ = ["cluster_accuracy"]


def _cluster_accuracy_compute(confmat: paddle.Tensor) -> paddle.Tensor:
    """Computes the clustering accuracy from a confusion matrix."""
    from torch_linear_assignment import batch_linear_assignment

    confmat = confmat[None]
    assignment = batch_linear_assignment(confmat._max() - confmat)
    confmat = confmat[0]
    tps = confmat[paddle.arange(confmat.shape[0]), assignment.flatten()]
    return tps.sum() / confmat.sum()


def cluster_accuracy(
    preds: paddle.Tensor, target: paddle.Tensor, num_classes: int
) -> paddle.Tensor:
    """Computes the clustering accuracy between the predicted and target clusters.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels
        num_classes: number of classes

    Returns:
        Scalar tensor with clustering accuracy between 0.0 and 1.0

    Raises:
        RuntimeError:
            If `torch_linear_assignment` is not installed

    Example:
        >>> from paddlemetrics.functional.clustering import cluster_accuracy
        >>> preds = paddle.to_tensor([0, 0, 1, 1])
        >>> target = paddle.to_tensor([1, 1, 0, 0])
        >>> cluster_accuracy(preds, target, 2)
        tensor(1.000)

    """
    if not _TORCH_LINEAR_ASSIGNMENT_AVAILABLE:
        raise RuntimeError(
            "Missing `torch_linear_assignment`. Please install it with `pip install paddlemetrics[clustering]`."
        )
    check_cluster_labels(preds, target)
    confmat = multiclass_confusion_matrix(preds, target, num_classes=num_classes)
    return _cluster_accuracy_compute(confmat)
