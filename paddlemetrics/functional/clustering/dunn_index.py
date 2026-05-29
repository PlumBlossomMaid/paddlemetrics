import sys

from itertools import combinations

import paddle


def _dunn_index_update(
    data: paddle.Tensor, labels: paddle.Tensor, p: float
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Update and return variables required to compute the Dunn index.

    Args:
        data: feature vectors of shape (n_samples, n_features)
        labels: cluster labels
        p: p-norm (distance metric)

    Returns:
        intercluster_distance: intercluster distances
        max_intracluster_distance: max intracluster distances

    """
    unique_labels, inverse_indices = labels.unique(return_inverse=True)
    clusters = [
        data[inverse_indices == label_idx] for label_idx in range(len(unique_labels))
    ]
    centroids = [c.mean(dim=0) for c in clusters]
    intercluster_distance = paddle.linalg.norm(
        paddle.stack([(a - b) for a, b in combinations(centroids, 2)], axis=0),
        ord=p, axis=1,
    )
    max_intracluster_distance = paddle.stack(
        [
            paddle.linalg.norm(ci - mu, ord=p, axis=1)._max()
            for ci, mu in zip(clusters, centroids)
        ]
    )
    return intercluster_distance, max_intracluster_distance


def _dunn_index_compute(
    intercluster_distance: paddle.Tensor, max_intracluster_distance: paddle.Tensor
) -> paddle.Tensor:
    """Compute the Dunn index based on updated state.

    Args:
        intercluster_distance: intercluster distances
        max_intracluster_distance: max intracluster distances

    Returns:
        scalar tensor with the dunn index

    """
    return intercluster_distance._min() / max_intracluster_distance._max()


def dunn_index(
    data: paddle.Tensor, labels: paddle.Tensor, p: float = 2
) -> paddle.Tensor:
    """Compute the Dunn index.

    Args:
        data: feature vectors
        labels: cluster labels
        p: p-norm used for distance metric

    Returns:
        scalar tensor with the dunn index

    Example:
        >>> from paddlemetrics.functional.clustering import dunn_index
        >>> data = paddle.to_tensor([[0, 0], [0.5, 0], [1, 0], [0.5, 1]])
        >>> labels = paddle.to_tensor([0, 0, 0, 1])
        >>> dunn_index(data, labels)
        tensor(2.)

    """
    pairwise_distance, max_distance = _dunn_index_update(data, labels, p)
    return _dunn_index_compute(pairwise_distance, max_distance)
