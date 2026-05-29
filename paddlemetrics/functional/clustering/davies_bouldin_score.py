import paddle

from paddlemetrics.functional.clustering.utils import (
    _validate_intrinsic_cluster_data, _validate_intrinsic_labels_to_samples)


def davies_bouldin_score(data: paddle.Tensor, labels: paddle.Tensor) -> paddle.Tensor:
    """Compute the Davies bouldin score for clustering algorithms.

    Args:
        data: float tensor with shape ``(N,d)`` with the embedded data.
        labels: single integer tensor with shape ``(N,)`` with cluster labels

    Returns:
        Scalar tensor with the Davies bouldin score

    Example:
        >>> from paddle import randn, randint
        >>> from paddlemetrics.functional.clustering import davies_bouldin_score
        >>> data = randn(20, 3)
        >>> labels = randint(0, 3, (20,))
        >>> davies_bouldin_score(data, labels)
        tensor(2.7418)

    """
    _validate_intrinsic_cluster_data(data, labels)
    unique_labels, labels = paddle.unique(labels, return_inverse=True)
    num_labels = len(unique_labels)
    num_samples, dim = data.shape
    _validate_intrinsic_labels_to_samples(num_labels, num_samples)
    intra_dists = paddle.zeros(num_labels, device=data.place)
    centroids = paddle.zeros((num_labels, dim), device=data.place)
    for k in range(num_labels):
        cluster_k = data[labels == k, :]
        centroids[k] = cluster_k.mean(dim=0)
        intra_dists[k] = (cluster_k - centroids[k]).pow(2.0).sum(dim=1).sqrt().mean()
    centroid_distances = paddle.cdist(x=centroids, y=centroids)
    cond1 = paddle.allclose(x=intra_dists, y=paddle.zeros_like(intra_dists)).item()
    cond2 = paddle.allclose(
        x=centroid_distances, y=paddle.zeros_like(centroid_distances)
    ).item()
    if cond1 or cond2:
        return paddle.tensor(0.0, device=data.device, dtype=paddle.float32)
    centroid_distances[centroid_distances == 0] = float("inf")
    combined_intra_dists = intra_dists.unsqueeze(0) + intra_dists.unsqueeze(1)
    scores = (
        (combined_intra_dists / centroid_distances).max(axis=1),
        (combined_intra_dists / centroid_distances).argmax(axis=1),
    ).values
    return scores.mean()
