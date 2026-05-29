import paddle

from paddlemetrics.functional.clustering.utils import (
    _validate_intrinsic_cluster_data, _validate_intrinsic_labels_to_samples)


def calinski_harabasz_score(
    data: paddle.Tensor, labels: paddle.Tensor
) -> paddle.Tensor:
    """Compute the Calinski Harabasz Score (also known as variance ratio criterion) for clustering algorithms.

    Args:
        data: float tensor with shape ``(N,d)`` with the embedded data.
        labels: single integer tensor with shape ``(N,)`` with cluster labels

    Returns:
        Scalar tensor with the Calinski Harabasz Score

    Example:
        >>> from paddle import randn, randint
        >>> from paddlemetrics.functional.clustering import calinski_harabasz_score
        >>> data = randn(20, 3)
        >>> labels = randint(0, 3, (20,))
        >>> calinski_harabasz_score(data, labels)
        tensor(2.2128)

    """
    _validate_intrinsic_cluster_data(data, labels)
    unique_labels, labels = paddle.unique(labels, return_inverse=True)
    num_labels = len(unique_labels)
    num_samples = data.shape[0]
    _validate_intrinsic_labels_to_samples(num_labels, num_samples)
    mean = data.mean(dim=0)
    between_cluster_dispersion = paddle.tensor(0.0, device=data.place)
    within_cluster_dispersion = paddle.tensor(0.0, device=data.place)
    for k in range(num_labels):
        cluster_k = data[labels == k, :]
        mean_k = cluster_k.mean(dim=0)
        between_cluster_dispersion += ((mean_k - mean) ** 2).sum() * cluster_k.shape[0]
        within_cluster_dispersion += ((cluster_k - mean_k) ** 2).sum()
    if within_cluster_dispersion == 0:
        return paddle.tensor(1.0, device=data.device, dtype=paddle.float32)
    return (
        between_cluster_dispersion
        * (num_samples - num_labels)
        / (within_cluster_dispersion * (num_labels - 1.0))
    )
