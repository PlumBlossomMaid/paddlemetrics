import sys

from typing import Literal

import paddle

from paddlemetrics.functional.clustering.mutual_info_score import (
    _mutual_info_score_compute, _mutual_info_score_update)
from paddlemetrics.functional.clustering.utils import (
    _validate_average_method_arg, calculate_entropy,
    calculate_generalized_mean)


def adjusted_mutual_info_score(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    average_method: Literal["min", "geometric", "arithmetic", "max"] = "arithmetic",
) -> paddle.Tensor:
    """Compute adjusted mutual information between two clusterings.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels
        average_method: normalizer computation method

    Returns:
        Scalar tensor with adjusted mutual info score between 0.0 and 1.0

    Example:
        >>> from paddlemetrics.functional.clustering import adjusted_mutual_info_score
        >>> preds = paddle.to_tensor([2, 1, 0, 1, 0])
        >>> target = paddle.to_tensor([0, 2, 1, 1, 0])
        >>> adjusted_mutual_info_score(preds, target, "arithmetic")
        tensor(-0.2500)

    """
    _validate_average_method_arg(average_method)
    contingency = _mutual_info_score_update(preds, target)
    mutual_info = _mutual_info_score_compute(contingency)
    expected_mutual_info = expected_mutual_info_score(contingency, target.size)
    normalizer = calculate_generalized_mean(
        paddle.stack([calculate_entropy(preds), calculate_entropy(target)]),
        average_method,
    )
    denominator = normalizer - expected_mutual_info
    if denominator < 0:
        denominator = paddle.min(
            paddle.tensor([denominator, -paddle.finfo(denominator.dtype).eps])
        )
    else:
        denominator = paddle.max(
            paddle.tensor([denominator, paddle.finfo(denominator.dtype).eps])
        )
    return (mutual_info - expected_mutual_info) / denominator


def expected_mutual_info_score(
    contingency: paddle.Tensor, n_samples: int
) -> paddle.Tensor:
    """Calculated expected mutual information score between two clusterings.

    Implementation taken from sklearn/metrics/cluster/_expected_mutual_info_fast.pyx.

    Args:
        contingency: contingency matrix
        n_samples: number of samples

    Returns:
        expected_mutual_info_score: expected mutual information score

    """
    n_rows, n_cols = contingency.shape
    a = paddle.ravel(contingency.sum(dim=1))
    b = paddle.ravel(contingency.sum(dim=0))
    if a.size == 1 or b.size == 1:
        return paddle.tensor(0.0, device=a.place)
    nijs = paddle.arange(
        0, max([a._max().item(), b._max().item()]) + 1, device=a.device
    )
    nijs[0] = 1
    term1 = nijs / n_samples
    log_a = paddle.log(a)
    log_b = paddle.log(b)
    log_nnij = paddle.log(paddle.tensor(n_samples, device=a.place)) + paddle.log(nijs)
    gln_a = paddle.lgamma(x=a + 1)
    gln_b = paddle.lgamma(x=b + 1)
    gln_na = paddle.lgamma(x=n_samples - a + 1)
    gln_nb = paddle.lgamma(x=n_samples - b + 1)
    gln_nnij = paddle.lgamma(x=nijs + 1) + paddle.lgamma(
        x=paddle.tensor(n_samples + 1, dtype=a.dtype, device=a.place)
    )
    emi = paddle.tensor(0.0, device=a.place)
    for i in range(n_rows):
        for j in range(n_cols):
            start = int(max(1, a[i].item() - n_samples + b[j].item()))
            end = int(min(a[i].item(), b[j].item()) + 1)
            for nij in range(start, end):
                term2 = log_nnij[nij] - log_a[i] - log_b[j]
                gln = (
                    gln_a[i]
                    + gln_b[j]
                    + gln_na[i]
                    + gln_nb[j]
                    - gln_nnij[nij]
                    - paddle.lgamma(x=a[i] - nij + 1)
                    - paddle.lgamma(x=b[j] - nij + 1)
                    - paddle.lgamma(x=n_samples - a[i] - b[j] + nij + 1)
                )
                term3 = paddle.exp(gln)
                emi += term1[nij] * term2 * term3
    return emi
