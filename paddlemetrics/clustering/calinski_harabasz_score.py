from collections.abc import Sequence
from typing import Any, List, Optional, Union

import paddle

from paddlemetrics.functional.clustering.calinski_harabasz_score import \
    calinski_harabasz_score
from paddlemetrics.metric import Metric
from paddlemetrics.utils.data import dim_zero_cat
from paddlemetrics.utils.imports import _MATPLOTLIB_AVAILABLE
from paddlemetrics.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["CalinskiHarabaszScore.plot"]


class CalinskiHarabaszScore(Metric):
    """Compute Calinski Harabasz Score (also known as variance ratio criterion) for clustering algorithms.

    .. math::
        CHS(X, L) = \\frac{B(X, L) \\cdot (n_\\text{samples} - n_\\text{labels})}{W(X, L) \\cdot (n_\\text{labels} - 1)}

    where :math:`B(X, L)` is the between-cluster dispersion, which is the squared distance between the cluster centers
    and the dataset mean, weighted by the size of the clusters, :math:`n_\\text{samples}` is the number of samples,
    :math:`n_\\text{labels}` is the number of labels, and :math:`W(X, L)` is the within-cluster dispersion e.g. the
    sum of squared distances between each samples and its closest cluster center.

    This clustering metric is an intrinsic measure, because it does not rely on ground truth labels for the evaluation.
    Instead it examines how well the clusters are separated from each other. The score is higher when clusters are dense
    and well separated, which relates to a standard concept of a cluster.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``data`` (:class:`~paddle.Tensor`): float tensor with shape ``(N,d)`` with the embedded data. ``d`` is the
      dimensionality of the embedding space.
    - ``labels`` (:class:`~paddle.Tensor`): single integer tensor with shape ``(N,)`` with cluster labels

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``chs`` (:class:`~paddle.Tensor`): A tensor with the Calinski Harabasz Score

    Args:
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example::
        >>> from paddle import randn, randint
        >>> from paddlemetrics.clustering import CalinskiHarabaszScore
        >>> data = randn(20, 3)
        >>> labels = randint(3, (20,))
        >>> metric = CalinskiHarabaszScore()
        >>> metric(data, labels)
        tensor(2.2128)

    """

    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    data: List[paddle.Tensor]
    labels: List[paddle.Tensor]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("data", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")

    def update(self, data: paddle.Tensor, labels: paddle.Tensor) -> None:
        """Update metric state with new data and labels."""
        self.data.append(data)
        self.labels.append(labels)

    def compute(self) -> paddle.Tensor:
        """Compute the Calinski Harabasz Score over all data and labels."""
        return calinski_harabasz_score(
            dim_zero_cat(self.data), dim_zero_cat(self.labels)
        )

    def plot(
        self,
        val: Union[paddle.Tensor, Sequence[paddle.Tensor], None] = None,
        ax: Optional[_AX_TYPE] = None,
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> import paddle
            >>> from paddlemetrics.clustering import CalinskiHarabaszScore
            >>> metric = CalinskiHarabaszScore()
            >>> metric.update(paddle.randn(20, 3), paddle.randint(3, (20,)))
            >>> fig_, ax_ = metric.plot(metric.compute())

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import paddle
            >>> from paddlemetrics.clustering import CalinskiHarabaszScore
            >>> metric = CalinskiHarabaszScore()
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(paddle.randn(20, 3), paddle.randint(3, (20,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
