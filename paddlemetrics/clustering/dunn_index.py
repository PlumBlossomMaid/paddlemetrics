from collections.abc import Sequence
from typing import Any, List, Optional, Union

import paddle

from paddlemetrics.functional.clustering.dunn_index import dunn_index
from paddlemetrics.metric import Metric
from paddlemetrics.utils.data import dim_zero_cat
from paddlemetrics.utils.imports import _MATPLOTLIB_AVAILABLE
from paddlemetrics.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["DunnIndex.plot"]


class DunnIndex(Metric):
    """Compute `Dunn Index`_.

    .. math::
        DI_m = \\frac{\\min_{1\\leq i<j\\leq m} \\delta(C_i,C_j)}{\\max_{1\\leq k\\leq m} \\Delta_k}

    Where :math:`C_i` is a cluster of tensors, :math:`C_j` is a cluster of tensors,
    and :math:`\\delta(C_i,C_j)` is the intercluster distance metric for :math:`m` clusters.

    This clustering metric is an intrinsic measure, because it does not rely on ground truth labels for the evaluation.
    Instead it examines how well the clusters are separated from each other. The score is higher when clusters are dense
    and well separated, which relates to a standard concept of a cluster.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``data`` (:class:`~paddle.Tensor`): float tensor with shape ``(N,d)`` with the embedded data. ``d`` is the
      dimensionality of the embedding space.
    - ``labels`` (:class:`~paddle.Tensor`): single integer tensor with shape ``(N,)`` with cluster labels

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``dunn_index`` (:class:`~paddle.Tensor`): A tensor with the Dunn Index

    Args:
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example::
        >>> import paddle
        >>> from paddlemetrics.clustering import DunnIndex
        >>> data = paddle.to_tensor([[0, 0], [0.5, 0], [1, 0], [0.5, 1]])
        >>> labels = paddle.to_tensor([0, 0, 0, 1])
        >>> dunn_index = DunnIndex(p=2)
        >>> dunn_index(data, labels)
        tensor(2.)

    """

    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    data: List[paddle.Tensor]
    labels: List[paddle.Tensor]

    def __init__(self, p: float = 2, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.p = p
        self.add_state("data", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")

    def update(self, data: paddle.Tensor, labels: paddle.Tensor) -> None:
        """Update state with predictions and targets."""
        self.data.append(data)
        self.labels.append(labels)

    def compute(self) -> paddle.Tensor:
        """Compute mutual information over state."""
        return dunn_index(dim_zero_cat(self.data), dim_zero_cat(self.labels), self.p)

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
            >>> from paddlemetrics.clustering import DunnIndex
            >>> data = paddle.to_tensor([[0, 0], [0.5, 0], [1, 0], [0.5, 1]])
            >>> labels = paddle.to_tensor([0, 0, 0, 1])
            >>> metric = DunnIndex(p=2)
            >>> metric.update(data, labels)
            >>> fig_, ax_ = metric.plot(metric.compute())

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import paddle
            >>> from paddlemetrics.clustering import DunnIndex
            >>> metric = DunnIndex(p=2)
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(paddle.randn(50, 3), paddle.randint(0, 2, (50,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
