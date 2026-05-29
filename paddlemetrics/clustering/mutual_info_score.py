from collections.abc import Sequence
from typing import Any, List, Optional, Union

import paddle

from paddlemetrics.functional.clustering.mutual_info_score import \
    mutual_info_score
from paddlemetrics.metric import Metric
from paddlemetrics.utils.data import dim_zero_cat
from paddlemetrics.utils.imports import _MATPLOTLIB_AVAILABLE
from paddlemetrics.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["MutualInfoScore.plot"]


class MutualInfoScore(Metric):
    """Compute `Mutual Information Score`_.

    .. math::
        MI(U,V) = \\sum_{i=1}^{|U|} \\sum_{j=1}^{|V|} \\frac{|U_i\\cap V_j|}{N}
        \\log\\frac{N|U_i\\cap V_j|}{|U_i||V_j|}

    Where :math:`U` is a tensor of target values, :math:`V` is a tensor of predictions,
    :math:`|U_i|` is the number of samples in cluster :math:`U_i`, and :math:`|V_i|` is the number of samples in
    cluster :math:`V_i`. The metric is symmetric, therefore swapping :math:`U` and :math:`V` yields the same mutual
    information score.

    This clustering metric is an extrinsic measure, because it requires ground truth clustering labels, which may not
    be available in practice since clustering in generally is used for unsupervised learning.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): single integer tensor with shape ``(N,)`` with predicted cluster labels
    - ``target`` (:class:`~paddle.Tensor`): single integer tensor with shape ``(N,)`` with ground truth cluster labels

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``mi_score`` (:class:`~paddle.Tensor`): A tensor with the Mutual Information Score

    Args:
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example::
        >>> import paddle
        >>> from paddlemetrics.clustering import MutualInfoScore
        >>> preds = paddle.to_tensor([2, 1, 0, 1, 0])
        >>> target = paddle.to_tensor([0, 2, 1, 1, 0])
        >>> mi_score = MutualInfoScore()
        >>> mi_score(preds, target)
        tensor(0.5004)

    """

    is_differentiable: bool = True
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    preds: List[paddle.Tensor]
    target: List[paddle.Tensor]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update state with predictions and targets."""
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> paddle.Tensor:
        """Compute mutual information over state."""
        return mutual_info_score(dim_zero_cat(self.preds), dim_zero_cat(self.target))

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
            >>> from paddlemetrics.clustering import MutualInfoScore
            >>> metric = MutualInfoScore()
            >>> metric.update(paddle.randint(0, 4, (10,)), paddle.randint(0, 4, (10,)))
            >>> fig_, ax_ = metric.plot(metric.compute())

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import paddle
            >>> from paddlemetrics.clustering import MutualInfoScore
            >>> metric = MutualInfoScore()
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(paddle.randint(0, 4, (10,)), paddle.randint(0, 4, (10,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
