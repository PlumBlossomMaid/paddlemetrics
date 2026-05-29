from collections.abc import Sequence
from typing import Any, List, Literal, Optional, Union

import paddle

from paddlemetrics.clustering.mutual_info_score import MutualInfoScore
from paddlemetrics.functional.clustering.normalized_mutual_info_score import (
    _validate_average_method_arg, normalized_mutual_info_score)
from paddlemetrics.utils.data import dim_zero_cat
from paddlemetrics.utils.imports import _MATPLOTLIB_AVAILABLE
from paddlemetrics.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["NormalizedMutualInfoScore.plot"]


class NormalizedMutualInfoScore(MutualInfoScore):
    """Compute `Normalized Mutual Information Score`_.

    .. math::
        NMI(U,V) = \\frac{MI(U,V)}{M_p(U,V)}

    Where :math:`U` is a tensor of target values, :math:`V` is a tensor of predictions, :math:`M_p(U,V)` is the
    generalized mean of order :math:`p` of :math:`U` and :math:`V`, and :math:`MI(U,V)` is the mutual information score
    between clusters :math:`U` and :math:`V`. The metric is symmetric, therefore swapping :math:`U` and :math:`V` yields
    the same mutual information score.

    This clustering metric is an extrinsic measure, because it requires ground truth clustering labels, which may not
    be available in practice since clustering in generally is used for unsupervised learning.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): single integer tensor with shape ``(N,)`` with predicted cluster labels
    - ``target`` (:class:`~paddle.Tensor`): single integer tensor with shape ``(N,)`` with ground truth cluster labels

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``nmi_score`` (:class:`~paddle.Tensor`): A tensor with the Normalized Mutual Information Score

    Args:
        average_method: Method used to calculate generalized mean for normalization
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example::
        >>> import paddle
        >>> from paddlemetrics.clustering import NormalizedMutualInfoScore
        >>> preds = paddle.to_tensor([2, 1, 0, 1, 0])
        >>> target = paddle.to_tensor([0, 2, 1, 1, 0])
        >>> nmi_score = NormalizedMutualInfoScore("arithmetic")
        >>> nmi_score(preds, target)
        tensor(0.4744)

    """

    is_differentiable: bool = True
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 0.0
    preds: List[paddle.Tensor]
    target: List[paddle.Tensor]

    def __init__(
        self,
        average_method: Literal["min", "geometric", "arithmetic", "max"] = "arithmetic",
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        _validate_average_method_arg(average_method)
        self.average_method = average_method

    def compute(self) -> paddle.Tensor:
        """Compute normalized mutual information over state."""
        return normalized_mutual_info_score(
            dim_zero_cat(self.preds), dim_zero_cat(self.target), self.average_method
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
            >>> from paddlemetrics.clustering import NormalizedMutualInfoScore
            >>> metric = NormalizedMutualInfoScore()
            >>> metric.update(paddle.randint(0, 4, (10,)), paddle.randint(0, 4, (10,)))
            >>> fig_, ax_ = metric.plot(metric.compute())

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import paddle
            >>> from paddlemetrics.clustering import NormalizedMutualInfoScore
            >>> metric = NormalizedMutualInfoScore()
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(paddle.randint(0, 4, (10,)), paddle.randint(0, 4, (10,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
