from collections.abc import Sequence
from typing import Any, List, Optional, Union

import paddle
from typing_extensions import Literal

from paddlemetrics.functional.regression.cosine_similarity import (
    _cosine_similarity_compute, _cosine_similarity_update)
from paddlemetrics.metric import Metric
from paddlemetrics.utils.data import dim_zero_cat
from paddlemetrics.utils.imports import _MATPLOTLIB_AVAILABLE
from paddlemetrics.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["CosineSimilarity.plot"]


class CosineSimilarity(Metric):
    """Compute the `Cosine Similarity`_.

    .. math::
        cos_{sim}(x,y) = \\frac{x \\cdot y}{||x|| \\cdot ||y||} =
        \\frac{\\sum_{i=1}^n x_i y_i}{\\sqrt{\\sum_{i=1}^n x_i^2}\\sqrt{\\sum_{i=1}^n y_i^2}}

    where :math:`y` is a tensor of target values, and :math:`x` is a tensor of predictions.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): Predicted float tensor with shape ``(N,d)``
    - ``target`` (:class:`~paddle.Tensor`): Ground truth float tensor with shape ``(N,d)``

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``cosine_similarity`` (:class:`~paddle.Tensor`): A float tensor with the cosine similarity

    Args:
        reduction: how to reduce over the batch dimension using 'sum', 'mean' or 'none' (taking the individual scores)
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from paddle import tensor
        >>> from paddlemetrics.regression import CosineSimilarity
        >>> target = tensor([[0, 1], [1, 1]])
        >>> preds = tensor([[0, 1], [0, 1]])
        >>> cosine_similarity = CosineSimilarity(reduction = 'mean')
        >>> cosine_similarity(preds, target)
        tensor(0.8536)

    """

    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    preds: List[paddle.Tensor]
    target: List[paddle.Tensor]

    def __init__(
        self, reduction: Literal["mean", "sum", "none", None] = "sum", **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        allowed_reduction = "sum", "mean", "none", None
        if reduction not in allowed_reduction:
            raise ValueError(
                f"Expected argument `reduction` to be one of {allowed_reduction} but got {reduction}"
            )
        self.reduction = reduction
        self.add_state("preds", [], dist_reduce_fx="cat")
        self.add_state("target", [], dist_reduce_fx="cat")

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update metric states with predictions and targets."""
        preds, target = _cosine_similarity_update(preds, target)
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        return _cosine_similarity_compute(preds, target, self.reduction)

    def plot(
        self,
        val: Optional[Union[paddle.Tensor, Sequence[paddle.Tensor]]] = None,
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

            >>> from paddle import randn
            >>> # Example plotting a single value
            >>> from paddlemetrics.regression import CosineSimilarity
            >>> metric = CosineSimilarity()
            >>> metric.update(randn(10,2), randn(10,2))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from paddle import randn
            >>> # Example plotting multiple values
            >>> from paddlemetrics.regression import CosineSimilarity
            >>> metric = CosineSimilarity()
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(randn(10,2), randn(10,2)))
            >>> fig, ax = metric.plot(values)

        """
        return self._plot(val, ax)
