from collections.abc import Sequence
from typing import Any, Optional, Union

import paddle
from paddle import Tensor

from paddlemetrics.functional.regression.wmape import (
    _weighted_mean_absolute_percentage_error_compute,
    _weighted_mean_absolute_percentage_error_update)
from paddlemetrics.metric import Metric
from paddlemetrics.utils.imports import _MATPLOTLIB_AVAILABLE
from paddlemetrics.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["WeightedMeanAbsolutePercentageError.plot"]


class WeightedMeanAbsolutePercentageError(Metric):
    """Compute weighted mean absolute percentage error (`WMAPE`_).

    The output of WMAPE metric is a non-negative floating point, where the optimal value is 0. It is computes as:

    .. math::
        \\text{WMAPE} = \\frac{\\sum_{t=1}^n | y_t - \\hat{y}_t | }{\\sum_{t=1}^n |y_t| }

    Where :math:`y` is a tensor of target values, and :math:`\\hat{y}` is a tensor of predictions.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): Predictions from model
    - ``target`` (:class:`~paddle.Tensor`): Ground truth float tensor with shape ``(N,d)``

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``wmape`` (:class:`~paddle.Tensor`): A tensor with non-negative floating point wmape value between 0 and 1

    Args:
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from paddle import randn
        >>> preds = randn(20,)
        >>> target = randn(20,)
        >>> wmape = WeightedMeanAbsolutePercentageError()
        >>> wmape(preds, target)
        tensor(1.3967)

    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    sum_abs_error: Tensor
    sum_scale: Tensor

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state(
            "sum_abs_error", default=paddle.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("sum_scale", default=paddle.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update state with predictions and targets."""
        sum_abs_error, sum_scale = _weighted_mean_absolute_percentage_error_update(
            preds, target
        )
        self.sum_abs_error += sum_abs_error
        self.sum_scale += sum_scale

    def compute(self) -> paddle.Tensor:
        """Compute weighted mean absolute percentage error over state."""
        return _weighted_mean_absolute_percentage_error_compute(
            self.sum_abs_error, self.sum_scale
        )

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
            >>> from paddlemetrics.regression import WeightedMeanAbsolutePercentageError
            >>> metric = WeightedMeanAbsolutePercentageError()
            >>> metric.update(randn(10,), randn(10,))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from paddle import randn
            >>> # Example plotting multiple values
            >>> from paddlemetrics.regression import WeightedMeanAbsolutePercentageError
            >>> metric = WeightedMeanAbsolutePercentageError()
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(randn(10,), randn(10,)))
            >>> fig, ax = metric.plot(values)

        """
        return self._plot(val, ax)
