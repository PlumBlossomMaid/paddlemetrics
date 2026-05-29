from collections.abc import Sequence
from typing import Any, Optional, Union

import paddle
from paddle import Tensor

from paddlemetrics.functional.regression.symmetric_mape import (
    _symmetric_mean_absolute_percentage_error_compute,
    _symmetric_mean_absolute_percentage_error_update)
from paddlemetrics.metric import Metric
from paddlemetrics.utils.imports import _MATPLOTLIB_AVAILABLE
from paddlemetrics.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["SymmetricMeanAbsolutePercentageError.plot"]


class SymmetricMeanAbsolutePercentageError(Metric):
    """Compute symmetric mean absolute percentage error (`SMAPE`_).

    .. math:: \\text{SMAPE} = \\frac{2}{n}\\sum_1^n\\frac{|   y_i - \\hat{y_i} |}{\\max(| y_i | + | \\hat{y_i} |, \\epsilon)}

    Where :math:`y` is a tensor of target values, and :math:`\\hat{y}` is a tensor of predictions.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): Predictions from model
    - ``target`` (:class:`~paddle.Tensor`): Ground truth values

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``smape`` (:class:`~paddle.Tensor`): A tensor with non-negative floating point smape value between 0 and 2

    Args:
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from paddlemetrics.regression import SymmetricMeanAbsolutePercentageError
        >>> target = tensor([1, 10, 1e6])
        >>> preds = tensor([0.9, 15, 1.2e6])
        >>> smape = SymmetricMeanAbsolutePercentageError()
        >>> smape(preds, target)
        tensor(0.2290)

    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 2.0
    sum_abs_per_error: Tensor
    total: Tensor

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state(
            "sum_abs_per_error", default=paddle.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("total", default=paddle.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update state with predictions and targets."""
        sum_abs_per_error, num_obs = _symmetric_mean_absolute_percentage_error_update(
            preds, target
        )
        self.sum_abs_per_error += sum_abs_per_error
        self.total += num_obs

    def compute(self) -> paddle.Tensor:
        """Compute mean absolute percentage error over state."""
        return _symmetric_mean_absolute_percentage_error_compute(
            self.sum_abs_per_error, self.total
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
            >>> from paddlemetrics.regression import SymmetricMeanAbsolutePercentageError
            >>> metric = SymmetricMeanAbsolutePercentageError()
            >>> metric.update(randn(10,), randn(10,))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from paddle import randn
            >>> # Example plotting multiple values
            >>> from paddlemetrics.regression import SymmetricMeanAbsolutePercentageError
            >>> metric = SymmetricMeanAbsolutePercentageError()
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(randn(10,), randn(10,)))
            >>> fig, ax = metric.plot(values)

        """
        return self._plot(val, ax)
