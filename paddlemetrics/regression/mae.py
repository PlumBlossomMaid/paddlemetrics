from collections.abc import Sequence
from typing import Any, Optional, Union

import paddle
from paddle import Tensor

from paddlemetrics.functional.regression.mae import (
    _mean_absolute_error_compute, _mean_absolute_error_update)
from paddlemetrics.metric import Metric
from paddlemetrics.utils.imports import _MATPLOTLIB_AVAILABLE
from paddlemetrics.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["MeanAbsoluteError.plot"]


class MeanAbsoluteError(Metric):
    """`Compute Mean Absolute Error`_ (MAE).

    .. math:: \\text{MAE} = \\frac{1}{N}\\sum_i^N | y_i - \\hat{y_i} |

    Where :math:`y` is a tensor of target values, and :math:`\\hat{y}` is a tensor of predictions.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): Predictions from model
    - ``target`` (:class:`~paddle.Tensor`): Ground truth values

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``mean_absolute_error`` (:class:`~paddle.Tensor`): A tensor with the mean absolute error over the state

    Args:
        num_outputs: Number of outputs in multioutput setting
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from paddle import tensor
        >>> from paddlemetrics.regression import MeanAbsoluteError
        >>> target = tensor([3.0, -0.5, 2.0, 7.0])
        >>> preds = tensor([2.5, 0.0, 2.0, 8.0])
        >>> mean_absolute_error = MeanAbsoluteError()
        >>> mean_absolute_error(preds, target)
        tensor(0.5000)

    Example::
        Multioutput mse computation:

        >>> from paddle import tensor
        >>> from paddlemetrics.regression import MeanAbsoluteError
        >>> target = tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        >>> preds = tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        >>> mean_absolute_error = MeanAbsoluteError(num_outputs=3)
        >>> mean_absolute_error(preds, target)
        tensor([1., 2., 3.])

    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    sum_abs_error: Tensor
    total: Tensor

    def __init__(self, num_outputs: int = 1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not (isinstance(num_outputs, int) and num_outputs > 0):
            raise ValueError(
                f"Expected num_outputs to be a positive integer but got {num_outputs}"
            )
        self.num_outputs = num_outputs
        self.add_state(
            "sum_abs_error", default=paddle.zeros(num_outputs), dist_reduce_fx="sum"
        )
        self.add_state("total", default=paddle.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update state with predictions and targets."""
        sum_abs_error, num_obs = _mean_absolute_error_update(
            preds, target, num_outputs=self.num_outputs
        )
        self.sum_abs_error += sum_abs_error
        self.total += num_obs

    def compute(self) -> paddle.Tensor:
        """Compute mean absolute error over state."""
        return _mean_absolute_error_compute(self.sum_abs_error, self.total)

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
            >>> from paddlemetrics.regression import MeanAbsoluteError
            >>> metric = MeanAbsoluteError()
            >>> metric.update(randn(10,), randn(10,))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from paddle import randn
            >>> # Example plotting multiple values
            >>> from paddlemetrics.regression import MeanAbsoluteError
            >>> metric = MeanAbsoluteError()
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(randn(10,), randn(10,)))
            >>> fig, ax = metric.plot(values)

        """
        return self._plot(val, ax)
