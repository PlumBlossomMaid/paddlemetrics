from collections.abc import Sequence
from typing import Any, Optional, Union

import paddle
from paddle import Tensor
from typing_extensions import Literal

from paddlemetrics.functional.nominal.tschuprows import (_tschuprows_t_compute,
                                                        _tschuprows_t_update)
from paddlemetrics.functional.nominal.utils import _nominal_input_validation
from paddlemetrics.metric import Metric
from paddlemetrics.utils.imports import _MATPLOTLIB_AVAILABLE
from paddlemetrics.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["TschuprowsT.plot"]


class TschuprowsT(Metric):
    """Compute `Tschuprow's T`_ statistic measuring the association between two categorical (nominal) data series.

    .. math::
        T = \\sqrt{\\frac{\\chi^2 / n}{\\sqrt{(r - 1) * (k - 1)}}}

    where

    .. math::
        \\chi^2 = \\sum_{i,j} \\ frac{\\left(n_{ij} - \\frac{n_{i.} n_{.j}}{n}\\right)^2}{\\frac{n_{i.} n_{.j}}{n}}

    where :math:`n_{ij}` denotes the number of times the values :math:`(A_i, B_j)` are observed with :math:`A_i, B_j`
    represent frequencies of values in ``preds`` and ``target``, respectively. Tschuprow's T is a symmetric coefficient,
    i.e. :math:`T(preds, target) = T(target, preds)`, so order of input arguments does not matter. The output values
    lies in [0, 1] with 1 meaning the perfect association.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): Either 1D or 2D tensor of categorical (nominal) data from the first data
      series with shape ``(batch_size,)`` or ``(batch_size, num_classes)``, respectively.
    - ``target`` (:class:`~paddle.Tensor`): Either 1D or 2D tensor of categorical (nominal) data from the second data
      series with shape ``(batch_size,)`` or ``(batch_size, num_classes)``, respectively.

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``tschuprows_t`` (:class:`~paddle.Tensor`): Scalar tensor containing the Tschuprow's T statistic.

    Args:
        num_classes: Integer specifying the number of classes
        bias_correction: Indication of whether to use bias correction.
        nan_strategy: Indication of whether to replace or drop ``NaN`` values
        nan_replace_value: Value to replace ``NaN``s when ``nan_strategy = 'replace'``
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If `nan_strategy` is not one of `'replace'` and `'drop'`
        ValueError:
            If `nan_strategy` is equal to `'replace'` and `nan_replace_value` is not an `int` or `float`

    Example::

        >>> from paddle import randint
        >>> from paddlemetrics.nominal import TschuprowsT
        >>> preds = randint(0, 4, (100,))
        >>> target = (preds + paddle.randn(100)).round().clamp(0, 4)
        >>> tschuprows_t = TschuprowsT(num_classes=5)
        >>> tschuprows_t(preds, target)
        tensor(0.4930)

    """

    full_state_update: bool = False
    is_differentiable: bool = False
    higher_is_better: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    confmat: Tensor

    def __init__(
        self,
        num_classes: int,
        bias_correction: bool = True,
        nan_strategy: Literal["replace", "drop"] = "replace",
        nan_replace_value: Optional[float] = 0.0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.bias_correction = bias_correction
        _nominal_input_validation(nan_strategy, nan_replace_value)
        self.nan_strategy = nan_strategy
        self.nan_replace_value = nan_replace_value
        self.add_state(
            "confmat", paddle.zeros(num_classes, num_classes), dist_reduce_fx="sum"
        )

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update state with predictions and targets."""
        confmat = _tschuprows_t_update(
            preds, target, self.num_classes, self.nan_strategy, self.nan_replace_value
        )
        self.confmat += confmat

    def compute(self) -> paddle.Tensor:
        """Compute Tschuprow's T statistic."""
        return _tschuprows_t_compute(self.confmat, self.bias_correction)

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
            >>> from paddlemetrics.nominal import TschuprowsT
            >>> metric = TschuprowsT(num_classes=5)
            >>> metric.update(paddle.randint(0, 4, (100,)), paddle.randint(0, 4, (100,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import paddle
            >>> from paddlemetrics.nominal import TschuprowsT
            >>> metric = TschuprowsT(num_classes=5)
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(paddle.randint(0, 4, (100,)), paddle.randint(0, 4, (100,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
