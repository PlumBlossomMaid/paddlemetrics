from collections.abc import Sequence
from typing import Any, List, Optional, Union

import paddle

from paddlemetrics.functional.clustering import fowlkes_mallows_index
from paddlemetrics.metric import Metric
from paddlemetrics.utils.data import dim_zero_cat
from paddlemetrics.utils.imports import _MATPLOTLIB_AVAILABLE
from paddlemetrics.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["FowlkesMallowsIndex.plot"]


class FowlkesMallowsIndex(Metric):
    """Compute `Fowlkes-Mallows Index`_.

    .. math::
        FMI(U,V) = \\frac{TP}{\\sqrt{(TP + FP) * (TP + FN)}}

    Where :math:`TP` is the number of true positives, :math:`FP` is the number of false positives, and :math:`FN` is
    the number of false negatives.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): single integer tensor with shape ``(N,)`` with predicted cluster labels
    - ``target`` (:class:`~paddle.Tensor`): single integer tensor with shape ``(N,)`` with ground truth cluster labels

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``fmi`` (:class:`~paddle.Tensor`): A tensor with the Fowlkes-Mallows index.

    Args:
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example::
        >>> import paddle
        >>> from paddlemetrics.clustering import FowlkesMallowsIndex
        >>> preds = paddle.to_tensor([2, 2, 0, 1, 0])
        >>> target = paddle.to_tensor([2, 2, 1, 1, 0])
        >>> fmi = FowlkesMallowsIndex()
        >>> fmi(preds, target)
        tensor(0.5000)

    """

    is_differentiable: bool = True
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
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
        """Compute Fowlkes-Mallows index over state."""
        return fowlkes_mallows_index(
            dim_zero_cat(self.preds), dim_zero_cat(self.target)
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
            >>> from paddlemetrics.clustering import FowlkesMallowsIndex
            >>> metric = FowlkesMallowsIndex()
            >>> metric.update(paddle.randint(0, 4, (10,)), paddle.randint(0, 4, (10,)))
            >>> fig_, ax_ = metric.plot(metric.compute())

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import paddle
            >>> from paddlemetrics.clustering import FowlkesMallowsIndex
            >>> metric = FowlkesMallowsIndex()
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(paddle.randint(0, 4, (10,)), paddle.randint(0, 4, (10,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
