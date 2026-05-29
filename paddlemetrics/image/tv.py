from collections.abc import Sequence
from typing import Any, List, Optional, Union

import paddle
from paddle import Tensor
from typing_extensions import Literal

from paddlemetrics.functional.image.tv import (_total_variation_compute,
                                              _total_variation_update)
from paddlemetrics.metric import Metric
from paddlemetrics.utils.data import dim_zero_cat
from paddlemetrics.utils.imports import _MATPLOTLIB_AVAILABLE
from paddlemetrics.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["TotalVariation.plot"]


class TotalVariation(Metric):
    """Compute Total Variation loss (`TV`_).

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``img`` (:class:`~paddle.Tensor`): A tensor of shape ``(N, C, H, W)`` consisting of images

    As output of `forward` and `compute` the metric returns the following output

    - ``sdi`` (:class:`~paddle.Tensor`): if ``reduction!='none'`` returns float scalar tensor with average TV value
      over sample else returns tensor of shape ``(N,)`` with TV values per sample

    Args:
        reduction: a method to reduce metric score over samples

            - ``'mean'``: takes the mean over samples
            - ``'sum'``: takes the sum over samples
            - ``None`` or ``'none'``: return the score per sample

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``reduction`` is not one of ``'sum'``, ``'mean'``, ``'none'`` or ``None``

    Example:
        >>> from paddle import rand
        >>> from paddlemetrics.image import TotalVariation
        >>> tv = TotalVariation()
        >>> img = paddle.rand(5, 3, 28, 28)
        >>> tv(img)
        tensor(7546.8018)

    """

    full_state_update: bool = False
    is_differentiable: bool = True
    higher_is_better: bool = False
    plot_lower_bound: float = 0.0
    num_elements: Tensor
    score_list: List[paddle.Tensor]
    score: Tensor

    def __init__(
        self, reduction: Optional[Literal["mean", "sum", "none"]] = "sum", **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if reduction is not None and reduction not in ("sum", "mean", "none"):
            raise ValueError(
                "Expected argument `reduction` to either be 'sum', 'mean', 'none' or None"
            )
        self.reduction = reduction
        self.add_state("score_list", default=[], dist_reduce_fx="cat")
        self.add_state(
            "score",
            default=paddle.tensor(0, dtype=paddle.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "num_elements",
            default=paddle.tensor(0, dtype=paddle.int32),
            dist_reduce_fx="sum",
        )

    def update(self, img: paddle.Tensor) -> None:
        """Update current score with batch of input images."""
        score, num_elements = _total_variation_update(img)
        if self.reduction is None or self.reduction == "none":
            self.score_list.append(score)
        else:
            self.score += score.sum()
        self.num_elements += num_elements

    def compute(self) -> paddle.Tensor:
        """Compute final total variation."""
        score = (
            dim_zero_cat(self.score_list)
            if self.reduction is None or self.reduction == "none"
            else self.score
        )
        return _total_variation_compute(score, self.num_elements, self.reduction)

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

            >>> # Example plotting a single value
            >>> import paddle
            >>> from paddlemetrics.image import TotalVariation
            >>> metric = TotalVariation()
            >>> metric.update(paddle.rand(5, 3, 28, 28))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import paddle
            >>> from paddlemetrics.image import TotalVariation
            >>> metric = TotalVariation()
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(paddle.rand(5, 3, 28, 28)))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
