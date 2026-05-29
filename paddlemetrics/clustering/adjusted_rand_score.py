from collections.abc import Sequence
from typing import Any, List, Optional, Union

import paddle

from paddlemetrics.functional.clustering.adjusted_rand_score import \
    adjusted_rand_score
from paddlemetrics.metric import Metric
from paddlemetrics.utils.data import dim_zero_cat
from paddlemetrics.utils.imports import _MATPLOTLIB_AVAILABLE
from paddlemetrics.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["AdjustedRandScore.plot"]


class AdjustedRandScore(Metric):
    """Compute `Adjusted Rand Score`_ (also known as Adjusted Rand Index).

    .. math::
        ARS(U, V) = (\\text{RS} - \\text{Expected RS}) / (\\text{Max RS} - \\text{Expected RS})

    The adjusted rand score :math:`\\text{ARS}` is in essence the :math:`\\text{RS}` (rand score) adjusted for chance.
    The score ensures that completely randomly cluster labels have a score close to zero and only a perfect match will
    have a score of 1 (up to a permutation of the labels). The adjusted rand score is symmetric, therefore swapping
    :math:`U` and :math:`V` yields the same adjusted rand score.

    This clustering metric is an extrinsic measure, because it requires ground truth clustering labels, which may not
    be available in practice since clustering is generally used for unsupervised learning.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): single integer tensor with shape ``(N,)`` with predicted cluster labels
    - ``target`` (:class:`~paddle.Tensor`): single integer tensor with shape ``(N,)`` with ground truth cluster labels

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``adj_rand_score`` (:class:`~paddle.Tensor`): Scalar tensor with the adjusted rand score

    Args:
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example::
        >>> import paddle
        >>> from paddlemetrics.clustering import AdjustedRandScore
        >>> metric = AdjustedRandScore()
        >>> metric(paddle.to_tensor([0, 0, 1, 1]), paddle.to_tensor([0, 0, 1, 1]))
        tensor(1.)
        >>> metric(paddle.to_tensor([0, 0, 1, 1]), paddle.to_tensor([0, 1, 0, 1]))
        tensor(-0.5000)

    """

    is_differentiable = True
    higher_is_better = None
    full_state_update: bool = False
    plot_lower_bound: float = -0.5
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
        """Compute mutual information over state."""
        return adjusted_rand_score(dim_zero_cat(self.preds), dim_zero_cat(self.target))

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
            >>> from paddlemetrics.clustering import AdjustedRandScore
            >>> metric = AdjustedRandScore()
            >>> metric.update(paddle.randint(0, 4, (10,)), paddle.randint(0, 4, (10,)))
            >>> fig_, ax_ = metric.plot(metric.compute())

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import paddle
            >>> from paddlemetrics.clustering import AdjustedRandScore
            >>> metric = AdjustedRandScore()
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(paddle.randint(0, 4, (10,)), paddle.randint(0, 4, (10,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
