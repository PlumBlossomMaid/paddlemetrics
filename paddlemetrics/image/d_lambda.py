from collections.abc import Sequence
from typing import Any, List, Optional, Union

import paddle
from typing_extensions import Literal

from paddlemetrics.functional.image.d_lambda import (
    _spectral_distortion_index_compute, _spectral_distortion_index_update)
from paddlemetrics.metric import Metric
from paddlemetrics.utils import rank_zero_warn
from paddlemetrics.utils.data import dim_zero_cat
from paddlemetrics.utils.imports import _MATPLOTLIB_AVAILABLE
from paddlemetrics.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["SpectralDistortionIndex.plot"]


class SpectralDistortionIndex(Metric):
    """Compute Spectral Distortion Index (SpectralDistortionIndex_) also now as D_lambda.

    The metric is used to compare the spectral distortion between two images.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``preds`` (:class:`~paddle.Tensor`): Low resolution multispectral image of shape ``(N,C,H,W)``
    - ``target``(:class:`~paddle.Tensor`): High resolution fused image of shape ``(N,C,H,W)``

    As output of `forward` and `compute` the metric returns the following output

    - ``sdi`` (:class:`~paddle.Tensor`): if ``reduction!='none'`` returns float scalar tensor with average SDI value
      over sample else returns tensor of shape ``(N,)`` with SDI values per sample

    Args:
        p: Large spectral differences
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from paddle import rand
        >>> from paddlemetrics.image import SpectralDistortionIndex
        >>> preds = rand([16, 3, 16, 16])
        >>> target = rand([16, 3, 16, 16])
        >>> sdi = SpectralDistortionIndex()
        >>> sdi(preds, target)
        tensor(0.0234)

    """

    higher_is_better: bool = True
    is_differentiable: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    preds: List[paddle.Tensor]
    target: List[paddle.Tensor]

    def __init__(
        self,
        p: int = 1,
        reduction: Literal["elementwise_mean", "sum", "none"] = "elementwise_mean",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        rank_zero_warn(
            "Metric `SpectralDistortionIndex` will save all targets and predictions in buffer. For large datasets this may lead to large memory footprint."
        )
        if not isinstance(p, int) or p <= 0:
            raise ValueError(f"Expected `p` to be a positive integer. Got p: {p}.")
        self.p = p
        allowed_reductions = "elementwise_mean", "sum", "none"
        if reduction not in allowed_reductions:
            raise ValueError(
                f"Expected argument `reduction` be one of {allowed_reductions} but got {reduction}"
            )
        self.reduction = reduction
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update state with preds and target."""
        preds, target = _spectral_distortion_index_update(preds, target)
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> paddle.Tensor:
        """Compute and returns spectral distortion index."""
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        return _spectral_distortion_index_compute(preds, target, self.p, self.reduction)

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
            >>> from paddle import rand
            >>> from paddlemetrics.image import SpectralDistortionIndex
            >>> preds = rand([16, 3, 16, 16])
            >>> target = rand([16, 3, 16, 16])
            >>> metric = SpectralDistortionIndex()
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> from paddle import rand
            >>> from paddlemetrics.image import SpectralDistortionIndex
            >>> preds = rand([16, 3, 16, 16])
            >>> target = rand([16, 3, 16, 16])
            >>> metric = SpectralDistortionIndex()
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(preds, target))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
