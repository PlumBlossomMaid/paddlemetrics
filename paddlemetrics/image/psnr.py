from collections.abc import Sequence
from functools import partial
from typing import Any, Optional, Union

import paddle
from paddle import Tensor
from typing_extensions import Literal

from paddlemetrics.functional.image.psnr import _psnr_compute, _psnr_update
from paddlemetrics.metric import Metric
from paddlemetrics.utils import rank_zero_warn
from paddlemetrics.utils.imports import _MATPLOTLIB_AVAILABLE
from paddlemetrics.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["PeakSignalNoiseRatio.plot"]


class PeakSignalNoiseRatio(Metric):
    """`Compute Peak Signal-to-Noise Ratio`_ (PSNR).

    .. math:: \\text{PSNR}(I, J) = 10 * \\log_{10} \\left(\\frac{\\max(I)^2}{\\text{MSE}(I, J)}\\right)

    Where :math:`\\text{MSE}` denotes the `mean-squared-error`_ function.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``preds`` (:class:`~paddle.Tensor`): Predictions from model of shape ``(N,C,H,W)``
    - ``target`` (:class:`~paddle.Tensor`): Ground truth values of shape ``(N,C,H,W)``

    As output of `forward` and `compute` the metric returns the following output

    - ``psnr`` (:class:`~paddle.Tensor`): if ``reduction!='none'`` returns float scalar tensor with average PSNR value
      over sample else returns tensor of shape ``(N,)`` with PSNR values per sample

    Args:
        data_range:
            the range of the data. If a tuple is provided, then the range is calculated as the difference and
            input is clamped between the values.
        base: a base of a logarithm to use.
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

        dim:
            Dimensions to reduce PSNR scores over, provided as either an integer or a list of integers. Default is
            None meaning scores will be reduced across all dimensions and all batches.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from paddlemetrics.image import PeakSignalNoiseRatio
        >>> psnr = PeakSignalNoiseRatio(data_range=3.0)
        >>> preds = paddle.to_tensor([[0.0, 1.0], [2.0, 3.0]])
        >>> target = paddle.to_tensor([[3.0, 2.0], [1.0, 0.0]])
        >>> psnr(preds, target)
        tensor(2.5527)

    """

    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    data_range: Tensor

    def __init__(
        self,
        data_range: Union[float, tuple[float, float]],
        base: float = 10.0,
        reduction: Literal[
            "elementwise_mean", "sum", "none", None
        ] = "elementwise_mean",
        dim: Optional[Union[int, tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if dim is None and reduction != "elementwise_mean":
            rank_zero_warn(
                f"The `reduction={reduction}` will not have any effect when `dim` is None."
            )
        if dim is None:
            self.add_state(
                "sum_squared_error", default=paddle.tensor(0.0), dist_reduce_fx="sum"
            )
            self.add_state("total", default=paddle.tensor(0), dist_reduce_fx="sum")
        else:
            self.add_state("sum_squared_error", default=[], dist_reduce_fx="cat")
            self.add_state("total", default=[], dist_reduce_fx="cat")
        self.clamping_fn = None
        if isinstance(data_range, tuple):
            self.add_state(
                "data_range",
                default=paddle.tensor(data_range[1] - data_range[0]),
                dist_reduce_fx="mean",
            )
            self.clamping_fn = partial(
                paddle.clamp, min=data_range[0], max=data_range[1]
            )
        else:
            self.add_state(
                "data_range",
                default=paddle.tensor(float(data_range)),
                dist_reduce_fx="mean",
            )
        self.base = base
        self.reduction = reduction
        self.dim = tuple(dim) if isinstance(dim, Sequence) else dim

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update state with predictions and targets."""
        if self.clamping_fn is not None:
            preds = self.clamping_fn(preds)
            target = self.clamping_fn(target)
        sum_squared_error, num_obs = _psnr_update(preds, target, axis=self.dim)
        if self.dim is None:
            if not isinstance(self.sum_squared_error, paddle.Tensor):
                raise TypeError(
                    f"Expected `self.sum_squared_error` to be a Tensor, but got {type(self.sum_squared_error)}"
                )
            if not isinstance(self.total, paddle.Tensor):
                raise TypeError(
                    f"Expected `self.total` to be a Tensor, but got {type(self.total)}"
                )
            self.sum_squared_error += sum_squared_error
            self.total += num_obs
        else:
            if not isinstance(self.sum_squared_error, list):
                raise TypeError(
                    f"Expected `self.sum_squared_error` to be a list, but got {type(self.sum_squared_error)}"
                )
            if not isinstance(self.total, list):
                raise TypeError(
                    f"Expected `self.total` to be a list, but got {type(self.total)}"
                )
            self.sum_squared_error.append(sum_squared_error)
            self.total.append(num_obs)

    def compute(self) -> paddle.Tensor:
        """Compute peak signal-to-noise ratio over state."""
        if isinstance(self.sum_squared_error, paddle.Tensor):
            sum_squared_error = self.sum_squared_error
        elif isinstance(self.sum_squared_error, list):
            sum_squared_error = paddle.concat(
                [value.flatten() for value in self.sum_squared_error]
            )
        else:
            raise TypeError(
                "Expected sum_squared_error to be a Tensor or a list of Tensors"
            )
        if isinstance(self.total, paddle.Tensor):
            total = self.total
        elif isinstance(self.total, list):
            total = paddle.concat([value.flatten() for value in self.total])
        else:
            raise TypeError("Expected total to be a Tensor or a list of Tensors")
        return _psnr_compute(
            sum_squared_error,
            total,
            self.data_range,
            base=self.base,
            reduction=self.reduction,
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

            >>> # Example plotting a single value
            >>> import paddle
            >>> from paddlemetrics.image import PeakSignalNoiseRatio
            >>> metric = PeakSignalNoiseRatio(data_range=1.0)
            >>> preds = paddle.to_tensor([[0.0, 1.0], [2.0, 3.0]])
            >>> target = paddle.to_tensor([[3.0, 2.0], [1.0, 0.0]])
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import paddle
            >>> from paddlemetrics.image import PeakSignalNoiseRatio
            >>> metric = PeakSignalNoiseRatio(data_range=1.0)
            >>> preds = paddle.to_tensor([[0.0, 1.0], [2.0, 3.0]])
            >>> target = paddle.to_tensor([[3.0, 2.0], [1.0, 0.0]])
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(preds, target))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
