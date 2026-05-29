from collections.abc import Sequence
from typing import Any, Optional, Union

import paddle
from paddle import Tensor

from paddlemetrics.functional.audio.snr import (
    complex_scale_invariant_signal_noise_ratio,
    scale_invariant_signal_noise_ratio, signal_noise_ratio)
from paddlemetrics.metric import Metric
from paddlemetrics.utils.imports import _MATPLOTLIB_AVAILABLE
from paddlemetrics.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = [
        "SignalNoiseRatio.plot",
        "ScaleInvariantSignalNoiseRatio.plot",
        "ComplexScaleInvariantSignalNoiseRatio.plot",
    ]


class SignalNoiseRatio(Metric):
    """Calculate `Signal-to-noise ratio`_ (SNR_) meric for evaluating quality of audio.

    .. math::
        \\text{SNR} = \\frac{P_{signal}}{P_{noise}}

    where  :math:`P` denotes the power of each signal. The SNR metric compares the level of the desired signal to
    the level of background noise. Therefore, a high value of SNR means that the audio is clear.

    As input to `forward` and `update` the metric accepts the following input

    - ``preds`` (:class:`~paddle.Tensor`): float tensor with shape ``(...,time)``
    - ``target`` (:class:`~paddle.Tensor`): float tensor with shape ``(...,time)``

    As output of `forward` and `compute` the metric returns the following output

    - ``snr`` (:class:`~paddle.Tensor`): float scalar tensor with average SNR value over samples

    Args:
        zero_mean: if to zero mean target and preds or not
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        TypeError:
            if target and preds have a different shape

    Example:
        >>> from paddle import tensor
        >>> from paddlemetrics.audio import SignalNoiseRatio
        >>> target = tensor([3.0, -0.5, 2.0, 7.0])
        >>> preds = tensor([2.5, 0.0, 2.0, 8.0])
        >>> snr = SignalNoiseRatio()
        >>> snr(preds, target)
        tensor(16.1805)

    """

    full_state_update: bool = False
    is_differentiable: bool = True
    higher_is_better: bool = True
    sum_snr: Tensor
    total: Tensor
    plot_lower_bound: Optional[float] = None
    plot_upper_bound: Optional[float] = None

    def __init__(self, zero_mean: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.zero_mean = zero_mean
        self.add_state("sum_snr", default=paddle.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=paddle.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update state with predictions and targets."""
        snr_batch = signal_noise_ratio(
            preds=preds, target=target, zero_mean=self.zero_mean
        )
        self.sum_snr += snr_batch.sum()
        self.total += snr_batch.size

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        return self.sum_snr / self.total

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
            >>> from paddlemetrics.audio import SignalNoiseRatio
            >>> metric = SignalNoiseRatio()
            >>> metric.update(paddle.rand(4), paddle.rand(4))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import paddle
            >>> from paddlemetrics.audio import SignalNoiseRatio
            >>> metric = SignalNoiseRatio()
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(paddle.rand(4), paddle.rand(4)))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class ScaleInvariantSignalNoiseRatio(Metric):
    """Calculate `Scale-invariant signal-to-noise ratio`_ (SI-SNR) metric for evaluating quality of audio.

    As input to `forward` and `update` the metric accepts the following input

    - ``preds`` (:class:`~paddle.Tensor`): float tensor with shape ``(...,time)``
    - ``target`` (:class:`~paddle.Tensor`): float tensor with shape ``(...,time)``

    As output of `forward` and `compute` the metric returns the following output

    - ``si_snr`` (:class:`~paddle.Tensor`): float scalar tensor with average SI-SNR value over samples

    Args:
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        TypeError:
            if target and preds have a different shape

    Example:
        >>> import paddle
        >>> from paddle import tensor
        >>> from paddlemetrics.audio import ScaleInvariantSignalNoiseRatio
        >>> target = tensor([3.0, -0.5, 2.0, 7.0])
        >>> preds = tensor([2.5, 0.0, 2.0, 8.0])
        >>> si_snr = ScaleInvariantSignalNoiseRatio()
        >>> si_snr(preds, target)
        tensor(15.0918)

    """

    is_differentiable = True
    sum_si_snr: Tensor
    total: Tensor
    higher_is_better = True
    plot_lower_bound: Optional[float] = None
    plot_upper_bound: Optional[float] = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("sum_si_snr", default=paddle.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=paddle.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update state with predictions and targets."""
        si_snr_batch = scale_invariant_signal_noise_ratio(preds=preds, target=target)
        self.sum_si_snr += si_snr_batch.sum()
        self.total += si_snr_batch.size

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        return self.sum_si_snr / self.total

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
            >>> from paddlemetrics.audio import ScaleInvariantSignalNoiseRatio
            >>> metric = ScaleInvariantSignalNoiseRatio()
            >>> metric.update(paddle.rand(4), paddle.rand(4))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import paddle
            >>> from paddlemetrics.audio import ScaleInvariantSignalNoiseRatio
            >>> metric = ScaleInvariantSignalNoiseRatio()
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(paddle.rand(4), paddle.rand(4)))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class ComplexScaleInvariantSignalNoiseRatio(Metric):
    """Calculate `Complex scale-invariant signal-to-noise ratio`_ (C-SI-SNR) metric for evaluating quality of audio.

    As input to `forward` and `update` the metric accepts the following input

    - ``preds`` (:class:`~paddle.Tensor`): real float tensor with shape ``(...,frequency,time,2)`` or complex float
      tensor with shape ``(..., frequency,time)``

    - ``target`` (:class:`~paddle.Tensor`): real float tensor with shape ``(...,frequency,time,2)`` or complex float
      tensor with shape ``(..., frequency,time)``

    As output of `forward` and `compute` the metric returns the following output

    - ``c_si_snr`` (:class:`~paddle.Tensor`): float scalar tensor with average C-SI-SNR value over samples

    Args:
        zero_mean: if to zero mean target and preds or not
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``zero_mean`` is not an bool
        TypeError:
            If ``preds`` is not the shape (..., frequency, time, 2) (after being converted to real if it is complex).
            If ``preds`` and ``target`` does not have the same shape.

    Example:
        >>> from paddle import randn
        >>> from paddlemetrics.audio import ComplexScaleInvariantSignalNoiseRatio
        >>> preds = randn((1,257,100,2))
        >>> target = randn((1,257,100,2))
        >>> c_si_snr = ComplexScaleInvariantSignalNoiseRatio()
        >>> c_si_snr(preds, target)
        tensor(-38.8832)

    """

    is_differentiable = True
    ci_snr_sum: Tensor
    num: Tensor
    higher_is_better = True
    plot_lower_bound: Optional[float] = None
    plot_upper_bound: Optional[float] = None

    def __init__(self, zero_mean: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not isinstance(zero_mean, bool):
            raise ValueError(
                f"Expected argument `zero_mean` to be an bool, but got {zero_mean}"
            )
        self.zero_mean = zero_mean
        self.add_state("ci_snr_sum", default=paddle.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num", default=paddle.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update state with predictions and targets."""
        v = complex_scale_invariant_signal_noise_ratio(
            preds=preds, target=target, zero_mean=self.zero_mean
        )
        self.ci_snr_sum += v.sum()
        self.num += v.size

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        return self.ci_snr_sum / self.num

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
            >>> from paddlemetrics.audio import ComplexScaleInvariantSignalNoiseRatio
            >>> metric = ComplexScaleInvariantSignalNoiseRatio()
            >>> metric.update(paddle.rand(1,257,100,2), paddle.rand(1,257,100,2))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import paddle
            >>> from paddlemetrics.audio import ComplexScaleInvariantSignalNoiseRatio
            >>> metric = ComplexScaleInvariantSignalNoiseRatio()
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(paddle.rand(1,257,100,2), paddle.rand(1,257,100,2)))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
