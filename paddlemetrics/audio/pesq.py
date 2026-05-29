from collections.abc import Sequence
from typing import Any, Optional, Union

import paddle
from paddle import Tensor

from paddlemetrics.functional.audio.pesq import \
    perceptual_evaluation_speech_quality
from paddlemetrics.metric import Metric
from paddlemetrics.utils.imports import (_MATPLOTLIB_AVAILABLE,
                                            _PESQ_AVAILABLE)
from paddlemetrics.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

__doctest_requires__ = {"PerceptualEvaluationSpeechQuality": ["pesq"]}
if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["PerceptualEvaluationSpeechQuality.plot"]


class PerceptualEvaluationSpeechQuality(Metric):
    """Calculate `Perceptual Evaluation of Speech Quality`_ (PESQ).

    It's a recognized industry standard for audio quality that takes into considerations characteristics such as:
    audio sharpness, call volume, background noise, clipping, audio interference etc. PESQ returns a score between
    -0.5 and 4.5 with the higher scores indicating a better quality.

    This metric is a wrapper for the `pesq package`_. Note that input will be moved to ``cpu`` to perform the metric
    calculation.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``preds`` (:class:`~paddle.Tensor`): float tensor with shape ``(...,time)``
    - ``target`` (:class:`~paddle.Tensor`): float tensor with shape ``(...,time)``

    As output of `forward` and `compute` the metric returns the following output

    - ``pesq`` (:class:`~paddle.Tensor`): float tensor of PESQ value reduced across the batch

    .. hint::
        Using this metrics requires you to have ``pesq`` install. Either install as ``pip install
        paddlemetrics[audio]`` or ``pip install pesq``. ``pesq`` will compile with your currently
        installed version of numpy, meaning that if you upgrade numpy at some point in the future you will
        most likely have to reinstall ``pesq``.

    .. caution::
        The ``forward`` and ``compute`` methods in this class return a single (reduced) PESQ value
        for a batch. To obtain a PESQ value for each sample, you may use the functional counterpart in
        :func:`~paddlemetrics.functional.audio.pesq.perceptual_evaluation_speech_quality`.

    Args:
        fs: sampling frequency, should be 16000 or 8000 (Hz)
        mode: ``'wb'`` (wide-band) or ``'nb'`` (narrow-band)
        keep_same_device: whether to move the pesq value to the device of preds
        n_processes: integer specifying the number of processes to run in parallel for the metric calculation.
            Only applies to batches of data and if ``multiprocessing`` package is installed.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ModuleNotFoundError:
            If ``pesq`` package is not installed
        ValueError:
            If ``fs`` is not either  ``8000`` or ``16000``
        ValueError:
            If ``mode`` is not either ``"wb"`` or ``"nb"``

    Example:
        >>> from paddle import randn
        >>> from paddlemetrics.audio import PerceptualEvaluationSpeechQuality
        >>> preds = randn(8000)
        >>> target = randn(8000)
        >>> pesq = PerceptualEvaluationSpeechQuality(8000, 'nb')
        >>> pesq(preds, target)
        tensor(2.2885)
        >>> wb_pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')
        >>> wb_pesq(preds, target)
        tensor(1.6805)

    """

    sum_pesq: Tensor
    total: Tensor
    full_state_update: bool = False
    is_differentiable: bool = False
    higher_is_better: bool = True
    plot_lower_bound: float = -0.5
    plot_upper_bound: float = 4.5

    def __init__(self, fs: int, mode: str, n_processes: int = 1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not _PESQ_AVAILABLE:
            raise ModuleNotFoundError(
                "PerceptualEvaluationSpeechQuality metric requires that `pesq` is installed. Either install as `pip install paddlemetrics[audio]` or `pip install pesq`."
            )
        if fs not in (8000, 16000):
            raise ValueError(
                f"Expected argument `fs` to either be 8000 or 16000 but got {fs}"
            )
        self.fs = fs
        if mode not in ("wb", "nb"):
            raise ValueError(
                f"Expected argument `mode` to either be 'wb' or 'nb' but got {mode}"
            )
        self.mode = mode
        if not isinstance(n_processes, int) and n_processes <= 0:
            raise ValueError(
                f"Expected argument `n_processes` to be an int larger than 0 but got {n_processes}"
            )
        self.n_processes = n_processes
        self.add_state("sum_pesq", default=paddle.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=paddle.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update state with predictions and targets."""
        pesq_batch = perceptual_evaluation_speech_quality(
            preds, target, self.fs, self.mode, False, self.n_processes
        ).to(self.sum_pesq.place)
        self.sum_pesq += pesq_batch.sum()
        self.total += pesq_batch.size

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        return self.sum_pesq / self.total

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
            >>> from paddlemetrics.audio import PerceptualEvaluationSpeechQuality
            >>> metric = PerceptualEvaluationSpeechQuality(8000, 'nb')
            >>> metric.update(paddle.rand(8000), paddle.rand(8000))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import paddle
            >>> from paddlemetrics.audio import PerceptualEvaluationSpeechQuality
            >>> metric = PerceptualEvaluationSpeechQuality(8000, 'nb')
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(paddle.rand(8000), paddle.rand(8000)))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
