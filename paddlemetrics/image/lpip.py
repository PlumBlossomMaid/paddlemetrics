from collections.abc import Sequence
from typing import Any, ClassVar, Optional, Union

import paddle
from typing_extensions import Literal

from paddlemetrics.functional.image.lpips import (_LPIPS, _lpips_compute,
                                                 _lpips_update, _NoTrainLpips)
from paddlemetrics.metric import Metric
from paddlemetrics.utils import dim_zero_cat
from paddlemetrics.utils.checks import (_SKIP_SLOW_DOCTEST,
                                           _try_proceed_with_timeout)
from paddlemetrics.utils.imports import (_MATPLOTLIB_AVAILABLE,
                                            _TORCHVISION_AVAILABLE)
from paddlemetrics.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["LearnedPerceptualImagePatchSimilarity.plot"]
if _TORCHVISION_AVAILABLE:

    def _download_lpips() -> None:
        _LPIPS(pretrained=True, net="vgg")

    if _SKIP_SLOW_DOCTEST and not _try_proceed_with_timeout(_download_lpips):
        __doctest_skip__ = [
            "LearnedPerceptualImagePatchSimilarity",
            "LearnedPerceptualImagePatchSimilarity.plot",
        ]
else:
    __doctest_skip__ = [
        "LearnedPerceptualImagePatchSimilarity",
        "LearnedPerceptualImagePatchSimilarity.plot",
    ]


class LearnedPerceptualImagePatchSimilarity(Metric):
    """The Learned Perceptual Image Patch Similarity (`LPIPS_`) calculates perceptual similarity between two images.

    LPIPS essentially computes the similarity between the activations of two image patches for some pre-defined network.
    This measure has been shown to match human perception well. A low LPIPS score means that image patches are
    perceptual similar.

    Both input image patches are expected to have shape ``(N, 3, H, W)``. The minimum size of `H, W` depends on the
    chosen backbone (see `net_type` arg).

    .. hint::
        Using this metrics requires you to have ``torchvision`` package installed. Either install as
        ``pip install paddlemetrics[image]`` or ``pip install torchvision``.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``img1`` (:class:`~paddle.Tensor`): tensor with images of shape ``(N, 3, H, W)``
    - ``img2`` (:class:`~paddle.Tensor`): tensor with images of shape ``(N, 3, H, W)``

    As output of `forward` and `compute` the metric returns the following output

    - ``lpips`` (:class:`~paddle.Tensor`): returns float scalar tensor with average LPIPS value over samples

    Args:
        net_type: str indicating backbone network type to use. Choose between `'alex'`, `'vgg'` or `'squeeze'`
        reduction: str indicating how to reduce over the batch dimension. Choose between `'sum'`, `'mean'`,`'none'`
            or `None`.
        normalize: by default this is ``False`` meaning that the input is expected to be in the [-1,1] range. If set
            to ``True`` will instead expect input to be in the ``[0,1]`` range.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ModuleNotFoundError:
            If ``torchvision`` package is not installed
        ValueError:
            If ``net_type`` is not one of ``"vgg"``, ``"alex"`` or ``"squeeze"``
        ValueError:
            If ``reduction`` is not one of ``"mean"`` or ``"sum"``

    Example:
        >>> from paddle import rand
        >>> from paddlemetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        >>> lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
        >>> # LPIPS needs the images to be in the [-1, 1] range.
        >>> img1 = (rand(10, 3, 100, 100) * 2) - 1
        >>> img2 = (rand(10, 3, 100, 100) * 2) - 1
        >>> lpips(img1, img2)
        tensor(0.1024)

        >>> from paddle import rand, Generator
        >>> from paddlemetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        >>> gen = Generator().manual_seed(42)
        >>> lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze', reduction='none')
        >>> # LPIPS needs the images to be in the [-1, 1] range.
        >>> img1 = (rand(2, 3, 100, 100, generator=gen) * 2) - 1
        >>> img2 = (rand(2, 3, 100, 100, generator=gen) * 2) - 1
        >>> lpips(img1, img2)
        tensor([0.1024, 0.0938])

    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    all_scores: list[paddle.Tensor]
    feature_network: str = "net"
    __jit_ignored_attributes__: ClassVar[list[str]] = ["net"]

    def __init__(
        self,
        net_type: Literal["vgg", "alex", "squeeze"] = "alex",
        reduction: Optional[Literal["sum", "mean", "none"]] = "mean",
        normalize: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not _TORCHVISION_AVAILABLE:
            raise ModuleNotFoundError(
                "LPIPS metric requires that torchvision is installed. Either install as `pip install paddlemetrics[image]` or `pip install torchvision`."
            )
        valid_net_type = "vgg", "alex", "squeeze"
        if net_type not in valid_net_type:
            raise ValueError(
                f"Argument `net_type` must be one of {valid_net_type}, but got {net_type}."
            )
        self.net = _NoTrainLpips(net=net_type)
        valid_reduction = "mean", "sum", "none", None
        if reduction not in valid_reduction:
            raise ValueError(
                f"Argument `reduction` must be one of {valid_reduction}, but got {reduction}"
            )
        self.reduction = reduction
        if not isinstance(normalize, bool):
            raise ValueError(
                f"Argument `normalize` should be an bool but got {normalize}"
            )
        self.normalize = normalize
        self.add_state("all_scores", default=[], dist_reduce_fx=None)

    def update(self, img1: paddle.Tensor, img2: paddle.Tensor) -> None:
        """Update internal states with lpips score."""
        loss = _lpips_update(img1, img2, net=self.net, normalize=self.normalize)
        self.all_scores.append(loss)

    def compute(self) -> paddle.Tensor:
        """Compute final perceptual similarity metric."""
        scores = dim_zero_cat(self.all_scores)
        return _lpips_compute(scores, reduction=self.reduction)

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
            >>> from paddlemetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            >>> metric = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
            >>> metric.update(paddle.rand(10, 3, 100, 100), paddle.rand(10, 3, 100, 100))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import paddle
            >>> from paddlemetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            >>> metric = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
            >>> values = [ ]
            >>> for _ in range(3):
            ...     values.append(metric(paddle.rand(10, 3, 100, 100), paddle.rand(10, 3, 100, 100)))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
