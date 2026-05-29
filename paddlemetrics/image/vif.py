from typing import Any, List

import paddle
from paddle import Tensor
from typing_extensions import Literal

from paddlemetrics.functional.image.vif import _vif_per_channel
from paddlemetrics.metric import Metric
from paddlemetrics.utils.data import dim_zero_cat


class VisualInformationFidelity(Metric):
    """Compute Pixel Based Visual Information Fidelity (VIF_).

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``preds`` (:class:`~paddle.Tensor`): Predictions from model of shape ``(N,C,H,W)`` with H,W ≥ 41
    - ``target`` (:class:`~paddle.Tensor`): Ground truth values of shape ``(N,C,H,W)`` with H,W ≥ 41

    As output of `forward` and `compute` the metric returns the following output

    - ``vif-p`` (:class:`~paddle.Tensor`):
        - If ``reduction='mean'`` (default), returns a Tensor mean VIF score.
        - If ``reduction='none'``, returns a tensor of shape ``(N,)`` with VIF values per sample.

    Args:
        sigma_n_sq: variance of the visual noise
        reduction: The reduction method for aggregating scores.

            - ``'mean'``: return the average VIF across the batch.
            - ``'none'``: return a VIF score for each sample in the batch.

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from paddle import randn
        >>> from paddlemetrics.image import VisualInformationFidelity
        >>> preds = randn([32, 3, 41, 41], generator=paddle.Generator().manual_seed(42))
        >>> target = randn([32, 3, 41, 41], generator=paddle.Generator().manual_seed(43))
        >>> vif_mean = VisualInformationFidelity(reduction='mean')
        >>> vif_mean(preds, target)
        tensor(0.0032)
        >>> vif_none = VisualInformationFidelity(reduction='none')
        >>> vif_none(preds, target)
        tensor([0.0040, 0.0049, 0.0017, 0.0039, 0.0041, 0.0043, 0.0030, 0.0028, 0.0012,
                0.0067, 0.0010, 0.0014, 0.0030, 0.0048, 0.0050, 0.0038, 0.0037, 0.0025,
                0.0041, 0.0019, 0.0007, 0.0034, 0.0037, 0.0016, 0.0026, 0.0021, 0.0038,
                0.0033, 0.0031, 0.0020, 0.0036, 0.0057])

    """

    is_differentiable = True
    higher_is_better = True
    full_state_update = False
    vif_score: List[paddle.Tensor]
    total: Tensor

    def __init__(
        self,
        sigma_n_sq: float = 2.0,
        reduction: Literal["mean", "none"] = "mean",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(sigma_n_sq, (float, int)) or sigma_n_sq < 0:
            raise ValueError(
                f"Argument `sigma_n_sq` is expected to be a positive float or int, but got {sigma_n_sq}"
            )
        if reduction not in ("mean", "none"):
            raise ValueError(
                f"Argument `reduction` must be 'mean' or 'none', but got {reduction}"
            )
        self.sigma_n_sq = sigma_n_sq
        self.reduction = reduction
        self.add_state("vif_score", default=[], dist_reduce_fx=None)

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update state with predictions and targets."""
        channels = preds.size(1)
        vif_per_channel = [
            _vif_per_channel(preds[:, i, :, :], target[:, i, :, :], self.sigma_n_sq)
            for i in range(channels)
        ]
        vif_per_channel = (
            paddle.mean(paddle.stack(vif_per_channel), 0)
            if channels > 1
            else paddle.concat(vif_per_channel)
        )
        self.vif_score.append(vif_per_channel)

    def compute(self) -> paddle.Tensor:
        """Compute VIF over state."""
        vif_score = dim_zero_cat(self.vif_score)
        if self.reduction == "mean":
            return vif_score.mean()
        return vif_score
