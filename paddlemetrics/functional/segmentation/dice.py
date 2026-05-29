from typing import Optional

import paddle
from typing_extensions import Literal

from paddlemetrics.functional.segmentation.utils import \
    _segmentation_inputs_format
from paddlemetrics.utils import rank_zero_warn
from paddlemetrics.utils.compute import _safe_divide


def _dice_score_validate_args(
    num_classes: int,
    include_background: bool,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
    input_format: Literal["one-hot", "index", "mixed"] = "one-hot",
    aggregation_level: Optional[Literal["samplewise", "global"]] = "samplewise",
) -> None:
    """Validate the arguments of the metric."""
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError(
            f"Expected argument `num_classes` must be a positive integer, but got {num_classes}."
        )
    if not isinstance(include_background, bool):
        raise ValueError(
            f"Expected argument `include_background` must be a boolean, but got {include_background}."
        )
    allowed_average = ["micro", "macro", "weighted", "none"]
    if average is not None and average not in allowed_average:
        raise ValueError(
            f"Expected argument `average` to be one of {allowed_average} or None, but got {average}."
        )
    if input_format not in ["one-hot", "index", "mixed"]:
        raise ValueError(
            f"Expected argument `input_format` to be one of 'one-hot', 'index', 'mixed', but got {input_format}."
        )
    if aggregation_level not in ("samplewise", "global"):
        raise ValueError(
            f"Expected argument `aggregation_level` to be one of `samplewise`, `global`, but got {aggregation_level}"
        )


def _dice_score_update(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_classes: int,
    include_background: bool,
    input_format: Literal["one-hot", "index", "mixed"] = "one-hot",
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Update the state with the current prediction and target."""
    preds, target = _segmentation_inputs_format(
        preds, target, include_background, num_classes, input_format
    )
    reduce_axis = list(range(2, target.ndim))
    intersection = paddle.sum(preds * target, axis=reduce_axis)
    target_sum = paddle.sum(target, axis=reduce_axis)
    pred_sum = paddle.sum(preds, axis=reduce_axis)
    numerator = 2 * intersection
    denominator = pred_sum + target_sum
    support = target_sum
    return numerator, denominator, support


def _dice_score_compute(
    numerator: paddle.Tensor,
    denominator: paddle.Tensor,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
    aggregation_level: Optional[Literal["samplewise", "global"]] = "samplewise",
    support: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    """Compute the Dice score from the numerator and denominator."""
    if aggregation_level == "global":
        numerator = paddle.sum(numerator, axis=0).unsqueeze(0)
        denominator = paddle.sum(denominator, axis=0).unsqueeze(0)
        support = paddle.sum(support, axis=0) if support is not None else None
    if average == "micro":
        numerator = paddle.sum(numerator, axis=-1)
        denominator = paddle.sum(denominator, axis=-1)
        return _safe_divide(numerator, denominator, zero_division="nan")
    dice = _safe_divide(numerator, denominator, zero_division="nan")
    if average == "macro":
        return paddle.nanmean(x=dice, axis=-1)
    if average == "weighted":
        if not isinstance(support, paddle.Tensor):
            raise ValueError(
                f"Expected argument `support` to be a tensor, got: {type(support)}."
            )
        weights = _safe_divide(
            support, paddle.sum(support, axis=-1, keepdim=True), zero_division="nan"
        )
        nan_mask = dice.isnan().all(dim=-1)
        dice = paddle.nansum(x=dice * weights, axis=-1)
        dice[nan_mask] = paddle.nan
        return dice
    if average in ("none", None):
        return dice
    raise ValueError(f"Invalid value for `average`: {average}.")


def dice_score(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_classes: int,
    include_background: bool = True,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    input_format: Literal["one-hot", "index", "mixed"] = "one-hot",
    aggregation_level: Optional[Literal["samplewise", "global"]] = "samplewise",
) -> paddle.Tensor:
    """Compute the Dice score for semantic segmentation.

    Args:
        preds: Predictions from model
        target: Ground truth values
        num_classes: Number of classes
        include_background: Whether to include the background class in the computation
        average: The method to average the dice score. Options are ``"micro"``, ``"macro"``, ``"weighted"``, ``"none"``
            or ``None``. This determines how to average the dice score across different classes.
        input_format: What kind of input the function receives.
            Choose between ``"one-hot"`` for one-hot encoded tensors, ``"index"`` for index tensors
            or ``"mixed"`` for one one-hot encoded and one index tensor
        aggregation_level: The level at which to aggregate the dice score. Options are ``"samplewise"`` or ``"global"``.
            For ``"samplewise"`` the dice score is computed for each sample and then averaged. For ``"global"`` the dice
            score is computed globally over all samples.

    Returns:
        The Dice score.

    Example (with one-hot encoded tensors):
        >>> from paddle import randint
        >>> from paddlemetrics.functional.segmentation import dice_score
        >>> _ = paddle.seed(42)
        >>> preds = randint(0, 2, (4, 5, 16, 16))  # 4 samples, 5 classes, 16x16 prediction
        >>> target = randint(0, 2, (4, 5, 16, 16))  # 4 samples, 5 classes, 16x16 target
        >>> # dice score micro averaged over all classes
        >>> dice_score(preds, target, num_classes=5, average="micro")
        tensor([0.4842, 0.4968, 0.5053, 0.4902])
        >>> # dice score per sample and class
        >>> dice_score(preds, target, num_classes=5, average="none")
        tensor([[0.4724, 0.5185, 0.4710, 0.5062, 0.4500],
                [0.4571, 0.4980, 0.5191, 0.4380, 0.5649],
                [0.5428, 0.4904, 0.5358, 0.4830, 0.4724],
                [0.4715, 0.4925, 0.4797, 0.5267, 0.4788]])
        >>> # global dice score over all samples with macro averaging
        >>> dice_score(preds, target, num_classes=5, average="macro", aggregation_level="global")
        tensor([0.4942])

    Example (with index tensors):
        >>> from paddle import randint
        >>> from paddlemetrics.functional.segmentation import dice_score
        >>> _ = paddle.seed(42)
        >>> preds = randint(0, 5, (4, 16, 16))  # 4 samples, 5 classes, 16x16 prediction
        >>> target = randint(0, 5, (4, 16, 16))  # 4 samples, 5 classes, 16x16 target
        >>> # dice score micro averaged over all classes
        >>> dice_score(preds, target, num_classes=5, average="micro", input_format="index")
        tensor([0.2031, 0.1914, 0.2266, 0.1641])
        >>> # dice score per sample and class
        >>> dice_score(preds, target, num_classes=5, average="none", input_format="index")
        tensor([[0.1731, 0.1667, 0.2400, 0.2424, 0.1947],
                [0.2245, 0.2247, 0.2321, 0.1132, 0.1682],
                [0.2500, 0.2476, 0.1887, 0.1818, 0.2718],
                [0.1308, 0.1800, 0.1980, 0.1607, 0.1522]])
        >>> # global dice score over all samples with macro averaging
        >>> dice_score(preds, target, num_classes=5, average="macro", aggregation_level="global", input_format="index")
        tensor([0.1965])

    """
    if average == "micro":
        rank_zero_warn(
            "dice_score metric currently defaults to `average=micro`, but will change to`average=macro` in the v1.9 release. If you've explicitly set this parameter, you can ignore this warning.",
            UserWarning,
        )
    _dice_score_validate_args(
        num_classes, include_background, average, input_format, aggregation_level
    )
    numerator, denominator, support = _dice_score_update(
        preds, target, num_classes, include_background, input_format
    )
    return _dice_score_compute(
        numerator,
        denominator,
        average,
        aggregation_level=aggregation_level,
        support=support,
    )
