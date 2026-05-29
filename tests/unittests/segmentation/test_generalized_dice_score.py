from functools import partial

import paddle
import pytest
from lightning_utilities.core.imports import RequirementCache
from monai.metrics.generalized_dice import compute_generalized_dice
from unittests import NUM_CLASSES
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester
from unittests.segmentation.inputs import (_index_input_1, _index_input_2,
                                           _mixed_input_1, _mixed_input_2,
                                           _mixed_logits_input,
                                           _one_hot_input_1, _one_hot_input_2)

from paddlemetrics.functional.segmentation.generalized_dice import \
    generalized_dice_score
from paddlemetrics.segmentation.generalized_dice import GeneralizedDiceScore

seed_all(42)


def _reference_generalized_dice(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    input_format: str,
    include_background: bool = True,
    reduce: bool = True,
):
    """Calculate reference metric for generalized dice metric."""
    if input_format == "index":
        preds = paddle.nn.functional.one_hot(preds, num_classes=NUM_CLASSES).moveaxis(
            -1, 1
        )
        target = paddle.nn.functional.one_hot(target, num_classes=NUM_CLASSES).moveaxis(
            -1, 1
        )
    elif input_format == "mixed":
        if preds.dim() == target.dim() + 1:
            if paddle.is_floating_point(preds):
                preds = preds.argmax(dim=1)
                preds = paddle.nn.functional.one_hot(
                    preds, num_classes=NUM_CLASSES
                ).moveaxis(-1, 1)
            target = paddle.nn.functional.one_hot(
                target, num_classes=NUM_CLASSES
            ).moveaxis(-1, 1)
        elif preds.dim() + 1 == target.dim():
            if paddle.is_floating_point(target):
                target = target.argmax(dim=1)
                target = paddle.nn.functional.one_hot(
                    target, num_classes=NUM_CLASSES
                ).moveaxis(-1, 1)
            preds = paddle.nn.functional.one_hot(
                preds, num_classes=NUM_CLASSES
            ).moveaxis(-1, 1)
    monai_extra_arg = (
        {"sum_over_classes": True} if RequirementCache("monai>=1.4.0") else {}
    )
    val = compute_generalized_dice(
        preds, target, include_background=include_background, **monai_extra_arg
    )
    if reduce:
        val = val.mean()
    return val.squeeze()


@pytest.mark.parametrize(
    ("preds", "target", "input_format"),
    [
        (_one_hot_input_1.preds, _one_hot_input_1.target, "one-hot"),
        (_one_hot_input_2.preds, _one_hot_input_2.target, "one-hot"),
        (_index_input_1.preds, _index_input_1.target, "index"),
        (_index_input_2.preds, _index_input_2.target, "index"),
        (_mixed_input_1.preds, _mixed_input_1.target, "mixed"),
        (_mixed_input_2.preds, _mixed_input_2.target, "mixed"),
        (_mixed_logits_input.preds, _mixed_logits_input.target, "mixed"),
    ],
)
@pytest.mark.parametrize("include_background", [True, False])
class TestGeneralizedDiceScore(MetricTester):
    """Test class for `GeneralizedDiceScore` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_generalized_dice_class(
        self, preds, target, input_format, include_background, ddp
    ):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=GeneralizedDiceScore,
            reference_metric=partial(
                _reference_generalized_dice,
                input_format=input_format,
                include_background=include_background,
                reduce=True,
            ),
            metric_args={
                "num_classes": NUM_CLASSES,
                "include_background": include_background,
                "input_format": input_format,
            },
        )

    def test_generalized_dice_functional(
        self, preds, target, input_format, include_background
    ):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=generalized_dice_score,
            reference_metric=partial(
                _reference_generalized_dice,
                input_format=input_format,
                include_background=include_background,
                reduce=False,
            ),
            metric_args={
                "num_classes": NUM_CLASSES,
                "include_background": include_background,
                "per_class": False,
                "input_format": input_format,
            },
        )
