from functools import partial
from typing import Any

import paddle
import pytest
from monai.metrics.hausdorff_distance import \
    compute_hausdorff_distance as monai_hausdorff_distance
from unittests import NUM_BATCHES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

from paddlemetrics.functional.segmentation.hausdorff_distance import \
    hausdorff_distance
from paddlemetrics.segmentation.hausdorff_distance import HausdorffDistance

seed_all(42)
BATCH_SIZE = 4
NUM_CLASSES = 3
_inputs1 = _Input(
    preds=paddle.randint(
        low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, 16, 16)
    ),
    target=paddle.randint(
        low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, 16, 16)
    ),
)
_inputs2 = _Input(
    preds=paddle.randint(
        low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE, 32, 32)
    ),
    target=paddle.randint(
        low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE, 32, 32)
    ),
)
_inputs3 = _Input(
    preds=paddle.randint(
        low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE, 16, 16)
    ),
    target=paddle.randint(
        low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, 16, 16)
    ),
)
_inputs4 = _Input(
    preds=paddle.randint(
        low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, 16, 16)
    ),
    target=paddle.randint(
        low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE, 16, 16)
    ),
)
_inputs5 = _Input(
    preds=paddle.rand((NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, 16, 16)) * 12 - 6,
    target=paddle.randint(
        low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE, 16, 16)
    ),
)


def reference_metric(preds, target, input_format, reduce, **kwargs: Any):
    """Reference implementation of metric."""
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
    score = monai_hausdorff_distance(preds, target, **kwargs)
    return score.mean() if reduce else score


@pytest.mark.parametrize(
    ("inputs", "input_format"),
    [
        (_inputs1, "one-hot"),
        (_inputs2, "index"),
        (_inputs3, "mixed"),
        (_inputs4, "mixed"),
        (_inputs5, "mixed"),
    ],
)
@pytest.mark.parametrize("distance_metric", ["euclidean", "chessboard", "taxicab"])
@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("spacing", [None, [2, 2]])
class TestHausdorffDistance(MetricTester):
    """Test class for `HausdorffDistance` metric."""

    atol = 1e-05

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_hausdorff_distance_class(
        self, inputs, input_format, distance_metric, directed, spacing, ddp
    ):
        """Test class implementation of metric."""
        if spacing is not None and distance_metric != "euclidean":
            pytest.skip("Spacing is only supported for Euclidean distance metric.")
        preds, target = inputs
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=HausdorffDistance,
            reference_metric=partial(
                reference_metric,
                input_format=input_format,
                distance_metric=distance_metric,
                directed=directed,
                spacing=spacing,
                reduce=True,
            ),
            metric_args={
                "num_classes": NUM_CLASSES,
                "distance_metric": distance_metric,
                "directed": directed,
                "spacing": spacing,
                "input_format": input_format,
            },
        )

    def test_hausdorff_distance_functional(
        self, inputs, input_format, distance_metric, directed, spacing
    ):
        """Test functional implementation of metric."""
        if spacing is not None and distance_metric != "euclidean":
            pytest.skip("Spacing is only supported for Euclidean distance metric.")
        preds, target = inputs
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=hausdorff_distance,
            reference_metric=partial(
                reference_metric,
                input_format=input_format,
                distance_metric=distance_metric,
                directed=directed,
                spacing=spacing,
                reduce=False,
            ),
            metric_args={
                "num_classes": NUM_CLASSES,
                "distance_metric": distance_metric,
                "directed": directed,
                "spacing": spacing,
                "input_format": input_format,
            },
        )


def test_hausdorff_distance_raises_error():
    """Check that metric raises appropriate errors."""
    preds, target = _inputs1
