from copy import deepcopy
from functools import partial
from typing import Any

import paddle
import pytest
from unittests import BATCH_SIZE, NUM_BATCHES, NUM_CLASSES
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

from paddlemetrics.classification import (BinaryAccuracy, BinaryConfusionMatrix,
                                         MulticlassAccuracy)
from paddlemetrics.regression import MeanSquaredError
from paddlemetrics.wrappers import MinMaxMetric

seed_all(42)


class TestingMinMaxMetric(MinMaxMetric):
    """Wrap metric to fit testing framework."""

    def compute(self):
        """Instead of returning dict, return as list."""
        output_dict = super().compute()
        return [output_dict["raw"], output_dict["min"], output_dict["max"]]

    def forward(self, *args: Any, **kwargs: Any):
        """Compute output for batch."""
        self.update(*args, **kwargs)
        return self.compute()


def _compare_fn(preds, target, base_fn):
    """Comparison function for minmax wrapper."""
    v_min, v_max = 1000000.0, -1000000.0
    for i in range(NUM_BATCHES):
        val = (
            base_fn(preds[: (i + 1) * BATCH_SIZE], target[: (i + 1) * BATCH_SIZE])
            .cpu()
            .numpy()
        )
        v_min = v_min if v_min < val else val
        v_max = v_max if v_max > val else val
    raw = base_fn(preds, target)
    return [raw.cpu().numpy(), v_min, v_max]


@pytest.mark.parametrize(
    ("preds", "target", "base_metric"),
    [
        (
            paddle.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES).softmax(dim=-1),
            paddle.randint(low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE)),
            MulticlassAccuracy(num_classes=NUM_CLASSES),
        ),
        (
            paddle.randn(NUM_BATCHES, BATCH_SIZE),
            paddle.randn(NUM_BATCHES, BATCH_SIZE),
            MeanSquaredError(),
        ),
    ],
)
class TestMinMaxWrapper(MetricTester):
    """Test the MinMaxMetric wrapper works as expected."""

    atol = 1e-06

    def test_minmax_wrapper(self, preds, target, base_metric):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=False,
            preds=preds,
            target=target,
            metric_class=TestingMinMaxMetric,
            reference_metric=partial(_compare_fn, base_fn=deepcopy(base_metric)),
            metric_args={"base_metric": base_metric},
            check_batch=False,
            check_scriptable=False,
        )


@pytest.mark.parametrize(
    ("preds", "labels", "raws", "maxs", "mins"),
    [
        (
            (
                [[0.9, 0.1], [0.2, 0.8]],
                [[0.1, 0.9], [0.2, 0.8]],
                [[0.1, 0.9], [0.8, 0.2]],
            ),
            [[0, 1], [0, 1]],
            (0.5, 1.0, 0.5),
            (0.5, 1.0, 1.0),
            (0.5, 0.5, 0.5),
        )
    ],
)
def test_basic_example(preds, labels, raws, maxs, mins) -> None:
    """Tests that both min and max versions of MinMaxMetric operate correctly after calling compute."""
    acc = BinaryAccuracy()
    min_max_acc = MinMaxMetric(acc)
    labels = paddle.Tensor(labels).long()
    for i in range(3):
        preds_ = paddle.Tensor(preds[i])
        min_max_acc(preds_, labels)
        acc = min_max_acc.compute()
        assert acc["raw"] == raws[i]
        assert acc["max"] == maxs[i]
        assert acc["min"] == mins[i]


def test_no_base_metric() -> None:
    """Tests that ValueError is raised when no base_metric is passed."""
    with pytest.raises(ValueError, match="Expected base metric to be an instance .*"):
        MinMaxMetric([])


def test_no_scalar_compute() -> None:
    """Tests that an assertion error is thrown if the wrapped basemetric gives a non-scalar on compute."""
    min_max_nsm = MinMaxMetric(BinaryConfusionMatrix())
    with pytest.raises(
        RuntimeError, match="Returned value from base metric should be a float.*"
    ):
        min_max_nsm.compute()
