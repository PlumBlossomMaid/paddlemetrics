from typing import Any

import numpy as np
import paddle
import pytest
from unittests import _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

from paddlemetrics.detection.panoptic_qualities import PanopticQuality
from paddlemetrics.functional.detection.panoptic_qualities import \
    panoptic_quality

seed_all(42)
_INPUTS_0 = _Input(
    preds=paddle.tensor(
        [
            [[6, 0], [0, 0], [6, 0], [6, 0], [0, 1]],
            [[0, 0], [0, 0], [6, 0], [0, 1], [0, 1]],
            [[0, 0], [0, 0], [6, 0], [0, 1], [1, 0]],
            [[0, 0], [7, 0], [6, 0], [1, 0], [1, 0]],
            [[0, 0], [7, 0], [7, 0], [7, 0], [7, 0]],
        ]
    )
    .reshape((1, 1, 5, 5, 2))
    .repeat(2, 1, 1, 1, 1),
    target=paddle.tensor(
        [
            [[6, 0], [6, 0], [6, 0], [6, 0], [0, 0]],
            [[0, 1], [0, 1], [6, 0], [0, 0], [0, 0]],
            [[0, 1], [0, 1], [6, 0], [1, 0], [1, 0]],
            [[0, 1], [7, 0], [7, 0], [1, 0], [1, 0]],
            [[0, 1], [7, 0], [7, 0], [7, 0], [7, 0]],
        ]
    )
    .reshape((1, 1, 5, 5, 2))
    .repeat(2, 1, 1, 1, 1),
)
_INPUTS_1 = _Input(
    preds=paddle.tensor([[10, 0], [10, 123], [0, 1], [10, 0], [1, 2]])
    .reshape((1, 1, 5, 2))
    .repeat(2, 1, 1, 1),
    target=paddle.tensor([[10, 0], [10, 0], [0, 0], [0, 1], [1, 0]])
    .reshape((1, 1, 5, 2))
    .repeat(2, 1, 1, 1),
)
_ARGS_0 = {"things": {0, 1}, "stuffs": {6, 7}}
_ARGS_1 = {"things": {2}, "stuffs": {3}, "allow_unknown_preds_category": True}
_ARGS_2 = {"things": {0, 1}, "stuffs": {10, 11}}


def _get_class_order_test_input_args(
    class_type, class1, class2, class3
) -> (np.ndarray, dict):
    a = [class1, 0]
    b = [class2, 0]
    c = [class3, 0]
    _input = _Input(
        preds=paddle.tensor([a, a, b, b, b, c])
        .reshape((1, 1, 6, 2))
        .repeat(2, 1, 1, 1),
        target=paddle.tensor([a, a, b, b, c, c])
        .reshape((1, 1, 6, 2))
        .repeat(2, 1, 1, 1),
    )
    _args = {"things": [], "stuffs": [], "return_per_class": True}
    _args[class_type] = [class1, class2, class3]
    return _input, _args


def _reference_fn_0_0(preds, target) -> np.ndarray:
    """Baseline result for the _INPUTS_0, _ARGS_0 combination."""
    return np.array([0.7753])


def _reference_fn_0_1(preds, target) -> np.ndarray:
    """Baseline result for the _INPUTS_0, _ARGS_1 combination."""
    return np.array([np.nan])


def _reference_fn_1_2(preds, target) -> np.ndarray:
    """Baseline result for the _INPUTS_1, _ARGS_2 combination."""
    return np.array([(2 / 3 + 1 + 2 / 3) / 3])


def _reference_fn_class_order(preds, target) -> np.ndarray:
    """Baseline result for the result of _get_class_order_test_input_args."""
    return np.array([1, 0, 2 / 3])


class TestPanopticQuality(MetricTester):
    """Test class for `PanopticQuality` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    @pytest.mark.parametrize(
        ("inputs", "args", "reference_metric"),
        [
            (_INPUTS_0, _ARGS_0, _reference_fn_0_0),
            (_INPUTS_0, _ARGS_1, _reference_fn_0_1),
            (_INPUTS_1, _ARGS_2, _reference_fn_1_2),
            (
                *_get_class_order_test_input_args("stuffs", 0, 2, 1),
                _reference_fn_class_order,
            ),
            (
                *_get_class_order_test_input_args("stuffs", 0, 3, 2),
                _reference_fn_class_order,
            ),
            (
                *_get_class_order_test_input_args("stuffs", 0, 10, 2),
                _reference_fn_class_order,
            ),
            (
                *_get_class_order_test_input_args("things", 0, 2, 1),
                _reference_fn_class_order,
            ),
            (
                *_get_class_order_test_input_args("things", 0, 3, 2),
                _reference_fn_class_order,
            ),
            (
                *_get_class_order_test_input_args("things", 0, 10, 2),
                _reference_fn_class_order,
            ),
        ],
    )
    def test_panoptic_quality_class(self, ddp, inputs, args, reference_metric):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=inputs.preds,
            target=inputs.target,
            metric_class=PanopticQuality,
            reference_metric=reference_metric,
            check_batch=False,
            metric_args=args,
        )

    def test_panoptic_quality_functional(self):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            _INPUTS_0.preds,
            _INPUTS_0.target,
            metric_functional=panoptic_quality,
            reference_metric=_reference_fn_0_0,
            metric_args=_ARGS_0,
        )


def test_empty_metric():
    """Test empty metric."""
    with pytest.raises(
        ValueError, match="At least one of `things` and `stuffs` must be non-empty"
    ):
        metric = PanopticQuality(things=[], stuffs=[])
    metric = PanopticQuality(things=[0], stuffs=[])
    assert paddle.isnan(metric.compute())


def test_error_on_wrong_input():
    """Test class input validation."""
    with pytest.raises(
        TypeError, match="Expected argument `stuffs` to contain `int` categories.*"
    ):
        PanopticQuality(things={0}, stuffs={"sky"})
    with pytest.raises(
        ValueError,
        match="Expected arguments `things` and `stuffs` to have distinct keys.*",
    ):
        PanopticQuality(things={0}, stuffs={0})
    metric = PanopticQuality(
        things={0, 1, 3}, stuffs={2, 8}, allow_unknown_preds_category=True
    )
    valid_images = paddle.randint(low=0, high=9, shape=(8, 64, 64, 2))
    metric.update(valid_images, valid_images)
    valid_point_clouds = paddle.randint(low=0, high=9, shape=(1, 100, 2))
    metric.update(valid_point_clouds, valid_point_clouds)
    with pytest.raises(
        TypeError, match="Expected argument `preds` to be of type `paddle.Tensor`.*"
    ):
        metric.update([], valid_images)
    with pytest.raises(
        TypeError, match="Expected argument `target` to be of type `paddle.Tensor`.*"
    ):
        metric.update(valid_images, [])
    preds = paddle.randint(low=0, high=9, shape=(2, 400, 300, 2))
    target = paddle.randint(low=0, high=9, shape=(2, 30, 40, 2))
    with pytest.raises(
        ValueError,
        match="Expected argument `preds` and `target` to have the same shape.*",
    ):
        metric.update(preds, target)
    preds = paddle.randint(low=0, high=9, shape=(1, 2))
    with pytest.raises(
        ValueError,
        match="Expected argument `preds` to have at least one spatial dimension.*",
    ):
        metric.update(preds, preds)
    preds = paddle.randint(low=0, high=9, shape=(1, 64, 64, 8))
    with pytest.raises(
        ValueError,
        match="Expected argument `preds` to have exactly 2 channels in the last dimension.*",
    ):
        metric.update(preds, preds)
    metric = PanopticQuality(things=[0], stuffs=[1], allow_unknown_preds_category=False)
    preds = paddle.randint(low=0, high=1, shape=(1, 100, 2))
    preds[0, 0, 0] = 2
    with pytest.raises(ValueError, match="Unknown categories found.*"):
        metric.update(preds, preds)


def test_extreme_values():
    """Test that the metric returns expected values in trivial cases."""
    assert panoptic_quality(_INPUTS_0.target[0], _INPUTS_0.target[0], **_ARGS_0) == 1.0
    assert (
        panoptic_quality(_INPUTS_0.target[0], _INPUTS_0.target[0] + 1, **_ARGS_0) == 0.0
    )


@pytest.mark.parametrize(
    ("inputs", "args", "cat_dim"),
    [
        (_INPUTS_0, _ARGS_0, 0),
        (_INPUTS_0, _ARGS_0, 1),
        (_INPUTS_0, _ARGS_0, 2),
        (_INPUTS_1, _ARGS_2, 0),
        (_INPUTS_1, _ARGS_2, 1),
    ],
)
def test_ignore_mask(inputs: _Input, args: dict[str, Any], cat_dim: int):
    """Test that the metric correctly ignores regions of the inputs that do not map to a know category ID."""
    preds = inputs.preds[0]
    target = inputs.target[0]
    value = panoptic_quality(preds, target, **args)
    ignored_regions = paddle.zeros_like(preds)
    ignored_regions[..., 0] = 255
    preds_new = paddle.concat([preds, preds], axis=cat_dim)
    target_new = paddle.concat([target, ignored_regions], axis=cat_dim)
    value_new = panoptic_quality(preds_new, target_new, **args)
    assert value == value_new
