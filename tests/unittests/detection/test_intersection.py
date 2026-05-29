from functools import partial

import paddle
from paddle import Tensor
import pytest

from paddlemetrics.detection.ciou import CompleteIntersectionOverUnion
from paddlemetrics.detection.diou import DistanceIntersectionOverUnion
from paddlemetrics.detection.giou import GeneralizedIntersectionOverUnion
from paddlemetrics.detection.iou import IntersectionOverUnion
from paddlemetrics.functional.detection.ciou import \
    complete_intersection_over_union
from paddlemetrics.functional.detection.diou import \
    distance_intersection_over_union
from paddlemetrics.functional.detection.giou import \
    generalized_intersection_over_union
from paddlemetrics.functional.detection.iou import intersection_over_union
from paddlemetrics.utils.imports import _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    pass
else:
    tv_iou, tv_ciou, tv_diou, tv_giou = ..., ..., ..., ...
from unittests._helpers.testers import MetricTester


def _tv_wrapper(preds, target, base_fn, aggregate=True, iou_threshold=None):
    out = base_fn(preds, target)
    if iou_threshold is not None:
        out[out < iou_threshold] = 0
    if aggregate:
        return out.diag().mean()
    return out


def _tv_wrapper_class(
    preds, target, base_fn, respect_labels, iou_threshold, class_metrics
):
    iou = []
    target_labels = []
    pred_labels = []
    for p, t in zip(preds, target):
        out = base_fn(p["boxes"], t["boxes"])
        if iou_threshold is not None:
            out[out < iou_threshold] = -1
        if respect_labels:
            labels_eq = p["labels"].unsqueeze(1) == t["labels"].unsqueeze(0)
            out[~labels_eq] = -1
        iou.append(out)
        target_labels.append(t["labels"])
        pred_labels.append(p["labels"])
    valid_scores = [mat[mat != -1] for mat in iou if mat[mat != -1].size > 0]
    if valid_scores:
        all_valid = paddle.concat(valid_scores)
        score = all_valid.mean()
    else:
        score = paddle.tensor(0.0)
    base_name = {
_preds_fn = (
    paddle.tensor(
        [
            [296.55, 93.96, 314.97, 152.79],
            [328.94, 97.05, 342.49, 122.98],
            [356.62, 95.47, 372.33, 147.55],
        ]
    )
    .unsqueeze(0)
    .repeat(4, 1, 1)
)
_target_fn = (
    paddle.tensor(
        [
            [300.0, 100.0, 315.0, 150.0],
            [330.0, 100.0, 350.0, 125.0],
            [350.0, 100.0, 375.0, 150.0],
        ]
    )
    .unsqueeze(0)
    .repeat(4, 1, 1)
)
_preds_class = [
    [
        {
            "boxes": paddle.tensor(
                [[296.55, 93.96, 314.97, 152.79], [298.55, 98.96, 314.97, 151.79]]
            ),
            "labels": paddle.tensor([4, 5]),
        }
    ],
    [
        {
            "boxes": paddle.tensor(
                [[296.55, 93.96, 314.97, 152.79], [298.55, 98.96, 314.97, 151.79]]
            ),
            "labels": paddle.tensor([4, 5]),
        }
    ],
    [
        {
            "boxes": paddle.tensor([[328.94, 97.05, 342.49, 122.98]]),
            "labels": paddle.tensor([4]),
        },
        {
            "boxes": paddle.tensor([[356.62, 95.47, 372.33, 147.55]]),
            "labels": paddle.tensor([4]),
        },
    ],
    [
        {
            "boxes": paddle.tensor([[328.94, 97.05, 342.49, 122.98]]),
            "labels": paddle.tensor([5]),
        },
        {
            "boxes": paddle.tensor([[356.62, 95.47, 372.33, 147.55]]),
            "labels": paddle.tensor([5]),
        },
    ],
]
_target_class = [
    [
        {
            "boxes": paddle.tensor([[300.0, 100.0, 315.0, 150.0]]),
            "labels": paddle.tensor([5]),
        }
    ],
    [
        {
            "boxes": paddle.tensor([[300.0, 100.0, 315.0, 150.0]]),
            "labels": paddle.tensor([5]),
        }
    ],
    [
        {
            "boxes": paddle.tensor([[330.0, 100.0, 350.0, 125.0]]),
            "labels": paddle.tensor([4]),
        },
        {
            "boxes": paddle.tensor([[350.0, 100.0, 375.0, 150.0]]),
            "labels": paddle.tensor([4]),
        },
    ],
    [
        {
            "boxes": paddle.tensor([[330.0, 100.0, 350.0, 125.0]]),
            "labels": paddle.tensor([5]),
        },
        {
            "boxes": paddle.tensor([[350.0, 100.0, 375.0, 150.0]]),
            "labels": paddle.tensor([4]),
        },
    ],
]


def _add_noise(x, scale=10):
    """Add noise to boxes and labels to make testing non-deterministic."""
    if isinstance(x, paddle.Tensor):
        return x + scale * paddle.rand_like(x)
    for batch in x:
        for sample in batch:
            sample["boxes"] = _add_noise(sample["boxes"], scale)
            sample["labels"] += abs(
                paddle.randint_like(x=sample["labels"], low=0, high=10)
            )
    return x


@pytest.mark.parametrize(
    ("class_metric", "functional_metric", "reference_metric"),
    [
        (IntersectionOverUnion, intersection_over_union, tv_iou),
        (CompleteIntersectionOverUnion, complete_intersection_over_union, tv_ciou),
        (DistanceIntersectionOverUnion, distance_intersection_over_union, tv_diou),
        (
            GeneralizedIntersectionOverUnion,
            generalized_intersection_over_union,
            tv_giou,
        ),
    ],
)
class TestIntersectionMetrics(MetricTester):
    """Tester class for the different intersection metrics."""

    @pytest.mark.parametrize(
        ("preds", "target"),
        [
            (_preds_class, _target_class),
            (_add_noise(_preds_class), _add_noise(_target_class)),
        ],
    )
    @pytest.mark.parametrize("respect_labels", [True, False])
    @pytest.mark.parametrize("iou_threshold", [None, 0.5, 0.7, 0.9])
    @pytest.mark.parametrize("class_metrics", [True, False])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_intersection_class(
        self,
        class_metric,
        functional_metric,
        reference_metric,
        preds,
        target,
        respect_labels,
        iou_threshold,
        class_metrics,
        ddp,
    ):
        """Test class implementation for correctness."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=class_metric,
            reference_metric=partial(
                _tv_wrapper_class,
                base_fn=reference_metric,
                respect_labels=respect_labels,
                iou_threshold=iou_threshold,
                class_metrics=class_metrics,
            ),
            metric_args={
                "respect_labels": respect_labels,
                "iou_threshold": iou_threshold,
                "class_metrics": class_metrics,
            },
            check_batch=not class_metrics,
        )

    @pytest.mark.parametrize(
        ("preds", "target"),
        [(_preds_fn, _target_fn), (_add_noise(_preds_fn), _add_noise(_target_fn))],
    )
    @pytest.mark.parametrize("aggregate", [True, False])
    @pytest.mark.parametrize("iou_threshold", [None, 0.5, 0.7, 0.9])
    def test_intersection_function(
        self,
        class_metric,
        functional_metric,
        reference_metric,
        preds,
        target,
        aggregate,
        iou_threshold,
    ):
        """Test functional implementation for correctness."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=functional_metric,
            reference_metric=partial(
                _tv_wrapper,
                base_fn=reference_metric,
                aggregate=aggregate,
                iou_threshold=iou_threshold,
            ),
            metric_args={"aggregate": aggregate, "iou_threshold": iou_threshold},
        )

    def test_error_on_wrong_input(
        self, class_metric, functional_metric, reference_metric
    ):
        """Test class input validation."""
        metric = class_metric()
        metric.update([], [])
        with pytest.raises(
            ValueError, match="Expected argument `preds` to be of type Sequence"
        ):
            metric.update(paddle.Tensor(), [])
        with pytest.raises(
            ValueError, match="Expected argument `target` to be of type Sequence"
        ):
            metric.update([], paddle.Tensor())
        with pytest.raises(
            ValueError,
            match="Expected argument `preds` and `target` to have the same length",
        ):
            metric.update([{}], [{}, {}])
        with pytest.raises(
            ValueError, match="Expected all dicts in `preds` to contain the `boxes` key"
        ):
            metric.update(
                [{"scores": paddle.Tensor(), "labels": paddle.IntTensor}],
                [{"boxes": paddle.Tensor(), "labels": paddle.IntTensor()}],
            )
        with pytest.raises(
            ValueError,
            match="Expected all dicts in `preds` to contain the `labels` key",
        ):
            metric.update(
                [{"boxes": paddle.Tensor(), "scores": paddle.IntTensor}],
                [{"boxes": paddle.Tensor(), "labels": paddle.IntTensor()}],
            )
        with pytest.raises(
            ValueError,
            match="Expected all dicts in `target` to contain the `boxes` key",
        ):
            metric.update(
                [
                    {
                        "boxes": paddle.Tensor(),
                        "scores": paddle.IntTensor,
                        "labels": paddle.IntTensor,
                    }
                ],
                [{"labels": paddle.IntTensor()}],
            )
        with pytest.raises(
            ValueError,
            match="Expected all dicts in `target` to contain the `labels` key",
        ):
            metric.update(
                [
                    {
                        "boxes": paddle.Tensor(),
                        "scores": paddle.IntTensor,
                        "labels": paddle.IntTensor,
                    }
                ],
                [{"boxes": paddle.IntTensor()}],
            )
        with pytest.raises(
            ValueError, match="Expected all boxes in `preds` to be of type Tensor"
        ):
            metric.update(
                [
                    {
                        "boxes": [],
                        "scores": paddle.Tensor(),
                        "labels": paddle.IntTensor(),
                    }
                ],
                [{"boxes": paddle.Tensor(), "labels": paddle.IntTensor()}],
            )
        with pytest.raises(
            ValueError, match="Expected all labels in `preds` to be of type Tensor"
        ):
            metric.update(
                [{"boxes": paddle.Tensor(), "scores": paddle.Tensor(), "labels": []}],
                [{"boxes": paddle.Tensor(), "labels": paddle.IntTensor()}],
            )
        with pytest.raises(
            ValueError, match="Expected all boxes in `target` to be of type Tensor"
        ):
            metric.update(
                [
                    {
                        "boxes": paddle.Tensor(),
                        "scores": paddle.Tensor(),
                        "labels": paddle.IntTensor(),
                    }
                ],
                [{"boxes": [], "labels": paddle.IntTensor()}],
            )
        with pytest.raises(
            ValueError, match="Expected all labels in `target` to be of type Tensor"
        ):
            metric.update(
                [
                    {
                        "boxes": paddle.Tensor(),
                        "scores": paddle.Tensor(),
                        "labels": paddle.IntTensor(),
                    }
                ],
                [{"boxes": paddle.Tensor(), "labels": []}],
            )

    def test_functional_error_on_wrong_input_shape(
        self, class_metric, functional_metric, reference_metric
    ):
        """Test functional input validation."""
        with pytest.raises(ValueError, match="Expected preds to be of shape.*"):
            functional_metric(paddle.randn(25), paddle.randn(25, 4))
        with pytest.raises(ValueError, match="Expected target to be of shape.*"):
            functional_metric(paddle.randn(25, 4), paddle.randn(25))
        with pytest.raises(ValueError, match="Expected preds to be of shape.*"):
            functional_metric(paddle.randn(25, 25), paddle.randn(25, 4))
        with pytest.raises(ValueError, match="Expected target to be of shape.*"):
            functional_metric(paddle.randn(25, 4), paddle.randn(25, 25))

    def test_corner_case_only_one_empty_prediction(
        self, class_metric, functional_metric, reference_metric
    ):
        """Test that the metric does not crash when there is only one empty prediction."""
        target = [
            {
                "boxes": paddle.tensor(
                    [
                        [8.0, 70.0, 76.0, 110.0],
                        [247.0, 131.0, 315.0, 175.0],
                        [361.0, 177.0, 395.0, 203.0],
                    ]
                ),
                "labels": paddle.tensor([0, 0, 0]),
            }
        ]
        preds = [
            {
                "boxes": paddle.empty(size=(0, 4)),
                "labels": paddle.tensor([], dtype=paddle.int64),
                "scores": paddle.tensor([]),
            }
        ]
        metric = class_metric()
        metric.update(preds, target)
        res = metric.compute()
        for val in res.values():
            assert val == paddle.tensor(0.0)

    def test_empty_preds_and_target(
        self, class_metric, functional_metric, reference_metric
    ):
        """Check that for either empty preds and targets that the metric returns 0 in these cases before averaging."""
        x = [
            {
                "boxes": paddle.empty(size=(0, 4), dtype=paddle.float32),
                "labels": paddle.tensor([], dtype=paddle.long),
            },
            {
                "boxes": paddle.FloatTensor(
                    [[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]]
                ),
                "labels": paddle.LongTensor([1, 2]),
            },
        ]
        y = [
            {
                "boxes": paddle.FloatTensor(
                    [[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]]
                ),
                "labels": paddle.LongTensor([1, 2]),
                "scores": paddle.FloatTensor([0.9, 0.8]),
            },
            {
                "boxes": paddle.FloatTensor(
                    [[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]]
                ),
                "labels": paddle.LongTensor([1, 2]),
                "scores": paddle.FloatTensor([0.9, 0.8]),
            },
        ]
        metric = class_metric()
        metric.update(x, y)
        res = metric.compute()
        for val in res.values():
            assert val == paddle.tensor(0.5)
        metric = class_metric()
        metric.update(y, x)
        res = metric.compute()
        for val in res.values():
            assert val == paddle.tensor(0.5)


def test_corner_case():
    """See issue: https://github.com/Lightning-AI/paddlemetrics/issues/1921."""
    preds = [
        {
            "boxes": paddle.tensor(
                [[300.0, 100.0, 315.0, 150.0], [298.55, 98.96, 314.97, 151.79]]
            ),
            "scores": paddle.tensor([0.236, 0.56]),
            "labels": paddle.tensor([4, 5]),
        }
    ]
    target = [
        {
            "boxes": paddle.tensor(
                [[300.0, 100.0, 315.0, 150.0], [298.55, 98.96, 314.97, 151.79]]
            ),
            "labels": paddle.tensor([4, 5]),
        }
    ]
    metric = IntersectionOverUnion(
        class_metrics=True, iou_threshold=0.75, respect_labels=True
    )
    iou = metric(preds, target)
    for val in iou.values():
        assert val == paddle.tensor(1.0)
    preds = [
        {
            "boxes": paddle.tensor(
                [[296.55, 93.96, 314.97, 152.79], [298.55, 98.96, 314.97, 151.79]]
            ),
            "labels": paddle.tensor([4, 6]),
        }
    ]
    target = [
        {
            "boxes": paddle.tensor(
                [[300.0, 100.0, 315.0, 150.0], [300.0, 100.0, 315.0, 150.0]]
            ),
            "labels": paddle.tensor([4, 5]),
        }
    ]
    expected_out = {
        "iou": 0.6897670030593872,
        "iou/cl_4": 0.6897670030593872,
        "iou/cl_5": 0.0,
        "iou/cl_6": 0.0,
    }
    metric = IntersectionOverUnion(class_metrics=True)
    iou = metric(preds, target)
    for key, val in expected_out.items():
        assert iou[key].item() == val
