import contextlib
import io
import json
from copy import deepcopy
from functools import partial
from itertools import product
from typing import Any

import numpy as np
import paddle
from paddle import Tensor
import pytest
from paddlemetrics.utils.data import apply_to_collection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from unittests._helpers.testers import MetricTester
from unittests.detection import (_DETECTION_BBOX, _DETECTION_SEGM,
                                 _DETECTION_VAL)

from paddlemetrics.detection.mean_ap import MeanAveragePrecision
from paddlemetrics.functional.detection.map import mean_average_precision
from paddlemetrics.utils.imports import (_FASTER_COCO_EVAL_AVAILABLE,
                                            _PYCOCOTOOLS_AVAILABLE)


def _skip_if_faster_coco_eval_missing(backend):
    if backend == "faster_coco_eval" and not _FASTER_COCO_EVAL_AVAILABLE:
        pytest.skip("test requires that faster_coco_eval is installed")


def _generate_coco_inputs(iou_type):
    """Generates inputs for the MAP metric.

    The inputs are generated from the official COCO results json files:
    https://github.com/cocodataset/cocoapi/tree/master/results
    and should therefore correspond directly to the result on the webpage

    """
    batched_preds, batched_target = MeanAveragePrecision().coco_to_tm(
        _DETECTION_BBOX if iou_type == "bbox" else _DETECTION_SEGM,
        _DETECTION_VAL,
        iou_type,
    )
    batched_preds = [batched_preds[10 * i : 10 * (i + 1)] for i in range(10)]
    batched_target = [batched_target[10 * i : 10 * (i + 1)] for i in range(10)]
    return batched_preds, batched_target


_coco_bbox_input = _generate_coco_inputs("bbox")
_coco_segm_input = _generate_coco_inputs("segm")


@pytest.mark.skipif(
    not _PYCOCOTOOLS_AVAILABLE,
    reason="test requires that torchvision=>0.8.0 and pycocotools is installed",
)
@pytest.mark.parametrize("iou_type", ["bbox", "segm"])
@pytest.mark.parametrize("backend", ["pycocotools", "faster_coco_eval"])
def test_tm_to_coco(tmpdir, iou_type, backend):
    """Test that the conversion from TM to COCO format works."""
    preds, target = _coco_bbox_input if iou_type == "bbox" else _coco_segm_input
    metric = MeanAveragePrecision(iou_type=iou_type, backend=backend, box_format="xywh")
    for bp, bt in zip(preds, target):
        metric.update(bp, bt)
    metric.tm_to_coco(f"{tmpdir}/tm_map_input")
    preds_2, target_2 = MeanAveragePrecision().coco_to_tm(
        f"{tmpdir}/tm_map_input_preds.json",
        f"{tmpdir}/tm_map_input_target.json",
        iou_type=iou_type,
        backend=backend,
    )
    preds = [p for batch in preds for p in batch]
    target = [t for batch in target for t in batch]
    for sample1 in preds:
        sample_found = False
        for sample2 in preds_2:
            if iou_type == "segm":
                if (
                    sample1["masks"].shape == sample2["masks"].shape
                    and paddle.allclose(x=sample1["masks"], y=sample2["masks"]).item()
                ):
                    sample_found = True
            elif (
                sample1["boxes"].shape == sample2["boxes"].shape
                and paddle.allclose(x=sample1["boxes"], y=sample2["boxes"]).item()
            ):
                sample_found = True
        assert sample_found, "preds not found"
    for sample1 in target:
        sample_found = False
        for sample2 in target_2:
            if iou_type == "segm":
                if (
                    sample1["masks"].shape == sample2["masks"].shape
                    and paddle.allclose(x=sample1["masks"], y=sample2["masks"]).item()
                ):
                    sample_found = True
            elif (
                sample1["boxes"].shape == sample2["boxes"].shape
                and paddle.allclose(x=sample1["boxes"], y=sample2["boxes"]).item()
            ):
                sample_found = True
        assert sample_found, "target not found"


def _compare_against_coco_fn(
    preds,
    target,
    iou_type,
    iou_thresholds=None,
    rec_thresholds=None,
    class_metrics=True,
):
    """Taken from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb."""
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(_DETECTION_VAL)
        dt = (
            gt.loadRes(_DETECTION_BBOX)
            if iou_type == "bbox"
            else gt.loadRes(_DETECTION_SEGM)
        )
        coco_eval = COCOeval(gt, dt, iou_type)
        if iou_thresholds is not None:
            coco_eval.params.iouThrs = np.array(iou_thresholds, dtype=np.float64)
        if rec_thresholds is not None:
            coco_eval.params.recThrs = np.array(rec_thresholds, dtype=np.float64)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    global_stats = deepcopy(coco_eval.stats)
    map_per_class_values = paddle.Tensor([-1])
    mar_100_per_class_values = paddle.Tensor([-1])
    classes = paddle.tensor(
        list(
            set(paddle.arange(91).tolist())
            - {0, 12, 19, 26, 29, 30, 45, 66, 68, 69, 71, 76, 83, 87, 89}
        )
    )
    if class_metrics:
        map_per_class_list = []
        mar_100_per_class_list = []
        for class_id in classes.tolist():
            coco_eval.params.catIds = [class_id]
            with contextlib.redirect_stdout(io.StringIO()):
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
            class_stats = coco_eval.stats
            map_per_class_list.append(paddle.Tensor([class_stats[0]]))
            mar_100_per_class_list.append(paddle.Tensor([class_stats[8]]))
        map_per_class_values = paddle.Tensor(map_per_class_list)
        mar_100_per_class_values = paddle.Tensor(mar_100_per_class_list)
    return {
        "map": paddle.Tensor([global_stats[0]]),
        "map_50": paddle.Tensor([global_stats[1]]),
        "map_75": paddle.Tensor([global_stats[2]]),
        "map_small": paddle.Tensor([global_stats[3]]),
        "map_medium": paddle.Tensor([global_stats[4]]),
        "map_large": paddle.Tensor([global_stats[5]]),
        "mar_1": paddle.Tensor([global_stats[6]]),
        "mar_10": paddle.Tensor([global_stats[7]]),
        "mar_100": paddle.Tensor([global_stats[8]]),
        "mar_small": paddle.Tensor([global_stats[9]]),
        "mar_medium": paddle.Tensor([global_stats[10]]),
        "mar_large": paddle.Tensor([global_stats[11]]),
        "map_per_class": map_per_class_values,
        "mar_100_per_class": mar_100_per_class_values,
        "classes": classes,
    }


@pytest.mark.skipif(
    not _PYCOCOTOOLS_AVAILABLE,
    reason="test requires that torchvision=>0.8.0 and pycocotools is installed",
)
@pytest.mark.parametrize("iou_type", ["bbox", "segm"])
@pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
@pytest.mark.parametrize("backend", ["pycocotools", "faster_coco_eval"])
class TestMAPUsingCOCOReference(MetricTester):
    """Test map metric on the reference coco data."""

    @pytest.mark.parametrize("iou_thresholds", [None, [0.25, 0.5, 0.75]])
    @pytest.mark.parametrize("rec_thresholds", [None, [0.25, 0.5, 0.75]])
    def test_map(self, iou_type, iou_thresholds, rec_thresholds, ddp, backend):
        """Test modular implementation for correctness."""
        _skip_if_faster_coco_eval_missing(backend)
        preds, target = _coco_bbox_input if iou_type == "bbox" else _coco_segm_input
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MeanAveragePrecision,
            reference_metric=partial(
                _compare_against_coco_fn,
                iou_type=iou_type,
                iou_thresholds=iou_thresholds,
                rec_thresholds=rec_thresholds,
                class_metrics=False,
            ),
            metric_args={
                "iou_type": iou_type,
                "iou_thresholds": iou_thresholds,
                "rec_thresholds": rec_thresholds,
                "class_metrics": False,
                "box_format": "xywh",
                "backend": backend,
            },
            check_batch=False,
            atol=0.01,
        )

    def test_map_classwise(self, iou_type, ddp, backend):
        """Test modular implementation for correctness with classwise=True.

        Needs bigger atol to be stable.

        """
        _skip_if_faster_coco_eval_missing(backend)
        preds, target = _coco_bbox_input if iou_type == "bbox" else _coco_segm_input
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MeanAveragePrecision,
            reference_metric=partial(
                _compare_against_coco_fn, iou_type=iou_type, class_metrics=True
            ),
            metric_args={
                "box_format": "xywh",
                "iou_type": iou_type,
                "class_metrics": True,
                "backend": backend,
            },
            check_batch=False,
            atol=0.1,
        )


@pytest.mark.parametrize("backend", ["pycocotools", "faster_coco_eval"])
def test_compare_both_same_time(tmpdir, backend):
    """Test that the class support evaluating both bbox and segm at the same time."""
    _skip_if_faster_coco_eval_missing(backend)
    with open(_DETECTION_BBOX) as f:
        boxes = json.load(f)
    with open(_DETECTION_SEGM) as f:
        segmentations = json.load(f)
    combined = [{**box, **seg} for box, seg in zip(boxes, segmentations)]
    with open(f"{tmpdir}/combined.json", "w") as f:
        json.dump(combined, f)
    batched_preds, batched_target = MeanAveragePrecision().coco_to_tm(
        f"{tmpdir}/combined.json", _DETECTION_VAL, iou_type=["bbox", "segm"]
    )
    batched_preds = [batched_preds[10 * i : 10 * (i + 1)] for i in range(10)]
    batched_target = [batched_target[10 * i : 10 * (i + 1)] for i in range(10)]
    metric = MeanAveragePrecision(
        iou_type=["bbox", "segm"], box_format="xywh", backend=backend
    )
    for bp, bt in zip(batched_preds, batched_target):
        metric.update(bp, bt)
    res = metric.compute()
    res1 = _compare_against_coco_fn([], [], iou_type="bbox", class_metrics=False)
    res2 = _compare_against_coco_fn([], [], iou_type="segm", class_metrics=False)
    for k, v in res1.items():
        if k == "classes":
            continue
        assert f"bbox_{k}" in res
        assert paddle.allclose(x=res[f"bbox_{k}"], y=v, atol=0.01).item()
    for k, v in res2.items():
        if k == "classes":
            continue
        assert f"segm_{k}" in res
        assert paddle.allclose(x=res[f"segm_{k}"], y=v, atol=0.01).item()


_inputs = {
    "preds": [
        [
            {
                "boxes": paddle.Tensor([[258.15, 41.29, 606.41, 285.07]]),
                "scores": paddle.Tensor([0.236]),
                "labels": paddle.IntTensor([4]),
            },
            {
                "boxes": paddle.Tensor(
                    [[61.0, 22.75, 565.0, 632.42], [12.66, 3.32, 281.26, 275.23]]
                ),
                "scores": paddle.Tensor([0.318, 0.726]),
                "labels": paddle.IntTensor([3, 2]),
            },
        ],
        [
            {
                "boxes": paddle.Tensor(
                    [
                        [87.87, 276.25, 384.29, 379.43],
                        [0.0, 3.66, 142.15, 316.06],
                        [296.55, 93.96, 314.97, 152.79],
                        [328.94, 97.05, 342.49, 122.98],
                        [356.62, 95.47, 372.33, 147.55],
                        [464.08, 105.09, 495.74, 146.99],
                        [276.11, 103.84, 291.44, 150.72],
                    ]
                ),
                "scores": paddle.Tensor(
                    [0.546, 0.3, 0.407, 0.611, 0.335, 0.805, 0.953]
                ),
                "labels": paddle.IntTensor([4, 1, 0, 0, 0, 0, 0]),
            },
            {
                "boxes": paddle.Tensor(
                    [
                        [72.92, 45.96, 91.23, 80.57],
                        [45.17, 45.34, 66.28, 79.83],
                        [82.28, 47.04, 99.66, 78.5],
                        [59.96, 46.17, 80.35, 80.48],
                        [75.29, 23.01, 91.85, 50.85],
                        [71.14, 1.1, 96.96, 28.33],
                        [61.34, 55.23, 77.14, 79.57],
                        [41.17, 45.78, 60.99, 78.48],
                        [56.18, 44.8, 64.42, 56.25],
                    ]
                ),
                "scores": paddle.Tensor(
                    [0.532, 0.204, 0.782, 0.202, 0.883, 0.271, 0.561, 0.204, 0.349]
                ),
                "labels": paddle.IntTensor([49, 49, 49, 49, 49, 49, 49, 49, 49]),
            },
        ],
    ],
    "target": [
        [
            {
                "boxes": paddle.Tensor([[214.15, 41.29, 562.41, 285.07]]),
                "labels": paddle.IntTensor([4]),
            },
            {
                "boxes": paddle.Tensor(
                    [[13.0, 22.75, 548.98, 632.42], [1.66, 3.32, 270.26, 275.23]]
                ),
                "labels": paddle.IntTensor([2, 2]),
            },
        ],
        [
            {
                "boxes": paddle.Tensor(
                    [
                        [61.87, 276.25, 358.29, 379.43],
                        [2.75, 3.66, 162.15, 316.06],
                        [295.55, 93.96, 313.97, 152.79],
                        [326.94, 97.05, 340.49, 122.98],
                        [356.62, 95.47, 372.33, 147.55],
                        [462.08, 105.09, 493.74, 146.99],
                        [277.11, 103.84, 292.44, 150.72],
                    ]
                ),
                "labels": paddle.IntTensor([4, 1, 0, 0, 0, 0, 0]),
            },
            {
                "boxes": paddle.Tensor(
                    [
                        [72.92, 45.96, 91.23, 80.57],
                        [50.17, 45.34, 71.28, 79.83],
                        [81.28, 47.04, 98.66, 78.5],
                        [63.96, 46.17, 84.35, 80.48],
                        [75.29, 23.01, 91.85, 50.85],
                        [56.39, 21.65, 75.66, 45.54],
                        [73.14, 1.1, 98.96, 28.33],
                        [62.34, 55.23, 78.14, 79.57],
                        [44.17, 45.78, 63.99, 78.48],
                        [58.18, 44.8, 66.42, 56.25],
                    ]
                ),
                "labels": paddle.IntTensor([49, 49, 49, 49, 49, 49, 49, 49, 49, 49]),
            },
        ],
    ],
}
_inputs2 = {
    "preds": [
        [
            {
                "boxes": paddle.Tensor([[258.0, 41.0, 606.0, 285.0]]),
                "scores": paddle.Tensor([0.536]),
                "labels": paddle.IntTensor([0]),
            }
        ],
        [
            {
                "boxes": paddle.Tensor([[258.0, 41.0, 606.0, 285.0]]),
                "scores": paddle.Tensor([0.536]),
                "labels": paddle.IntTensor([0]),
            }
        ],
    ],
    "target": [
        [
            {
                "boxes": paddle.Tensor([[214.0, 41.0, 562.0, 285.0]]),
                "labels": paddle.IntTensor([0]),
            }
        ],
        [{"boxes": paddle.Tensor([]), "labels": paddle.IntTensor([])}],
    ],
}
_inputs3 = {
    "preds": [
        [
            {
                "boxes": paddle.Tensor([[258.0, 41.0, 606.0, 285.0]]),
                "scores": paddle.Tensor([0.536]),
                "labels": paddle.IntTensor([0]),
            }
        ],
        [
            {
                "boxes": paddle.Tensor([]),
                "scores": paddle.Tensor([]),
                "labels": paddle.IntTensor([]),
            }
        ],
    ],
    "target": [
        [
            {
                "boxes": paddle.Tensor([[214.0, 41.0, 562.0, 285.0]]),
                "labels": paddle.IntTensor([0]),
            }
        ],
        [
            {
                "boxes": paddle.Tensor([[1.0, 2.0, 3.0, 4.0]]),
                "scores": paddle.Tensor([0.8]),
                "labels": paddle.IntTensor([1]),
            }
        ],
    ],
}


def _generate_random_segm_input(
    device, batch_size=2, num_preds_size=10, num_gt_size=10, random_size=True
):
    """Generate random inputs for mAP when iou_type=segm."""
    preds = []
    targets = []
    for _ in range(batch_size):
        result = {}
        num_preds = (
            paddle.randint(low=0, high=num_preds_size, shape=(1,)).item()
            if random_size
            else num_preds_size
        )
        result["scores"] = paddle.rand((num_preds,), device=device)
        result["labels"] = paddle.randint(low=0, high=10, shape=(num_preds,))
        result["masks"] = paddle.randint(
            low=0, high=2, shape=(num_preds, 10, 10)
        ).bool()
        preds.append(result)
        gt = {}
        num_gt = (
            paddle.randint(low=0, high=num_gt_size, shape=(1,)).item()
            if random_size
            else num_gt_size
        )
        gt["labels"] = paddle.randint(low=0, high=10, shape=(num_gt,))
        gt["masks"] = paddle.randint(low=0, high=2, shape=(num_gt, 10, 10)).bool()
        targets.append(gt)
    return preds, targets


@pytest.mark.skipif(
    not _PYCOCOTOOLS_AVAILABLE,
    reason="test requires that torchvision=>0.8.0 and pycocotools is installed",
)
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("pycocotools"),
        pytest.param(
            "faster_coco_eval",
            marks=pytest.mark.skipif(
                not _FASTER_COCO_EVAL_AVAILABLE,
                reason="test requires that faster_coco_eval is installed",
            ),
        ),
    ],
)
class TestMapProperties:
    """Test class collection different tests for different properties parametrized by backend argument."""

    def test_error_on_wrong_init(self, backend):
        """Test class raises the expected errors."""
        MeanAveragePrecision(backend=backend)
        with pytest.raises(
            ValueError, match="Expected argument `class_metrics` to be a boolean"
        ):
            MeanAveragePrecision(class_metrics=0, backend=backend)

    def test_empty_preds(self, backend):
        """Test empty predictions."""
        metric = MeanAveragePrecision(backend=backend)
        metric.update(
            [
                {
                    "boxes": paddle.Tensor([]),
                    "scores": paddle.Tensor([]),
                    "labels": paddle.IntTensor([]),
                }
            ],
            [
                {
                    "boxes": paddle.Tensor([[214.15, 41.29, 562.41, 285.07]]),
                    "labels": paddle.IntTensor([4]),
                }
            ],
        )
        metric.compute()

    def test_empty_ground_truths(self, backend):
        """Test empty ground truths."""
        metric = MeanAveragePrecision(backend=backend)
        metric.update(
            [
                {
                    "boxes": paddle.Tensor([[214.15, 41.29, 562.41, 285.07]]),
                    "scores": paddle.Tensor([0.5]),
                    "labels": paddle.IntTensor([4]),
                }
            ],
            [{"boxes": paddle.Tensor([]), "labels": paddle.IntTensor([])}],
        )
        metric.compute()

    def test_empty_ground_truths_xywh(self, backend):
        """Test empty ground truths in xywh format."""
        metric = MeanAveragePrecision(box_format="xywh", backend=backend)
        metric.update(
            [
                {
                    "boxes": paddle.Tensor([[214.15, 41.29, 348.26, 243.78]]),
                    "scores": paddle.Tensor([0.5]),
                    "labels": paddle.IntTensor([4]),
                }
            ],
            [{"boxes": paddle.Tensor([]), "labels": paddle.IntTensor([])}],
        )
        metric.compute()

    def test_empty_preds_xywh(self, backend):
        """Test empty predictions in xywh format."""
        metric = MeanAveragePrecision(box_format="xywh", backend=backend)
        metric.update(
            [
                {
                    "boxes": paddle.Tensor([]),
                    "scores": paddle.Tensor([]),
                    "labels": paddle.IntTensor([]),
                }
            ],
            [
                {
                    "boxes": paddle.Tensor([[214.15, 41.29, 348.26, 243.78]]),
                    "labels": paddle.IntTensor([4]),
                }
            ],
        )
        metric.compute()

    def test_empty_ground_truths_cxcywh(self, backend):
        """Test empty ground truths in cxcywh format."""
        metric = MeanAveragePrecision(box_format="cxcywh", backend=backend)
        metric.update(
            [
                {
                    "boxes": paddle.Tensor([[388.28, 163.18, 348.26, 243.78]]),
                    "scores": paddle.Tensor([0.5]),
                    "labels": paddle.IntTensor([4]),
                }
            ],
            [{"boxes": paddle.Tensor([]), "labels": paddle.IntTensor([])}],
        )
        metric.compute()

    def test_empty_preds_cxcywh(self, backend):
        """Test empty predictions in cxcywh format."""
        metric = MeanAveragePrecision(box_format="cxcywh", backend=backend)
        metric.update(
            [
                {
                    "boxes": paddle.Tensor([]),
                    "scores": paddle.Tensor([]),
                    "labels": paddle.IntTensor([]),
                }
            ],
            [
                {
                    "boxes": paddle.Tensor([[388.28, 163.18, 348.26, 243.78]]),
                    "labels": paddle.IntTensor([4]),
                }
            ],
        )
        metric.compute()

    @pytest.mark.skipif(
        not paddle.cuda.is_available(), reason="test requires CUDA availability"
    )
    @pytest.mark.parametrize("inputs", [_inputs, _inputs2, _inputs3])
    def test_map_gpu(self, backend, inputs):
        """Test predictions on single gpu."""
        metric = MeanAveragePrecision(backend=backend)
        metric = metric.to("cuda")
        for preds, targets in zip(
            deepcopy(inputs["preds"]), deepcopy(inputs["target"])
        ):
            metric.update(
                apply_to_collection(preds, Tensor, lambda x: x.to("cuda")),
                apply_to_collection(targets, Tensor, lambda x: x.to("cuda")),
            )
        metric.compute()

    @pytest.mark.skipif(
        not paddle.cuda.is_available(), reason="test requires CUDA availability"
    )
    def test_map_with_custom_thresholds(self, backend):
        """Test that map works with custom iou thresholds."""
        metric = MeanAveragePrecision(iou_thresholds=[0.1, 0.2], backend=backend)
        metric = metric.to("cuda")
        for preds, targets in zip(
            deepcopy(_inputs["preds"]), deepcopy(_inputs["target"])
        ):
            metric.update(
                apply_to_collection(preds, Tensor, lambda x: x.to("cuda")),
                apply_to_collection(targets, Tensor, lambda x: x.to("cuda")),
            )
        res = metric.compute()
        assert res["map_50"].item() == -1
        assert res["map_75"].item() == -1

    def test_empty_metric(self, backend):
        """Test empty metric."""
        metric = MeanAveragePrecision(backend=backend)
        metric.compute()

    def test_missing_pred(self, backend):
        """One good detection, one false negative.

        Map should be lower than 1. Actually it is 0.5, but the exact value depends on where we are sampling (i.e.
        recall's values)

        """
        gts = [
            {
                "boxes": paddle.Tensor([[10, 20, 15, 25]]),
                "labels": paddle.IntTensor([0]),
            },
            {
                "boxes": paddle.Tensor([[10, 20, 15, 25]]),
                "labels": paddle.IntTensor([0]),
            },
        ]
        preds = [
            {
                "boxes": paddle.Tensor([[10, 20, 15, 25]]),
                "scores": paddle.Tensor([0.9]),
                "labels": paddle.IntTensor([0]),
            },
            {
                "boxes": paddle.Tensor([]),
                "scores": paddle.Tensor([]),
                "labels": paddle.IntTensor([]),
            },
        ]
        metric = MeanAveragePrecision(backend=backend)
        metric.update(preds, gts)
        result = metric.compute()
        assert result["map"] < 1, "MAP cannot be 1, as there is a missing prediction."

    def test_missing_gt(self, backend):
        """The symmetric case of test_missing_pred.

        One good detection, one false positive. Map should be lower than 1. Actually it is 0.5, but the exact value
        depends on where we are sampling (i.e. recall's values)

        """
        gts = [
            {
                "boxes": paddle.Tensor([[10, 20, 15, 25]]),
                "labels": paddle.IntTensor([0]),
            },
            {"boxes": paddle.Tensor([]), "labels": paddle.IntTensor([])},
        ]
        preds = [
            {
                "boxes": paddle.Tensor([[10, 20, 15, 25]]),
                "scores": paddle.Tensor([0.9]),
                "labels": paddle.IntTensor([0]),
            },
            {
                "boxes": paddle.Tensor([[10, 20, 15, 25]]),
                "scores": paddle.Tensor([0.95]),
                "labels": paddle.IntTensor([0]),
            },
        ]
        metric = MeanAveragePrecision(backend=backend)
        metric.update(preds, gts)
        result = metric.compute()
        assert (
            result["map"] < 1
        ), "MAP cannot be 1, as there is an image with no ground truth, but some predictions."

    def test_segm_iou_empty_gt_mask(self, backend):
        """Test empty ground truths."""
        metric = MeanAveragePrecision(iou_type="segm", backend=backend)
        metric.update(
            [
                {
                    "masks": paddle.randint(low=0, high=1, shape=(1, 10, 10)).bool(),
                    "scores": paddle.Tensor([0.5]),
                    "labels": paddle.IntTensor([4]),
                }
            ],
            [{"masks": paddle.Tensor([]), "labels": paddle.IntTensor([])}],
        )
        res = metric.compute()
        for key, value in res.items():
            if key == "classes":
                continue
            assert value.item() == -1, f"Expected -1 for {key}"
        assert res["classes"] == 4

    def test_segm_iou_empty_pred_mask(self, backend):
        """Test empty predictions."""
        metric = MeanAveragePrecision(iou_type="segm", backend=backend)
        metric.update(
            [
                {
                    "masks": paddle.BoolTensor([]),
                    "scores": paddle.Tensor([]),
                    "labels": paddle.IntTensor([]),
                }
            ],
            [
                {
                    "masks": paddle.randint(low=0, high=1, shape=(1, 10, 10)).bool(),
                    "labels": paddle.IntTensor([4]),
                }
            ],
        )
        res = metric.compute()
        for key, value in res.items():
            if key == "classes":
                continue
            assert value.item() == -1, f"Expected -1 for {key}"
        assert res["classes"] == 4

    def test_error_on_wrong_input(self, backend):
        """Test class input validation."""
        metric = MeanAveragePrecision(backend=backend)
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
            match="Expected all dicts in `preds` to contain the `scores` key",
        ):
            metric.update(
                [{"boxes": paddle.Tensor(), "labels": paddle.IntTensor}],
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
            ValueError, match="Expected all scores in `preds` to be of type Tensor"
        ):
            metric.update(
                [
                    {
                        "boxes": paddle.Tensor(),
                        "scores": [],
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

    @pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires cuda")
    def test_device_changing(self, backend):
        """See issue: https://github.com/Lightning-AI/paddlemetrics/issues/1743.

        Checks that the custom apply function of the metric works as expected.
        """
        device = "cuda"
        metric = MeanAveragePrecision(iou_type="segm", backend=backend).to(device)
        for _ in range(2):
            preds, targets = _generate_random_segm_input(device)
            metric.update(preds, targets)
        metric = metric.cpu()
        val = metric.compute()
        assert isinstance(val, dict)

    @pytest.mark.parametrize(
        ("box_format", "iou_val_expected", "map_val_expected"),
        [("xyxy", 0.25, 1), ("xywh", 0.143, 0.0), ("cxcywh", 0.143, 0.0)],
    )
    def test_for_box_format(
        self, box_format, iou_val_expected, map_val_expected, backend
    ):
        """Test that only the correct box format lead to a score of 1.

        See issue: https://github.com/Lightning-AI/paddlemetrics/issues/1908.

        """
        predictions = [
            {
                "boxes": paddle.tensor([[0.5, 0.5, 1, 1]]),
                "scores": paddle.tensor([1.0]),
                "labels": paddle.tensor([0]),
            }
        ]
        targets = [
            {"boxes": paddle.tensor([[0, 0, 1, 1]]), "labels": paddle.tensor([0])}
        ]
        metric = MeanAveragePrecision(
            box_format=box_format,
            iou_thresholds=[0.2],
            extended_summary=True,
            backend=backend,
        )
        metric.update(predictions, targets)
        result = metric.compute()
        assert result["map"].item() == map_val_expected
        assert round(float(result["ious"][0, 0]), 3) == iou_val_expected

    @pytest.mark.parametrize("iou_type", ["bbox", "segm"])
    @pytest.mark.parametrize("warn_on_many_detections", [False])
    def test_warning_on_many_detections(
        self, iou_type, warn_on_many_detections, backend, recwarn
    ):
        """Test that a warning is raised when there are many detections."""
        if iou_type == "bbox":
            preds = [
                {
                    "boxes": paddle.tensor([[0.5, 0.5, 1, 1]]).repeat(101, 1),
                    "scores": paddle.tensor([1.0]).repeat(101),
                    "labels": paddle.tensor([0]).repeat(101),
                }
            ]
            targets = [
                {"boxes": paddle.tensor([[0, 0, 1, 1]]), "labels": paddle.tensor([0])}
            ]
        else:
            preds, targets = _generate_random_segm_input("cpu", 1, 101, 10, False)
        metric = MeanAveragePrecision(iou_type=iou_type, backend=backend)
        metric.warn_on_many_detections = warn_on_many_detections
        if warn_on_many_detections:
            with pytest.warns(
                UserWarning,
                match="Encountered more than 100 detections in a single image.*",
            ):
                metric.update(preds, targets)
        else:
            assert len(recwarn) == 0

    @pytest.mark.parametrize(
        (
            "preds",
            "target",
            "expected_iou_len",
            "iou_keys",
            "precision_shape",
            "recall_shape",
            "scores_shape",
        ),
        [
            (
                [
                    [
                        {
                            "boxes": paddle.tensor([[0.5, 0.5, 1, 1]]),
                            "scores": paddle.tensor([1.0]),
                            "labels": paddle.tensor([0]),
                        }
                    ]
                ],
                [
                    [
                        {
                            "boxes": paddle.tensor([[0, 0, 1, 1]]),
                            "labels": paddle.tensor([0]),
                        }
                    ]
                ],
                1,
                [(0, 0)],
                (10, 101, 1, 4, 3),
                (10, 1, 4, 3),
                (10, 101, 1, 4, 3),
            ),
            (
                _inputs["preds"],
                _inputs["target"],
                24,
                list(product([0, 1, 2, 3], [0, 1, 2, 3, 4, 49])),
                (10, 101, 6, 4, 3),
                (10, 6, 4, 3),
                (10, 101, 6, 4, 3),
            ),
        ],
    )
    def test_for_extended_stats(
        self,
        preds,
        target,
        expected_iou_len,
        iou_keys,
        precision_shape,
        recall_shape,
        scores_shape,
        backend,
    ):
        """Test that extended stats are computed correctly."""
        metric = MeanAveragePrecision(extended_summary=True, backend=backend)
        for p, t in zip(preds, target):
            metric.update(p, t)
        result = metric.compute()
        ious = result["ious"]
        assert isinstance(ious, dict)
        assert len(ious) == expected_iou_len
        for key in ious:
            assert key in iou_keys
        precision = result["precision"]
        assert isinstance(precision, paddle.Tensor)
        assert precision.shape == precision_shape
        recall = result["recall"]
        assert isinstance(recall, paddle.Tensor)
        assert recall.shape == recall_shape
        scores = result["scores"]
        assert isinstance(scores, paddle.Tensor)
        assert scores.shape == scores_shape

    @pytest.mark.parametrize("class_metrics", [False])
    def test_average_argument(self, class_metrics, backend):
        """Test that average argument works.

        Calculating macro on inputs that only have one label should be the same as micro. Calculating class metrics
        should be the same regardless of average argument.

        """
        _preds = deepcopy(_inputs["preds"])
        _target = deepcopy(_inputs["target"])
        for target in _target:
            for batch_idx in range(len(target)):
                target[batch_idx]["labels"] = target[batch_idx]["labels"] + 2
        for preds in _preds:
            for batch_idx in range(len(preds)):
                preds[batch_idx]["labels"] = preds[batch_idx]["labels"] + 2
        if not class_metrics:
            _preds = apply_to_collection(
                deepcopy(_preds), IntTensor, lambda x: paddle.ones_like(x)
            )
            _target = apply_to_collection(
                deepcopy(_target), IntTensor, lambda x: paddle.ones_like(x)
            )
        metric_micro = MeanAveragePrecision(
            average="micro", class_metrics=class_metrics, backend=backend
        )
        metric_micro.update(
            deepcopy(_inputs["preds"][0]), deepcopy(_inputs["target"][0])
        )
        metric_micro.update(
            deepcopy(_inputs["preds"][1]), deepcopy(_inputs["target"][1])
        )
        result_micro = metric_micro.compute()
        metric_macro = MeanAveragePrecision(
            average="macro", class_metrics=class_metrics, backend=backend
        )
        metric_macro.update(_preds[0], _target[0])
        metric_macro.update(_preds[1], _target[1])
        result_macro = metric_macro.compute()
        if class_metrics:
            print(result_macro["map_per_class"], result_micro["map_per_class"])
            assert paddle.allclose(
                x=result_macro["map_per_class"], y=result_micro["map_per_class"]
            ).item()
            assert paddle.allclose(
                x=result_macro["mar_100_per_class"], y=result_micro["mar_100_per_class"]
            ).item()
        else:
            for key in result_macro:
                if key == "classes":
                    continue
                assert paddle.allclose(x=result_macro[key], y=result_micro[key]).item()

    def test_many_detection_thresholds(self, backend):
        """Test how metric behaves when there are many detection thresholds.

        Known to fail with the default pycocotools backend.
        See issue: https://github.com/Lightning-AI/paddlemetrics/issues/1153

        """
        preds = [
            {
                "boxes": paddle.tensor([[258.0, 41.0, 606.0, 285.0]]),
                "scores": paddle.tensor([0.536]),
                "labels": paddle.tensor([0]),
            }
        ]
        target = [
            {
                "boxes": paddle.tensor([[214.0, 41.0, 562.0, 285.0]]),
                "labels": paddle.tensor([0]),
            }
        ]
        metric = MeanAveragePrecision(
            max_detection_thresholds=[1, 10, 1000], backend=backend
        )
        res = metric(preds, target)
        if backend == "pycocotools":
            assert round(res["map"].item(), 5) != 0.6
        else:
            assert round(res["map"].item(), 5) == 0.6
        assert "mar_1" in res
        assert "mar_10" in res
        assert "mar_1000" in res

    @pytest.mark.parametrize("max_detection_thresholds", [[1, 10], [1, 10, 50, 100]])
    def test_with_more_and_less_detection_thresholds(
        self, max_detection_thresholds, backend
    ):
        """Test how metric is working when list of max detection thresholds is not 3.

        This is a known limitation of the pycocotools where values are hardcoded to expect at least 3 elements
        https://github.com/ppwwyyxx/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py#L461

        """
        with pytest.raises(
            ValueError,
            match="When providing a list of max detection thresholds it should have length 3.*",
        ):
            MeanAveragePrecision(
                max_detection_thresholds=max_detection_thresholds, backend=backend
            )


def compare_with_class(functional_result, preds, target, **kwargs: Any):
    """Helper function to compare the functional output with the class-based implementation.

    kwargs are passed along to instantiate MeanAveragePrecision.

    """
    map_metric = MeanAveragePrecision(**kwargs)
    map_metric.update(preds, target)
    class_result = map_metric.compute()
    for key in class_result:
        paddle.testing.assert_close(
            functional_result[key], class_result[key], atol=5e-05, rtol=1e-05
        )


@pytest.mark.parametrize("backend", ["pycocotools", "faster_coco_eval"])
@pytest.mark.parametrize("iou_type", ["bbox", "segm"])
def test_mean_average_precision_iou_type_functional(backend, iou_type):
    """Test that the functional API returns a valid dictionary with the expected keys."""
    preds, target = _coco_bbox_input if iou_type == "bbox" else _coco_segm_input
    preds_flat = [p for batch in preds for p in batch]
    target_flat = [t for batch in target for t in batch]
    functional_result = mean_average_precision(
        preds_flat, target_flat, backend=backend, iou_type=iou_type, box_format="xywh"
    )
    compare_with_class(
        functional_result,
        preds_flat,
        target_flat,
        backend=backend,
        iou_type=iou_type,
        box_format="xywh",
    )


@pytest.mark.parametrize("backend", ["pycocotools", "faster_coco_eval"])
def test_mean_average_precision_basic_functional(backend):
    """Test basic functionality with nonempty inputs by comparing function and class outputs."""
    preds = _inputs["preds"]
    target = _inputs["target"]
    preds_flat = [p for batch in preds for p in batch]
    target_flat = [t for batch in target for t in batch]
    functional_result = mean_average_precision(
        preds_flat, target_flat, backend=backend, iou_type="bbox", box_format="xyxy"
    )
    compare_with_class(
        functional_result,
        preds_flat,
        target_flat,
        backend=backend,
        iou_type="bbox",
        box_format="xyxy",
    )


@pytest.mark.parametrize("backend", ["pycocotools", "faster_coco_eval"])
def test_mean_average_precision_empty_preds_functional(backend):
    """When there are no predictions at all but targets are available."""
    preds = [
        {
            "boxes": paddle.Tensor([]),
            "scores": paddle.Tensor([]),
            "labels": paddle.tensor([], dtype=paddle.int64),
        }
    ]
    target = [
        {
            "boxes": paddle.Tensor([[214.15, 41.29, 562.41, 285.07]]),
            "labels": paddle.tensor([4], dtype=paddle.int64),
        }
    ]
    functional_result = mean_average_precision(
        preds, target, backend=backend, iou_type="bbox", box_format="xywh"
    )
    compare_with_class(
        functional_result,
        preds,
        target,
        backend=backend,
        iou_type="bbox",
        box_format="xywh",
    )


@pytest.mark.parametrize("backend", ["pycocotools", "faster_coco_eval"])
def test_mean_average_precision_empty_targets_functional(backend):
    """When there are no ground truths."""
    preds = [
        {
            "boxes": paddle.Tensor([[214.15, 41.29, 562.41, 285.07]]),
            "scores": paddle.Tensor([0.5]),
            "labels": paddle.tensor([4], dtype=paddle.int64),
        }
    ]
    target = [
        {"boxes": paddle.Tensor([]), "labels": paddle.tensor([], dtype=paddle.int64)}
    ]
    functional_result = mean_average_precision(
        preds, target, backend=backend, iou_type="bbox", box_format="xywh"
    )
    compare_with_class(
        functional_result,
        preds,
        target,
        backend=backend,
        iou_type="bbox",
        box_format="xywh",
    )


@pytest.mark.parametrize("box_format", ["xyxy", "xywh", "cxcywh"])
@pytest.mark.parametrize("backend", ["pycocotools", "faster_coco_eval"])
def test_mean_average_precision_box_format_functional(box_format, backend):
    """Test that providing different box formats leads to the expected results."""
    predictions = [
        {
            "boxes": paddle.tensor([[0.5, 0.5, 1, 1]]),
            "scores": paddle.tensor([1.0]),
            "labels": paddle.tensor([0], dtype=paddle.int64),
        }
    ]
    targets = [
        {
            "boxes": paddle.tensor([[0, 0, 1, 1]]),
            "labels": paddle.tensor([0], dtype=paddle.int64),
        }
    ]
    functional_result = mean_average_precision(
        predictions,
        targets,
        iou_thresholds=[0.3, 0.4],
        backend=backend,
        box_format=box_format,
        iou_type="bbox",
    )
    compare_with_class(
        functional_result,
        predictions,
        targets,
        iou_thresholds=[0.3, 0.4],
        backend=backend,
        box_format=box_format,
        iou_type="bbox",
    )


@pytest.mark.parametrize("backend", ["pycocotools", "faster_coco_eval"])
def test_mean_average_precision_custom_thresholds_functional(backend):
    """Test that custom recall thresholds and a custom iou_thresholds."""
    preds = _inputs["preds"]
    target = _inputs["target"]
    preds_flat = [p for batch in preds for p in batch]
    target_flat = [t for batch in target for t in batch]
    functional_result = mean_average_precision(
        preds_flat,
        target_flat,
        iou_thresholds=[0.2, 0.7],
        rec_thresholds=[0.25, 0.5, 0.75],
        backend=backend,
        box_format="xyxy",
        iou_type="bbox",
    )
    compare_with_class(
        functional_result,
        preds_flat,
        target_flat,
        iou_thresholds=[0.2, 0.7],
        rec_thresholds=[0.25, 0.5, 0.75],
        backend=backend,
        box_format="xyxy",
        iou_type="bbox",
    )
