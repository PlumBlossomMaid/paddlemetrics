import sys

import logging
from collections.abc import Sequence
from typing import Any, Callable, List, Literal, Optional, Union

import numpy as np
import paddle
from paddle import Tensor

from paddlemetrics.detection.helpers import _fix_empty_tensors, _input_validator
from paddlemetrics.metric import Metric
from paddlemetrics.utils.data import _cumsum
from paddlemetrics.utils.imports import (_MATPLOTLIB_AVAILABLE,
                                            _PYCOCOTOOLS_AVAILABLE,
                                            _TORCHVISION_AVAILABLE)
from paddlemetrics.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["MeanAveragePrecision.plot"]
if not _TORCHVISION_AVAILABLE or not _PYCOCOTOOLS_AVAILABLE:
    __doctest_skip__ = ["MeanAveragePrecision.plot", "MeanAveragePrecision"]
log = logging.getLogger(__name__)


def compute_area(
    inputs: list[Any], iou_type: Literal["bbox", "segm"] = "bbox"
) -> paddle.Tensor:
    """Compute area of input depending on the specified iou_type.

    Default output for empty input is :class:`~paddle.Tensor`

    """
    import pycocotools.mask as mask_utils

    if len(inputs) == 0:
        return paddle.Tensor([])
    if iou_type == "bbox":
        pass  # TODO: fix removed code block
def compute_iou(
    det: list[Any], gt: list[Any], iou_type: Literal["bbox", "segm"] = "bbox"
) -> paddle.Tensor:
    """Compute IOU between detections and ground-truth using the specified iou_type."""
    if iou_type == "bbox":
        pass  # TODO: fix removed code block
class BaseMetricResults(dict):
    """Base metric class, that allows fields for pre-defined metrics."""

    def __getattr__(self, key: str) -> paddle.Tensor:
        """Get a specific metric attribute."""
        if key in self:
            return self[key]
        raise AttributeError(f"No such attribute: {key}")

    def __setattr__(self, key: str, value: paddle.Tensor) -> None:
        """Set a specific metric attribute."""
        self[key] = value

    def __delattr__(self, key: str) -> None:
        """Delete a specific metric attribute."""
        if key in self:
            del self[key]
        raise AttributeError(f"No such attribute: {key}")


class MAPMetricResults(BaseMetricResults):
    """Class to wrap the final mAP results."""

    __slots__ = (
        "classes",
        "map",
        "map_50",
        "map_75",
        "map_large",
        "map_medium",
        "map_small")


class MARMetricResults(BaseMetricResults):
    """Class to wrap the final mAR results."""

    __slots__ = ("mar_1", "mar_10", "mar_100", "mar_large", "mar_medium", "mar_small")


class COCOMetricResults(BaseMetricResults):
    """Class to wrap the final COCO metric results including various mAP/mAR values."""

    __slots__ = (
        "map",
        "map_50",
        "map_75",
        "map_large",
        "map_medium",
        "map_per_class",
        "map_small",
        "mar_1",
        "mar_10",
        "mar_100",
        "mar_100_per_class",
        "mar_large",
        "mar_medium",
        "mar_small")


def _segm_iou(
    det: list[tuple[np.ndarray, np.ndarray]], gt: list[tuple[np.ndarray, np.ndarray]]
) -> paddle.Tensor:
    """Compute IOU between detections and ground-truths using mask-IOU.

    Implementation is based on pycocotools toolkit for mask_utils.

    Args:
       det: A list of detection masks as ``[(RLE_SIZE, RLE_COUNTS)]``, where ``RLE_SIZE`` is (width, height) dimension
           of the input and RLE_COUNTS is its RLE representation;

       gt: A list of ground-truth masks as ``[(RLE_SIZE, RLE_COUNTS)]``, where ``RLE_SIZE`` is (width, height) dimension
           of the input and RLE_COUNTS is its RLE representation;

    """
    import pycocotools.mask as mask_utils

    det_coco_format = [{"size": i[0], "counts": i[1]} for i in det]
    gt_coco_format = [{"size": i[0], "counts": i[1]} for i in gt]
    return paddle.tensor(
        mask_utils.iou(det_coco_format, gt_coco_format, [(False) for _ in gt])
    )


class MeanAveragePrecision(Metric):
    """Compute the `Mean-Average-Precision (mAP) and Mean-Average-Recall (mAR)`_ for object detection predictions.

    .. math::
        \\text{mAP} = \\frac{1}{n} \\sum_{i=1}^{n} AP_i

    where :math:`AP_i` is the average precision for class :math:`i` and :math:`n` is the number of classes. The average
    precision is defined as the area under the precision-recall curve. If argument `class_metrics` is set to ``True``,
    the metric will also return the mAP/mAR per class.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~List`): A list consisting of dictionaries each containing the key-values
      (each dictionary corresponds to a single image). Parameters that should be provided per dict

        - boxes: (:class:`~paddle.FloatTensor`) of shape ``(num_boxes, 4)`` containing ``num_boxes`` detection
          boxes of the format specified in the constructor.
          By default, this method expects ``(xmin, ymin, xmax, ymax)`` in absolute image coordinates.
        - scores: :class:`~paddle.FloatTensor` of shape ``(num_boxes)`` containing detection scores for the boxes.
        - labels: :class:`~paddle.IntTensor` of shape ``(num_boxes)`` containing 0-indexed detection classes for
          the boxes.
        - masks: :class:`~paddle.bool` of shape ``(num_boxes, image_height, image_width)`` containing boolean masks.
          Only required when `iou_type="segm"`.

    - ``target`` (:class:`~List`) A list consisting of dictionaries each containing the key-values
      (each dictionary corresponds to a single image). Parameters that should be provided per dict:

        - boxes: :class:`~paddle.FloatTensor` of shape ``(num_boxes, 4)`` containing ``num_boxes`` ground truth
          boxes of the format specified in the constructor.
          By default, this method expects ``(xmin, ymin, xmax, ymax)`` in absolute image coordinates.
        - labels: :class:`~paddle.IntTensor` of shape ``(num_boxes)`` containing 0-indexed ground truth
          classes for the boxes.
        - masks: :class:`~paddle.bool` of shape ``(num_boxes, image_height, image_width)`` containing boolean masks.
          Only required when `iou_type="segm"`.

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``map_dict``: A dictionary containing the following key-values:

        - map: (:class:`~paddle.Tensor`)
        - map_small: (:class:`~paddle.Tensor`)
        - map_medium:(:class:`~paddle.Tensor`)
        - map_large: (:class:`~paddle.Tensor`)
        - mar_1: (:class:`~paddle.Tensor`)
        - mar_10: (:class:`~paddle.Tensor`)
        - mar_100: (:class:`~paddle.Tensor`)
        - mar_small: (:class:`~paddle.Tensor`)
        - mar_medium: (:class:`~paddle.Tensor`)
        - mar_large: (:class:`~paddle.Tensor`)
        - map_50: (:class:`~paddle.Tensor`) (-1 if 0.5 not in the list of iou thresholds)
        - map_75: (:class:`~paddle.Tensor`) (-1 if 0.75 not in the list of iou thresholds)
        - map_per_class: (:class:`~paddle.Tensor`) (-1 if class metrics are disabled)
        - mar_100_per_class: (:class:`~paddle.Tensor`) (-1 if class metrics are disabled)
        - classes (:class:`~paddle.Tensor`)

    For an example on how to use this metric check the `paddlemetrics mAP example`_.

    .. attention::
        The ``map`` score is calculated with @[ IoU=self.iou_thresholds | area=all | max_dets=max_detection_thresholds ]
        **Caution:** If the initialization parameters are changed, dictionary keys for mAR can change as well.
        The default properties are also accessible via fields and will raise an ``AttributeError`` if not available.

    .. important::
        This metric is following the mAP implementation of `pycocotools`_ a standard implementation for the mAP metric
        for object detection.

    .. hint::
        This metric requires you to have `torchvision` version 0.8.0 or newer installed
        (with corresponding version 1.7.0 of torch or newer). This metric requires `pycocotools`
        installed when iou_type is `segm`. Please install with ``pip install torchvision`` or
        ``pip install paddlemetrics[detection]``.

    Args:
        box_format:
            Input format of given boxes. Supported formats are ``[`xyxy`, `xywh`, `cxcywh`]``.
        iou_type:
            Type of input (either masks or bounding-boxes) used for computing IOU.
            Supported IOU types are ``["bbox", "segm"]``.
            If using ``"segm"``, masks should be provided (see :meth:`update`).
        iou_thresholds:
            IoU thresholds for evaluation. If set to ``None`` it corresponds to the stepped range ``[0.5,...,0.95]``
            with step ``0.05``. Else provide a list of floats.
        rec_thresholds:
            Recall thresholds for evaluation. If set to ``None`` it corresponds to the stepped range ``[0,...,1]``
            with step ``0.01``. Else provide a list of floats.
        max_detection_thresholds:
            Thresholds on max detections per image. If set to `None` will use thresholds ``[1, 10, 100]``.
            Else, please provide a list of ints.
        class_metrics:
            Option to enable per-class metrics for mAP and mAR_100. Has a performance impact.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ModuleNotFoundError:
            If ``torchvision`` is not installed or version installed is lower than 0.8.0
        ModuleNotFoundError:
            If ``iou_type`` is equal to ``segm`` and ``pycocotools`` is not installed
        ValueError:
            If ``class_metrics`` is not a boolean
        ValueError:
            If ``preds`` is not of type (:class:`~List[Dict[str, Tensor]]`)
        ValueError:
            If ``target`` is not of type ``List[Dict[str, Tensor]]``
        ValueError:
            If ``preds`` and ``target`` are not of the same length
        ValueError:
            If any of ``preds.boxes``, ``preds.scores`` and ``preds.labels`` are not of the same length
        ValueError:
            If any of ``target.boxes`` and ``target.labels`` are not of the same length
        ValueError:
            If any box is not type float and of length 4
        ValueError:
            If any class is not type int and of length 1
        ValueError:
            If any score is not type float and of length 1

    Example:
        >>> from paddle import tensor
        >>> from paddlemetrics.detection import MeanAveragePrecision
        >>> preds = [
        ...   dict(
        ...     boxes=tensor([[258.0, 41.0, 606.0, 285.0]]),
        ...     scores=tensor([0.536]),
        ...     labels=tensor([0]),
        ...   )
        ... ]
        >>> target = [
        ...   dict(
        ...     boxes=tensor([[214.0, 41.0, 562.0, 285.0]]),
        ...     labels=tensor([0]),
        ...   )
        ... ]
        >>> metric = MeanAveragePrecision()
        >>> metric.update(preds, target)
        >>> from pprint import pprint
        >>> pprint(metric.compute())
        {'classes': tensor(0, dtype=paddle.int32),
         'map': tensor(0.6000),
         'map_50': tensor(1.),
         'map_75': tensor(1.),
         'map_large': tensor(0.6000),
         'map_medium': tensor(-1.),
         'map_per_class': tensor(-1.),
         'map_small': tensor(-1.),
         'mar_1': tensor(0.6000),
         'mar_10': tensor(0.6000),
         'mar_100': tensor(0.6000),
         'mar_100_per_class': tensor(-1.),
         'mar_large': tensor(0.6000),
         'mar_medium': tensor(-1.),
         'mar_small': tensor(-1.)}

    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    detections: List[paddle.Tensor]
    detection_scores: List[paddle.Tensor]
    detection_labels: List[paddle.Tensor]
    groundtruths: List[paddle.Tensor]
    groundtruth_labels: List[paddle.Tensor]

    def __init__(
        self,
        box_format: str = "xyxy",
        iou_type: Literal["bbox", "segm"] = "bbox",
        iou_thresholds: Optional[list[float]] = None,
        rec_thresholds: Optional[list[float]] = None,
        max_detection_thresholds: Optional[list[int]] = None,
        class_metrics: bool = False,
        **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not _PYCOCOTOOLS_AVAILABLE:
            raise ModuleNotFoundError(
                "`MAP` metric requires that `pycocotools` installed. Please install with `pip install pycocotools` or `pip install paddlemetrics[detection]`"
            )
        if not _TORCHVISION_AVAILABLE:
            raise ModuleNotFoundError(
                "`MeanAveragePrecision` metric requires that `torchvision` is installed. Please install with `pip install paddlemetrics[detection]`."
            )
        allowed_box_formats = "xyxy", "xywh", "cxcywh"
        allowed_iou_types = "segm", "bbox"
        if box_format not in allowed_box_formats:
            raise ValueError(
                f"Expected argument `box_format` to be one of {allowed_box_formats} but got {box_format}"
            )
        self.box_format = box_format
        self.iou_thresholds = (
            iou_thresholds
            or paddle.linspace(0.5, 0.95, round((0.95 - 0.5) / 0.05) + 1).tolist()
        )
        self.rec_thresholds = (
            rec_thresholds or paddle.linspace(0.0, 1.0, round(1.0 / 0.01) + 1).tolist()
        )
        max_det_threshold, _ = paddle.sort(
            paddle.IntTensor(max_detection_thresholds or [1, 10, 100])
        )
        self.max_detection_thresholds = max_det_threshold.tolist()
        if iou_type not in allowed_iou_types:
            raise ValueError(
                f"Expected argument `iou_type` to be one of {allowed_iou_types} but got {iou_type}"
            )
        if iou_type == "segm" and not _PYCOCOTOOLS_AVAILABLE:
            raise ModuleNotFoundError(
                "When `iou_type` is set to 'segm', pycocotools need to be installed"
            )
        self.iou_type = iou_type
        self.bbox_area_ranges = {
            "all": (float(0**2), float(100000.0**2)),
            "small": (float(0**2), float(32**2)),
            "medium": (float(32**2), float(96**2)),
            "large": (float(96**2), float(100000.0**2)),
        }
        if not isinstance(class_metrics, bool):
            raise ValueError("Expected argument `class_metrics` to be a boolean")
        self.class_metrics = class_metrics
        self.add_state("detections", default=[], dist_reduce_fx=None)
        self.add_state("detection_scores", default=[], dist_reduce_fx=None)
        self.add_state("detection_labels", default=[], dist_reduce_fx=None)
        self.add_state("groundtruths", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_labels", default=[], dist_reduce_fx=None)

    def update(
        self,
        preds: list[dict[str, paddle.Tensor]],
        target: list[dict[str, paddle.Tensor]]) -> None:
        """Update state with predictions and targets."""
        _input_validator(preds, target, iou_type=self.iou_type)
        for item in preds:
            detections = self._get_safe_item_values(item)
            self.detections.append(detections)
            self.detection_labels.append(item["labels"])
            self.detection_scores.append(item["scores"])
        for item in target:
            groundtruths = self._get_safe_item_values(item)
            self.groundtruths.append(groundtruths)
            self.groundtruth_labels.append(item["labels"])

    def _move_list_states_to_cpu(self) -> None:
        """Move list states to cpu to save GPU memory."""
        for key in self._defaults:
            current_val = getattr(self, key)
            current_to_cpu = []
            if isinstance(current_val, Sequence):
                for cur_v in current_val:
                    if not isinstance(cur_v, tuple):
                        cur_v = cur_v.to("cpu")
                    current_to_cpu.append(cur_v)
            setattr(self, key, current_to_cpu)

    def _get_safe_item_values(
        self, item: dict[str, Any]
    ) -> Union[paddle.Tensor, tuple]:
        import pycocotools.mask as mask_utils

        if self.iou_type == "bbox":
            boxes = _fix_empty_tensors(item["boxes"])
            if boxes.size > 0:
                return boxes
            return boxes