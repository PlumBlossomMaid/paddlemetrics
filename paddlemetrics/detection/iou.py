from collections.abc import Sequence
from typing import Any, List, Optional, Union

import paddle

from paddlemetrics.detection.helpers import _fix_empty_tensors, _input_validator
from paddlemetrics.functional.detection.iou import _iou_compute, _iou_update
from paddlemetrics.metric import Metric
from paddlemetrics.utils.data import dim_zero_cat
from paddlemetrics.utils.imports import (_MATPLOTLIB_AVAILABLE,
                                            _TORCHVISION_AVAILABLE)
from paddlemetrics.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _TORCHVISION_AVAILABLE:
    __doctest_skip__ = ["IntersectionOverUnion", "IntersectionOverUnion.plot"]
elif not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["IntersectionOverUnion.plot"]


class IntersectionOverUnion(Metric):
    """Computes Intersection Over Union (IoU).

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~List`): A list consisting of dictionaries each containing the key-values
      (each dictionary corresponds to a single image). Parameters that should be provided per dict:

        - ``boxes`` (:class:`~paddle.Tensor`): float tensor of shape ``(num_boxes, 4)`` containing ``num_boxes``
          detection boxes of the format specified in the constructor.
          By default, this method expects ``(xmin, ymin, xmax, ymax)`` in absolute image coordinates.
        - labels: ``IntTensor`` of shape ``(num_boxes)`` containing 0-indexed detection classes for
          the boxes.

    - ``target`` (:class:`~List`): A list consisting of dictionaries each containing the key-values
      (each dictionary corresponds to a single image). Parameters that should be provided per dict:

        - ``boxes`` (:class:`~paddle.Tensor`): float tensor of shape ``(num_boxes, 4)`` containing ``num_boxes`` ground
          truth boxes of the format specified in the constructor.
          By default, this method expects ``(xmin, ymin, xmax, ymax)`` in absolute image coordinates.
        - ``labels`` (:class:`~paddle.Tensor`): integer tensor of shape ``(num_boxes)`` containing 0-indexed ground truth
          classes for the boxes.

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``iou_dict``: A dictionary containing the following key-values:

        - iou: (:class:`~paddle.Tensor`)
        - iou/cl_{cl}: (:class:`~paddle.Tensor`), if argument ``class metrics=True``

    Args:
        box_format:
            Input format of given boxes. Supported formats are ``[`xyxy`, `xywh`, `cxcywh`]``.
        iou_thresholds:
            Optional IoU thresholds for evaluation. If set to `None` the threshold is ignored.
        class_metrics:
            Option to enable per-class metrics for IoU. Has a performance impact.
        respect_labels:
            Ignore values from boxes that do not have the same label as the ground truth box. Else will compute Iou
                between all pairs of boxes.
        kwargs:
            Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example::

        >>> import paddle
        >>> from paddlemetrics.detection import IntersectionOverUnion
        >>> preds = [
        ...    {
        ...        "boxes": paddle.to_tensor([
        ...             [296.55, 93.96, 314.97, 152.79],
        ...             [298.55, 98.96, 314.97, 151.79]]),
        ...        "labels": paddle.to_tensor([4, 5]),
        ...    }
        ... ]
        >>> target = [
        ...    {
        ...        "boxes": paddle.to_tensor([[300.00, 100.00, 315.00, 150.00]]),
        ...        "labels": paddle.to_tensor([5]),
        ...    }
        ... ]
        >>> metric = IntersectionOverUnion()
        >>> metric(preds, target)
        {'iou': tensor(0.8614)}

    Example::

        The metric can also return the score per class:

        >>> import paddle
        >>> from paddlemetrics.detection import IntersectionOverUnion
        >>> preds = [
        ...    {
        ...        "boxes": paddle.to_tensor([
        ...             [296.55, 93.96, 314.97, 152.79],
        ...             [298.55, 98.96, 314.97, 151.79]]),
        ...        "labels": paddle.to_tensor([4, 5]),
        ...    }
        ... ]
        >>> target = [
        ...    {
        ...        "boxes": paddle.to_tensor([
        ...               [300.00, 100.00, 315.00, 150.00],
        ...               [300.00, 100.00, 315.00, 150.00]
        ...        ]),
        ...        "labels": paddle.to_tensor([4, 5]),
        ...    }
        ... ]
        >>> metric = IntersectionOverUnion(class_metrics=True)
        >>> metric(preds, target)
        {'iou': tensor(0.7756), 'iou/cl_4': tensor(0.6898), 'iou/cl_5': tensor(0.8614)}

    Raises:
        ModuleNotFoundError:
            If torchvision is not installed with version 0.8.0 or newer.

    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = True
    groundtruth_labels: List[paddle.Tensor]
    pred_labels: List[paddle.Tensor]
    iou_matrix: List[paddle.Tensor]
    _iou_type: str = "iou"
    _invalid_val: float = -1.0

    def __init__(
        self,
        box_format: str = "xyxy",
        iou_threshold: Optional[float] = None,
        class_metrics: bool = False,
        respect_labels: bool = True,
        **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not _TORCHVISION_AVAILABLE:
            raise ModuleNotFoundError(
                f"Metric `{self._iou_type.upper()}` requires that `torchvision` is installed. Please install with `pip install paddlemetrics[detection]`."
            )
        allowed_box_formats = "xyxy", "xywh", "cxcywh"
        if box_format not in allowed_box_formats:
            raise ValueError(
                f"Expected argument `box_format` to be one of {allowed_box_formats} but got {box_format}"
            )
        self.box_format = box_format
        self.iou_threshold = iou_threshold
        if not isinstance(class_metrics, bool):
            raise ValueError("Expected argument `class_metrics` to be a boolean")
        self.class_metrics = class_metrics
        if not isinstance(respect_labels, bool):
            raise ValueError("Expected argument `respect_labels` to be a boolean")
        self.respect_labels = respect_labels
        self.add_state("groundtruth_labels", default=[], dist_reduce_fx=None)
        self.add_state("pred_labels", default=[], dist_reduce_fx=None)
        self.add_state("iou_matrix", default=[], dist_reduce_fx=None)

    @staticmethod
    def _iou_update_fn(*args: Any, **kwargs: Any) -> paddle.Tensor:
        return _iou_update(*args, **kwargs)

    @staticmethod
    def _iou_compute_fn(*args: Any, **kwargs: Any) -> paddle.Tensor:
        return _iou_compute(*args, **kwargs)

    def update(
        self,
        preds: list[dict[str, paddle.Tensor]],
        target: list[dict[str, paddle.Tensor]]) -> None:
        """Update state with predictions and targets."""
        _input_validator(preds, target, ignore_score=True)
        for p_i, t_i in zip(preds, target):
            det_boxes = self._get_safe_item_values(p_i["boxes"])
            gt_boxes = self._get_safe_item_values(t_i["boxes"])
            self.groundtruth_labels.append(t_i["labels"])
            self.pred_labels.append(p_i["labels"])
            iou_matrix = self._iou_update_fn(
                det_boxes, gt_boxes, self.iou_threshold, self._invalid_val
            )
            if self.respect_labels:
                if det_boxes.size > 0 and gt_boxes.size > 0:
                    label_eq = p_i["labels"].unsqueeze(1) == t_i["labels"].unsqueeze(0)
                else:
                    label_eq = paddle.eye(
                        iou_matrix.shape[0], dtype=bool, device=iou_matrix.device
                    )
                iou_matrix[~label_eq] = self._invalid_val
            self.iou_matrix.append(iou_matrix)

    def _get_safe_item_values(self, boxes: paddle.Tensor) -> paddle.Tensor:
        boxes = _fix_empty_tensors(boxes)
        if boxes.size > 0:
            return boxes
        return boxes