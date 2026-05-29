from paddlemetrics.functional.detection.panoptic_qualities import (
    modified_panoptic_quality, panoptic_quality)
from paddlemetrics.utils.imports import _TORCHVISION_AVAILABLE

__all__ = ["modified_panoptic_quality", "panoptic_quality"]
if _TORCHVISION_AVAILABLE:
    from paddlemetrics.functional.detection.ciou import \
        complete_intersection_over_union
    from paddlemetrics.functional.detection.diou import \
        distance_intersection_over_union
    from paddlemetrics.functional.detection.giou import \
        generalized_intersection_over_union
    from paddlemetrics.functional.detection.iou import intersection_over_union
    from paddlemetrics.functional.detection.map import mean_average_precision

    __all__ += [
        "complete_intersection_over_union",
        "distance_intersection_over_union",
        "generalized_intersection_over_union",
        "intersection_over_union",
        "mean_average_precision",
    ]
