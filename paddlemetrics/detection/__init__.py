from paddlemetrics.detection.panoptic_qualities import (ModifiedPanopticQuality,
                                                       PanopticQuality)
from paddlemetrics.utils.imports import _TORCHVISION_AVAILABLE

__all__ = ["ModifiedPanopticQuality", "PanopticQuality"]
if _TORCHVISION_AVAILABLE:
    from paddlemetrics.detection.ciou import CompleteIntersectionOverUnion
    from paddlemetrics.detection.diou import DistanceIntersectionOverUnion
    from paddlemetrics.detection.giou import GeneralizedIntersectionOverUnion
    from paddlemetrics.detection.iou import IntersectionOverUnion
    from paddlemetrics.detection.mean_ap import MeanAveragePrecision

    __all__ += [
        "CompleteIntersectionOverUnion",
        "DistanceIntersectionOverUnion",
        "GeneralizedIntersectionOverUnion",
        "IntersectionOverUnion",
        "MeanAveragePrecision",
    ]
