from paddlemetrics.utils.imports import _TORCH_VMAF_AVAILABLE

__all__ = []
if _TORCH_VMAF_AVAILABLE:
    from paddlemetrics.video.vmaf import VideoMultiMethodAssessmentFusion

    __all__ += ["VideoMultiMethodAssessmentFusion"]
