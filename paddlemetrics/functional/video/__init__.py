from paddlemetrics.utils.imports import _TORCH_VMAF_AVAILABLE

__all__ = []
if _TORCH_VMAF_AVAILABLE:
    from paddlemetrics.functional.video.vmaf import \
        video_multi_method_assessment_fusion

    __all__ += ["video_multi_method_assessment_fusion"]
