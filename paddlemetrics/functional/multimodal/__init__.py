from paddlemetrics.functional.multimodal.lve import lip_vertex_error
from paddlemetrics.utils.imports import _TRANSFORMERS_GREATER_EQUAL_4_10

__all__ = ["lip_vertex_error"]
if _TRANSFORMERS_GREATER_EQUAL_4_10:
    from paddlemetrics.functional.multimodal.clip_iqa import \
        clip_image_quality_assessment
    from paddlemetrics.functional.multimodal.clip_score import clip_score

    __all__ += ["clip_image_quality_assessment", "clip_score"]
