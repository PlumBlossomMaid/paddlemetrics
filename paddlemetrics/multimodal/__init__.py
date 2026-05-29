from paddlemetrics.multimodal.lve import LipVertexError
from paddlemetrics.utils.imports import _TRANSFORMERS_GREATER_EQUAL_4_10

__all__ = ["LipVertexError"]
if _TRANSFORMERS_GREATER_EQUAL_4_10:
    from paddlemetrics.multimodal.clip_iqa import CLIPImageQualityAssessment
    from paddlemetrics.multimodal.clip_score import CLIPScore

    __all__ += ["CLIPImageQualityAssessment", "CLIPScore"]
