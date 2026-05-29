from typing import TYPE_CHECKING, Any, Callable, List, Union, cast

import paddle
from paddle import Tensor
import paddleformers
from typing_extensions import Literal

from paddlemetrics.utils import rank_zero_warn
from paddlemetrics.utils.checks import (_SKIP_SLOW_DOCTEST,
                                           _try_proceed_with_timeout)
from paddlemetrics.utils.imports import _TRANSFORMERS_GREATER_EQUAL_4_10

if TYPE_CHECKING and _TRANSFORMERS_GREATER_EQUAL_4_10:
    pass
if _SKIP_SLOW_DOCTEST and _TRANSFORMERS_GREATER_EQUAL_4_10:
    pass

    def _download_clip_for_clip_score() -> None:
        pass
else:
    __doctest_skip__ = ["clip_score"]
    _CLIPModel = None
    _CLIPProcessor = None


class JinaProcessorWrapper:
    """Wrapper class to convert tensors to PIL images if needed for Jina CLIP model."""

def _detect_modality(
    input_data: Union[paddle.Tensor, List[paddle.Tensor], List[str], str]
) -> Literal["image", "text"]:
    """Automatically detect the modality of the input data.

    Args:
        input_data: Input data that can be either image tensors or text strings

    Returns:
        str: Either "image" or "text"

    Raises:
        ValueError: If the input_data is an empty list or modality cannot be determined

    """
    if isinstance(input_data, paddle.Tensor):
        return "image"
    if isinstance(input_data, list):
        if len(input_data) == 0:
            raise ValueError("Empty input list")
        if isinstance(input_data[0], paddle.Tensor):
            return "image"
        if isinstance(input_data[0], str):
            return "text"
    if isinstance(input_data, str):
        return "text"
    raise ValueError("Could not automatically determine modality for input_data")


def _process_image_data(
    images: Union[paddle.Tensor, List[paddle.Tensor]]
) -> List[paddle.Tensor]:
    """Helper function to process image data."""
    images = (
        [images] if not isinstance(images, list) and images.ndim == 3 else list(images)
    )
    if not all(i.ndim == 3 for i in images):
        raise ValueError(
            "Expected all images to be 3d but found image that has either more or less"
        )
    return images


def _process_text_data(texts: Union[str, List[str]]) -> List[str]:
    """Helper function to process text data."""
    if not isinstance(texts, list):
        texts = [texts]
    return texts


def _get_features(
    data: List[Union[paddle.Tensor, str]],
    modality: str,
    device: paddle.device,
    model: "_CLIPModel",
    processor: "_CLIPProcessor") -> paddle.Tensor:
    """Get features from the CLIP model for either images or text.

    Args:
       data: List of input data (images or text)
       modality: String indicating the type of input data (must be either "image" or "text")
       device: Device to run the model on
       model: CLIP model instance
       processor: CLIP processor instance

    Returns:
       Tensor of features from the CLIP model

    Raises:
        ValueError: If modality is not "image" or "text"

    """
    if modality == "image":
        image_data = [i for i in data if isinstance(i, paddle.Tensor)]
        processed = processor(
            images=[i.cpu() for i in image_data], return_tensors="pt", padding=True
        )
        return model.get_image_features(processed["pixel_values"].to(device))
    if modality == "text":
        processed = processor(text=data, return_tensors="pt", padding=True)
        if hasattr(model.config, "text_config") and hasattr(
            model.config.text_config, "max_position_embeddings"
        ):
            max_position_embeddings = model.config.text_config.max_position_embeddings
            if processed["attention_mask"].shape[-1] > max_position_embeddings:
                rank_zero_warn(
                    f"Encountered caption longer than max_position_embeddings={max_position_embeddings!r}. Will truncate captions to thislength. If longer captions are needed, initialize argument `model_name_or_path` with a model thatsupports longer sequences.",
                    UserWarning)
                processed["attention_mask"] = processed["attention_mask"][
                    ..., :max_position_embeddings
                ]
                processed["input_ids"] = processed["input_ids"][
                    ..., :max_position_embeddings
                ]
        return model.get_text_features(
            processed["input_ids"].to(device), processed["attention_mask"].to(device)
        )
    raise ValueError(f"invalid modality {modality}")


def _clip_score_update(
    source: Union[paddle.Tensor, List[paddle.Tensor], List[str], str],
    target: Union[paddle.Tensor, List[paddle.Tensor], List[str], str]) -> tuple[paddle.Tensor, int]:
    """Update function for CLIP Score."""
    source_modality = _detect_modality(source)
    target_modality = _detect_modality(target)
    source_data = (
        _process_image_data(cast(Union[paddle.Tensor, List[paddle.Tensor]], source))
        if source_modality == "image"
        else _process_text_data(cast(Union[str, List[str]], source))
    )
    target_data = (
        _process_image_data(cast(Union[paddle.Tensor, List[paddle.Tensor]], target))
        if target_modality == "image"
        else _process_text_data(cast(Union[str, List[str]], target))
    )
    if len(source_data) != len(target_data):
        raise ValueError(
            f"Expected the number of source and target examples to be the same but got {len(source_data)} and {len(target_data)}"
        )
    device = (
        source_data[0].device
        if source_modality == "image" and isinstance(source_data[0], paddle.Tensor)
        else target_data[0].device
        if target_modality == "image" and isinstance(target_data[0], paddle.Tensor)
        else paddle.device("cuda" if paddle.cuda.is_available() else "cpu")
    )
    model = model.to(device)
    source_features = _get_features(
        cast(List[Union[paddle.Tensor, str]], source_data),
        source_modality,
        device,
        model,
        processor)
    target_features = _get_features(
        cast(List[Union[paddle.Tensor, str]], target_data),
        target_modality,
        device,
        model,
        processor)
    source_features = source_features / source_features.norm(p=2, axis=-1, keepdim=True)
    target_features = target_features / target_features.norm(p=2, axis=-1, keepdim=True)
    score = 100 * (source_features * target_features).sum(axis=-1)
    score = (
        score.cpu()
        if source_modality == "text" and target_modality == "text"
        else score
    )
    return score, len(source_data)


def _get_clip_model_and_processor(
    model_name_or_path: str = "openai/clip-vit-base-patch32",
) -> Any:
    """TODO: implement for paddle."""
    pass

def clip_score(
    source: Union[paddle.Tensor, List[paddle.Tensor], List[str], str],
    target: Union[paddle.Tensor, List[paddle.Tensor], List[str], str],
    model_name_or_path: str = "openai/clip-vit-base-patch32",
) -> paddle.Tensor:
    pass  # TODO: implement clip_score for paddle
