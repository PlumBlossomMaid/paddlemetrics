import sys

import warnings
from typing import Union

import paddle
from typing_extensions import Literal

from paddlemetrics.utils.imports import _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    pass
_AVAILABLE_REGRESSOR_DATASETS = {"kadid10k": (1, 5), "koniq10k": (1, 100)}
_TYPE_REGRESSOR_DATASET = Literal["kadid10k", "koniq10k"]
_base_url = "https://github.com/miccunifi/ARNIQA/releases/download/weights"
if not (True and _TORCHVISION_AVAILABLE):
    __doctest_skip__ = ["arniqa"]


class _ARNIQA(paddle.nn.Layer):
    """Initializes a No-Reference Image Quality Assessment ARNIQA paddle.nn.Layer.

    Args:
        regressor_dataset: dataset used for training the regressor, choose between [``koniq10k``, ``kadid10k``]

    """

    def __init__(self, regressor_dataset: _TYPE_REGRESSOR_DATASET = "koniq10k") -> None:
        super().__init__()
        if not True:
            raise RuntimeError("ARNIQA metric requires PyTorch >= 2.2.0")
        if not _TORCHVISION_AVAILABLE:
            raise ModuleNotFoundError(
                "ARNIQA metric requires that torchvision is installed. Either install as `pip install paddlemetrics[image]` or `pip install torchvision`."
            )
        valid_regressor_datasets = _AVAILABLE_REGRESSOR_DATASETS.keys()
        if regressor_dataset not in valid_regressor_datasets:
            raise ValueError(
                f"Argument `regressor_dataset` must be one of {valid_regressor_datasets}, but got {regressor_dataset}."
            )
        self.regressor_dataset = regressor_dataset
        self.imagenet_norm_mean = [0.485, 0.456, 0.406]
        self.imagenet_norm_std = [0.229, 0.224, 0.225]
        encoder = paddle.vision.models.resnet50()
        self.feat_dim = encoder.fc.in_features
        encoder = paddle.nn.Sequential(*list(encoder.children())[:-1])
        self.encoder = encoder
        self.regressor = paddle.nn.Linear(self.feat_dim * 2, 1)
        self._load_weights()

        def _freeze(module: paddle.nn.Layer) -> None:
            module.eval()
            for p in module.parameters():
                p.stop_gradient = not False

        _freeze(self.encoder)
        _freeze(self.regressor)

    def _load_weights(self) -> None:
        """Loads the weights of the encoder and regressor."""
        encoder_state_dict = paddle.hub.load_state_dict_from_url(
            url=f"{_base_url}/ARNIQA.pth", map_location="cpu"
        )
        filtered_encoder_state_dict = {
            k.replace("model.", ""): v
            for k, v in encoder_state_dict.items()
            if "projector" not in k
        }
        self.encoder.load_state_dict(filtered_encoder_state_dict, strict=True)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, module="paddle.serialization"
            )
            regressor_state_dict = paddle.hub.load_state_dict_from_url(
                url=f"{_base_url}/regressor_{self.regressor_dataset}.pth",
                map_location="cpu",
            ).state_dict()
            regressor_state_dict["weight"] = regressor_state_dict.pop("weights")
            regressor_state_dict["bias"] = regressor_state_dict.pop("biases").unsqueeze(
                0
            )
            self.regressor.load_state_dict(regressor_state_dict, strict=True)

    def _preprocess_input(
        self, img: paddle.Tensor, normalize: bool = False
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        """Preprocesses the input to the model.

        Obtains the half-scale version of the input image and applies normalization if needed.

        """
        h, w = img.shape[-2:]
        img_ds = paddle.vision.transforms.Resize(size=(h // 2, w // 2))(img)
        if normalize:
            img = paddle.vision.transforms.Normalize(
                mean=self.imagenet_norm_mean, std=self.imagenet_norm_std
            )(img)
            img_ds = paddle.vision.transforms.Normalize(
                mean=self.imagenet_norm_mean, std=self.imagenet_norm_std
            )(img_ds)
        return img, img_ds

    def _scale_score(self, score: paddle.Tensor) -> paddle.Tensor:
        """Scales the quality score to be in the [0, 1] range, where higher is better."""
        min_score, max_score = _AVAILABLE_REGRESSOR_DATASETS[self.regressor_dataset]
        return (score - min_score) / (max_score - min_score)

    def forward(self, img: paddle.Tensor, normalize: bool = False) -> paddle.Tensor:
        img, img_ds = self._preprocess_input(img, normalize)
        img_f = self.encoder(img)
        img_f = img_f.view(-1, self.feat_dim)
        img_f = paddle.nn.functional.normalize(img_f, axis=1)
        img_ds_f = self.encoder(img_ds)
        img_ds_f = img_ds_f.view(-1, self.feat_dim)
        img_ds_f = paddle.nn.functional.normalize(img_ds_f, axis=1)
        f = paddle.hstack(x=(img_f, img_ds_f))
        score = self.regressor(f)
        return self._scale_score(score)


class _NoTrainArniqa(_ARNIQA):
    """Wrapper to make sure ARNIQA never leaves evaluation mode."""

    def train(self, mode: bool) -> "_NoTrainArniqa":
        """Force network to always be in evaluation mode."""
        return super().train(False)


def _arniqa_update(
    img: paddle.Tensor, model: paddle.nn.Layer, normalize: bool, autocast: bool = False
) -> tuple[paddle.Tensor, Union[int, paddle.Tensor]]:
    """Update step for ARNIQA metric.

    Args:
        img: the input image
        model: the pre-trained model
        normalize: boolean indicating whether the input image is normalized
        autocast: boolean indicating whether to use automatic mixed precision

    """
    if not (img.ndim == 4 and img.shape[1] == 3):
        raise ValueError(
            f"Input image must have shape [N, 3, H, W]. Got input with shape {img.shape}."
        )
    if not (img._max() <= 1.0 and img._min() >= 0.0) and normalize:
        raise ValueError(
            f"Input image values must be in the [0, 1] range when normalize==True. Got input with values in range {img._min()} and {img._max()}."
        )
    if autocast:
        with paddle.amp.autocast(device_type=img.device.type, dtype=img.dtype):
            loss = model(img, normalize=normalize)
    else:
        loss = model.to(dtype=img.dtype)(img, normalize=normalize)
    return loss.squeeze(), img.shape[0]


def _arniqa_compute(
    scores: paddle.Tensor,
    num_scores: Union[paddle.Tensor, int],
    reduction: Literal["sum", "mean", "none"] = "mean",
) -> paddle.Tensor:
    """Compute step for ARNIQA metric."""
    sum_scores = scores.sum()
    if reduction == "none":
        return scores
    if reduction == "mean":
        return sum_scores / num_scores
    return sum_scores


def arniqa(
    img: paddle.Tensor,
    regressor_dataset: _TYPE_REGRESSOR_DATASET = "koniq10k",
    reduction: Literal["sum", "mean", "none"] = "mean",
    normalize: bool = True,
    autocast: bool = False,
) -> paddle.Tensor:
    """ARNIQA: leArning distoRtion maNifold for Image Quality Assessment metric.

    `ARNIQA`_ is a No-Reference Image Quality Assessment metric that predicts the technical quality of an image with
    a high correlation with human opinions. ARNIQA consists of an encoder and a regressor. The encoder is a ResNet-50
    model trained in a self-supervised way to model the image distortion manifold to generate similar representation for
    images with similar distortions, regardless of the image content. The regressor is a linear model trained on IQA
    datasets using the ground-truth quality scores. ARNIQA extracts the features from the full- and half-scale versions
    of the input image and then outputs a quality score in the [0, 1] range, where higher is better.

    The input image is expected to have shape ``(N, 3, H, W)``. The image should be in the [0, 1] range if `normalize`
    is set to ``True``, otherwise it should be normalized with the ImageNet mean and standard deviation.

    .. note::
        Using this metric requires you to have ``torchvision`` package installed. Either install as
        ``pip install paddlemetrics[image]`` or ``pip install torchvision``.

    Args:
        img: the input image
        regressor_dataset: dataset used for training the regressor. Choose between [``koniq10k``, ``kadid10k``].
            ``koniq10k`` corresponds to the `KonIQ-10k`_ dataset, which consists of real-world images with authentic
            distortions. ``kadid10k`` corresponds to the `KADID-10k`_ dataset, which consists of images with
            synthetically generated distortions.
        reduction: indicates how to reduce over the batch dimension. Choose between [``sum``, ``mean``, ``none``].
        normalize: by default this is ``True`` meaning that the input is expected to be in the [0, 1] range. If set
            to ``False`` will instead expect input to be already normalized with the ImageNet mean and standard
            deviation.
        autocast: boolean indicating whether to use automatic mixed precision

    Returns:
        A tensor in the [0, 1] range, where higher is better, representing the ARNIQA score of the input image. If
        `reduction` is set to ``none``, the output will have shape ``(N,)``, otherwise it will be a scalar tensor.

    Raises:
        ModuleNotFoundError:
            If ``torchvision`` package is not installed
        ValueError:
            If ``regressor_dataset`` is not in [``"kadid10k"``, ``"koniq10k"``]
        ValueError:
            If ``reduction`` is not in [``"sum"``, ``"mean"``, ``"none"``]
        ValueError:
            If ``normalize`` is not a bool
        ValueError:
            If the input image is not a valid image tensor with shape [N, 3, H, W].
        ValueError:
            If the input image values are not in the [0, 1] range when ``normalize`` is set to ``True``

    Examples:
        >>> from paddle import rand
        >>> from paddlemetrics.functional.image.arniqa import arniqa
        >>> img = rand(8, 3, 224, 224)
        >>> # Non-normalized input
        >>> arniqa(img, regressor_dataset='koniq10k', normalize=True)
        tensor(0.5308)


        >>> from paddle import rand
        >>> from paddlemetrics.functional.image.arniqa import arniqa
        >>> from torchvision.transforms import Normalize
        >>> img = rand(8, 3, 224, 224)
        >>> img = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        >>> # Normalized input
        >>> arniqa(img, regressor_dataset='koniq10k', normalize=False)
        tensor(0.5065)

    """
    valid_reduction = "mean", "sum", "none"
    if reduction not in valid_reduction:
        raise ValueError(
            f"Argument `reduction` must be one of {valid_reduction}, but got {reduction}"
        )
    if not isinstance(normalize, bool):
        raise ValueError(f"Argument `normalize` should be a bool but got {normalize}")
    model = _NoTrainArniqa(regressor_dataset=regressor_dataset).to(
        device=img.device, dtype=img.dtype
    )
    loss, num_scores = _arniqa_update(
        img, model, normalize=normalize, autocast=autocast
    )
    return _arniqa_compute(loss, num_scores, reduction)
