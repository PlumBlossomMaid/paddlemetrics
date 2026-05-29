import sys

import inspect
import os
from typing import List, NamedTuple, Optional, Union

import paddle
from paddle import Tensor
from typing_extensions import Literal

from paddlemetrics.utils.imports import _TORCHVISION_AVAILABLE

_weight_map = {
    "squeezenet1_1": "SqueezeNet1_1_Weights",
    "alexnet": "AlexNet_Weights",
    "vgg16": "VGG16_Weights",
}
if not _TORCHVISION_AVAILABLE:
    __doctest_skip__ = [
        "learned_perceptual_image_patch_similarity",
        "_get_tv_model_features",
    ]


def _get_tv_model_features(
    net: str, pretrained: bool = False
) -> paddle.nn.Layer:
    """Get TV model features. Requires torchvision."""
    raise NotImplementedError("LPIPS requires torchvision which is not available in Paddle ecosystem")


class SqueezeNet(paddle.nn.Layer):
    """SqueezeNet implementation."""

    def __init__(self, requires_grad: bool = False, pretrained: bool = True) -> None:
        super().__init__()
        pretrained_features = _get_tv_model_features("squeezenet1_1", pretrained)
        self.N_slices = 7
        slices = []
        feature_ranges = [
            range(2),
            range(2, 5),
            range(5, 8),
            range(8, 10),
            range(10, 11),
            range(11, 12),
            range(12, 13),
        ]
        for feature_range in feature_ranges:
            seq = paddle.nn.Sequential()
            for i in feature_range:
                seq.add_module(str(i), pretrained_features[i])
            slices.append(seq)
        self.slices = paddle.nn.LayerList(slices)
        if not requires_grad:
            for param in self.parameters():
                param.stop_gradient = not False

    def forward(self, x: paddle.Tensor) -> NamedTuple:
        """Process input."""

        class _SqueezeOutput(NamedTuple):
            relu1: Tensor
            relu2: Tensor
            relu3: Tensor
            relu4: Tensor
            relu5: Tensor
            relu6: Tensor
            relu7: Tensor

        relus = []
        for slice_ in self.slices:
            x = slice_(x)
            relus.append(x)
        return _SqueezeOutput(*relus)


class Alexnet(paddle.nn.Layer):
    """Alexnet implementation."""

    def __init__(self, requires_grad: bool = False, pretrained: bool = True) -> None:
        super().__init__()
        alexnet_pretrained_features = _get_tv_model_features("alexnet", pretrained)
        self.slice1 = paddle.nn.Sequential()
        self.slice2 = paddle.nn.Sequential()
        self.slice3 = paddle.nn.Sequential()
        self.slice4 = paddle.nn.Sequential()
        self.slice5 = paddle.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.stop_gradient = not False

    def forward(self, x: paddle.Tensor) -> NamedTuple:
        """Process input."""
        h = self.slice1(x)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h

        class _AlexnetOutputs(NamedTuple):
            relu1: Tensor
            relu2: Tensor
            relu3: Tensor
            relu4: Tensor
            relu5: Tensor

        return _AlexnetOutputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)


class Vgg16(paddle.nn.Layer):
    """Vgg16 implementation."""

    def __init__(self, requires_grad: bool = False, pretrained: bool = True) -> None:
        super().__init__()
        vgg_pretrained_features = _get_tv_model_features("vgg16", pretrained)
        self.slice1 = paddle.nn.Sequential()
        self.slice2 = paddle.nn.Sequential()
        self.slice3 = paddle.nn.Sequential()
        self.slice4 = paddle.nn.Sequential()
        self.slice5 = paddle.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.stop_gradient = not False

    def forward(self, x: paddle.Tensor) -> NamedTuple:
        """Process input."""
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h

        class _VGGOutputs(NamedTuple):
            relu1_2: Tensor
            relu2_2: Tensor
            relu3_3: Tensor
            relu4_3: Tensor
            relu5_3: Tensor

        return _VGGOutputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)


def _spatial_average(in_tens: paddle.Tensor, keep_dim: bool = True) -> paddle.Tensor:
    """Spatial averaging over height and width of images."""
    return in_tens.mean([2, 3], keepdim=keep_dim)


def _upsample(
    in_tens: paddle.Tensor, out_hw: tuple[int, ...] = (64, 64)
) -> paddle.Tensor:
    """Upsample input with bilinear interpolation."""
    return paddle.nn.Upsample(size=out_hw, mode="bilinear", align_corners=False)(
        in_tens
    )


def _normalize_tensor(in_feat: paddle.Tensor, eps: float = 1e-08) -> paddle.Tensor:
    """Normalize input tensor."""
    norm_factor = paddle.sqrt(eps + paddle.sum(in_feat**2, axis=1, keepdim=True))
    return in_feat / norm_factor


def _resize_tensor(x: paddle.Tensor, size: int = 64) -> paddle.Tensor:
    """https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/sample_similarity_lpips.py#L127C22-L132."""
    if x.shape[-1] > size and x.shape[-2] > size:
        return paddle.nn.functional.interpolate(x, (size, size), mode="area")
    return paddle.nn.functional.interpolate(
        x, (size, size), mode="bilinear", align_corners=False
    )


class ScalingLayer(paddle.nn.Layer):
    """Scaling layer."""

    shift: Tensor
    scale: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "shift",
            paddle.Tensor([-0.03, -0.088, -0.188])[None, :, None, None],
            persistent=False,
        )
        self.register_buffer(
            "scale",
            paddle.Tensor([0.458, 0.448, 0.45])[None, :, None, None],
            persistent=False,
        )

    def forward(self, inp: paddle.Tensor) -> paddle.Tensor:
        """Process input."""
        return (inp - self.shift) / self.scale


class NetLinLayer(paddle.nn.Layer):
    """A single linear layer which does a 1x1 conv."""

    def __init__(
        self, chn_in: int, chn_out: int = 1, use_dropout: bool = False
    ) -> None:
        super().__init__()
        layers = [paddle.nn.Dropout()] if use_dropout else []
        layers += [
            paddle.nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)
        ]
        self.model = paddle.nn.Sequential(*layers)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Process input."""
        return self.model(x)


class _LPIPS(paddle.nn.Layer):
    def __init__(
        self,
        pretrained: bool = True,
        net: Literal["alex", "vgg", "squeeze"] = "alex",
        spatial: bool = False,
        pnet_rand: bool = False,
        pnet_tune: bool = False,
        use_dropout: bool = True,
        model_path: Optional[str] = None,
        eval_mode: bool = True,
        resize: Optional[int] = None,
    ) -> None:
        """Initializes a perceptual loss paddle.nn.Layer.

        Args:
            pretrained: This flag controls the linear layers should be pretrained version or random
            net: Indicate backbone to use, choose between ['alex','vgg','squeeze']
            spatial: If input should be spatial averaged
            pnet_rand: If backbone should be random or use imagenet pre-trained weights
            pnet_tune: If backprop should be enabled for both backbone and linear layers
            use_dropout: If dropout layers should be added
            model_path: Model path to load pretained models from
            eval_mode: If network should be in evaluation mode
            resize: If input should be resized to this size

        """
        super().__init__()
        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.resize = resize
        self.scaling_layer = ScalingLayer()
        if self.pnet_type in ["vgg", "vgg16"]:
            net_type = Vgg16
            self.chns = [64, 128, 256, 512, 512]
        elif self.pnet_type == "alex":
            net_type = Alexnet
            self.chns = [64, 192, 384, 256, 256]
        elif self.pnet_type == "squeeze":
            net_type = SqueezeNet
            self.chns = [64, 128, 256, 384, 384, 512, 512]
        self.L = len(self.chns)
        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        if self.pnet_type == "squeeze":
            self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
            self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
            self.lins += [self.lin5, self.lin6]
        self.lins = paddle.nn.LayerList(self.lins)
        if pretrained:
            if model_path is None:
                model_path = os.path.abspath(
                    os.path.join(
                        inspect.getfile(self.__init__), "..", f"lpips_models/{net}.pth"
                    )
                )
            self.load_state_dict(paddle.load(path=str(model_path)), strict=False)
        if eval_mode:
            self.eval()
        if not self.pnet_tune:
            for param in self.parameters():
                param.stop_gradient = not False

    def forward(
        self,
        in0: paddle.Tensor,
        in1: paddle.Tensor,
        retperlayer: bool = False,
        normalize: bool = False,
    ) -> Union[paddle.Tensor, tuple[paddle.Tensor, List[paddle.Tensor]]]:
        if normalize:
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1
        in0_input, in1_input = self.scaling_layer(in0), self.scaling_layer(in1)
        if self.resize is not None:
            in0_input = _resize_tensor(in0_input, size=self.resize)
            in1_input = _resize_tensor(in1_input, size=self.resize)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        for kk in range(self.L):
            feats0[kk], feats1[kk] = _normalize_tensor(outs0[kk]), _normalize_tensor(
                outs1[kk]
            )
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
        res = []
        for kk in range(self.L):
            if self.spatial:
                res.append(
                    _upsample(self.lins[kk](diffs[kk]), out_hw=tuple(in0.shape[2:]))
                )
            else:
                res.append(_spatial_average(self.lins[kk](diffs[kk]), keep_dim=True))
        val: Tensor = sum(res)
        if retperlayer:
            return val, res
        return val


class _NoTrainLpips(_LPIPS):
    """Wrapper to make sure LPIPS never leaves evaluation mode."""

    def train(self, mode: bool) -> "_NoTrainLpips":
        """Force network to always be in evaluation mode."""
        return super().train(False)


def _valid_img(img: paddle.Tensor, normalize: bool) -> bool:
    """Check that input is a valid image to the network."""
    value_check = (
        img._max() <= 1.0 and img._min() >= 0.0 if normalize else img._min() >= -1
    )
    return img.ndim == 4 and img.shape[1] == 3 and value_check


def _lpips_update(
    img1: paddle.Tensor, img2: paddle.Tensor, net: paddle.nn.Layer, normalize: bool
) -> paddle.Tensor:
    if not (_valid_img(img1, normalize) and _valid_img(img2, normalize)):
        raise ValueError(
            f"Expected both input arguments to be normalized tensors with shape [N, 3, H, W]. Got input with shape {img1.shape} and {img2.shape} and values in range {[img1._min(), img1._max()]} and {[img2._min(), img2._max()]} when all values are expected to be in the {[0, 1] if normalize else [-1, 1]} range."
        )
    return net(img1, img2, normalize=normalize).squeeze()


def _lpips_compute(
    scores: paddle.Tensor, reduction: Optional[Literal["sum", "mean", "none"]] = "mean"
) -> paddle.Tensor:
    if reduction == "mean":
        return scores.mean()
    if reduction == "sum":
        return scores.sum()
    if reduction == "none" or reduction is None:
        return scores
    raise ValueError(f"Invalid reduction type: {reduction}")


def learned_perceptual_image_patch_similarity(
    img1: paddle.Tensor,
    img2: paddle.Tensor,
    net_type: Literal["alex", "vgg", "squeeze"] = "alex",
    reduction: Optional[Literal["sum", "mean", "none"]] = "mean",
    normalize: bool = False,
) -> paddle.Tensor:
    """The Learned Perceptual Image Patch Similarity (`LPIPS_`) calculates perceptual similarity between two images.

    LPIPS essentially computes the similarity between the activations of two image patches for some pre-defined network.
    This measure has been shown to match human perception well. A low LPIPS score means that image patches are
    perceptual similar.

    Both input image patches are expected to have shape ``(N, 3, H, W)``. The minimum size of `H, W` depends on the
    chosen backbone (see `net_type` arg).

    Args:
        img1: first set of images
        img2: second set of images
        net_type: str indicating backbone network type to use. Choose between `'alex'`, `'vgg'` or `'squeeze'`
        reduction: str indicating how to reduce over the batch dimension. Choose between `'sum'`, `'mean'`, `'none'`
            or `None`.
        normalize: by default this is ``False`` meaning that the input is expected to be in the [-1,1] range. If set
            to ``True`` will instead expect input to be in the ``[0,1]`` range.

    Example:
        >>> from paddle import rand
        >>> from paddlemetrics.functional.image.lpips import learned_perceptual_image_patch_similarity
        >>> img1 = (rand(10, 3, 100, 100) * 2) - 1
        >>> img2 = (rand(10, 3, 100, 100) * 2) - 1
        >>> learned_perceptual_image_patch_similarity(img1, img2, net_type='squeeze')
        tensor(0.1005)

        >>> from paddle import rand, Generator
        >>> from paddlemetrics.functional.image.lpips import learned_perceptual_image_patch_similarity
        >>> gen = Generator().manual_seed(42)
        >>> img1 = (rand(2, 3, 100, 100, generator=gen) * 2) - 1
        >>> img2 = (rand(2, 3, 100, 100, generator=gen) * 2) - 1
        >>> learned_perceptual_image_patch_similarity(img1, img2, net_type='squeeze', reduction='none')
        tensor([0.1024, 0.0938])

    """
    net = _NoTrainLpips(net=net_type).to(device=img1.device, dtype=img1.dtype)
    loss = _lpips_update(img1, img2, net, normalize)
    return _lpips_compute(loss, reduction)
