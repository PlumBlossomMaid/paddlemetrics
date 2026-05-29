from functools import partial
from typing import NamedTuple

import paddle
from paddle import Tensor
import pytest
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

from paddlemetrics.functional.image.lpips import \
    learned_perceptual_image_patch_similarity
from paddlemetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from paddlemetrics.utils.imports import _TORCHVISION_AVAILABLE

seed_all(42)


class _Input(NamedTuple):
    img1: Tensor
    img2: Tensor


_inputs = _Input(img1=paddle.rand(4, 2, 3, 50, 50), img2=paddle.rand(4, 2, 3, 50, 50))


def _reference_lpips(
    img1: paddle.Tensor,
    img2: paddle.Tensor,
    net_type: str,
    normalize: bool = False,
    reduction: str = "mean",
) -> paddle.Tensor:
    """Comparison function for tm implementation."""
    try:
        from lpips import LPIPS
    except ImportError:
        pytest.skip("test requires lpips package to be installed")
    ref = LPIPS(net=net_type)
    res = ref(img1, img2, normalize=normalize).detach().cpu().numpy()
    if reduction == "mean":
        return res.mean()
    if reduction == "sum":
        return res.sum()
    return res.flatten()


@pytest.mark.skipif(
    not _TORCHVISION_AVAILABLE, reason="test requires that torchvision is installed"
)
class TestLPIPS(MetricTester):
    """Test class for `LearnedPerceptualImagePatchSimilarity` metric."""

    atol: float = 0.0001

    @pytest.mark.parametrize("net_type", ["alex", "squeeze"])
    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_lpips(self, net_type, reduction, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=_inputs.img1,
            target=_inputs.img2,
            metric_class=LearnedPerceptualImagePatchSimilarity,
            reference_metric=partial(
                _reference_lpips, net_type=net_type, reduction=reduction
            ),
            check_scriptable=False,
            check_state_dict=False,
            metric_args={"net_type": net_type, "reduction": reduction},
            check_ddp_sorting=True,
        )

    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_lpips_functional(self, reduction):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=_inputs.img1,
            target=_inputs.img2,
            metric_functional=learned_perceptual_image_patch_similarity,
            reference_metric=partial(
                _reference_lpips, net_type="alex", reduction=reduction
            ),
            metric_args={"net_type": "alex", "reduction": reduction},
        )

    def test_lpips_differentiability(self):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            preds=_inputs.img1,
            target=_inputs.img2,
            metric_module=LearnedPerceptualImagePatchSimilarity,
        )

    def test_lpips_half_cpu(self):
        """Test for half + cpu support."""
        self.run_precision_test_cpu(
            _inputs.img1, _inputs.img2, LearnedPerceptualImagePatchSimilarity
        )

    @pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires cuda")
    def test_lpips_half_gpu(self):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            _inputs.img1, _inputs.img2, LearnedPerceptualImagePatchSimilarity
        )


@pytest.mark.parametrize("normalize", [False])
def test_normalize_arg(normalize):
    """Test that normalize argument works as expected."""
    metric = LearnedPerceptualImagePatchSimilarity(
        net_type="squeeze", normalize=normalize
    )
    res = metric(_inputs.img1[0], _inputs.img2[1])
    res2 = _reference_lpips(
        _inputs.img1[0], _inputs.img2[1], net_type="squeeze", normalize=normalize
    )
    assert res == res2


@pytest.mark.skipif(
    not _TORCHVISION_AVAILABLE, reason="test requires that torchvision is installed"
)
def test_error_on_wrong_init():
    """Test class raises the expected errors."""
    with pytest.raises(ValueError, match="Argument `net_type` must be one .*"):
        LearnedPerceptualImagePatchSimilarity(net_type="resnet")
    with pytest.raises(ValueError, match="Argument `reduction` must be one .*"):
        LearnedPerceptualImagePatchSimilarity(
            net_type="squeeze", reduction="invalid_option"
        )


@pytest.mark.skipif(
    not _TORCHVISION_AVAILABLE, reason="test requires that torchvision is installed"
)
@pytest.mark.parametrize(
    ("inp1", "inp2"),
    [
        (paddle.rand(1, 1, 28, 28), paddle.rand(1, 3, 28, 28)),
        (paddle.rand(1, 3, 28, 28), paddle.rand(1, 1, 28, 28)),
        (paddle.randn(1, 3, 28, 28), paddle.rand(1, 3, 28, 28)),
        (paddle.rand(1, 3, 28, 28), paddle.randn(1, 3, 28, 28)),
    ],
)
def test_error_on_wrong_update(inp1, inp2):
    """Test error is raised on wrong input to update method."""
    metric = LearnedPerceptualImagePatchSimilarity()
    with pytest.raises(
        ValueError, match="Expected both input arguments to be normalized tensors .*"
    ):
        metric(inp1, inp2)


def test_check_for_backprop():
    """Check that by default the metric supports propagation of gradients, but does not update its parameters."""
    metric = LearnedPerceptualImagePatchSimilarity()
    assert not metric.net.lin0.model[1].weight.requires_grad
    preds, target = _inputs.img1[0], _inputs.img2[0]
    preds.stop_gradient = not True
    loss = metric(preds, target)
    assert loss.requires_grad
    loss.backward()
    assert metric.net.lin0.model[1].weight.grad is None
