from operator import attrgetter

import paddle
import pytest
import paddle_fidelity
from torch_fidelity.sample_similarity_lpips import SampleSimilarityLPIPS
from torch_fidelity.utils import batch_interp
from unittests._helpers import seed_all, skip_on_running_out_of_memory

from paddlemetrics.functional.image.lpips import _LPIPS
from paddlemetrics.functional.image.perceptual_path_length import (
    _interpolate, perceptual_path_length)
from paddlemetrics.image.perceptual_path_length import PerceptualPathLength
from paddlemetrics.utils.imports import _TORCH_FIDELITY_AVAILABLE

seed_all(42)


@pytest.mark.skipif(
    not _TORCH_FIDELITY_AVAILABLE, reason="metric requires torch-fidelity"
)
@pytest.mark.parametrize("interpolation_method", ["lerp", "slerp_any", "slerp_unit"])
def test_interpolation_methods(interpolation_method):
    """Test that interpolation method works as expected."""
    latent1 = paddle.randn(100, 25)
    latent2 = paddle.randn(100, 25)
    res1 = _interpolate(latent1, latent2, 0.0001, interpolation_method)
    res2 = batch_interp(latent1, latent2, 0.0001, interpolation_method)
    assert paddle.allclose(x=res1, y=res2).item()


@pytest.mark.skipif(
    not _TORCH_FIDELITY_AVAILABLE, reason="metric requires torch-fidelity"
)
@skip_on_running_out_of_memory()
def test_sim_net():
    """Check that the similarity network is the same as the one used in torch_fidelity."""
    compare = SampleSimilarityLPIPS("sample_similarity", resize=64)
    simnet = _LPIPS(net="vgg", resize=64)
    for name, weight in compare.named_parameters():
        getter = attrgetter(name)
        weight2 = getter(simnet)
        assert paddle.allclose(x=weight, y=weight2).item()
    img1 = paddle.rand(1, 3, 64, 64)
    img2 = paddle.rand(1, 3, 64, 64)
    out = compare(255 * img1, 255 * img2)
    out2 = simnet(2 * img1 - 1, 2 * img2 - 1)
    assert paddle.allclose(x=out, y=out2).item()


class DummyGenerator(paddle.nn.Layer):
    """From https://github.com/toshas/torch-fidelity/blob/master/examples/sngan_cifar10.py."""

    def __init__(self, z_size) -> None:
        super().__init__()
        self.z_size = z_size
        self.model = paddle.nn.Sequential(
            paddle.nn.Conv2DTranspose(
                in_channels=z_size, out_channels=512, kernel_size=4, stride=1
            ),
            paddle.nn.BatchNorm2D(num_features=512),
            paddle.nn.ReLU(),
            paddle.nn.Conv2DTranspose(
                in_channels=512,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=(1, 1),
            ),
            paddle.nn.BatchNorm2D(num_features=256),
            paddle.nn.ReLU(),
            paddle.nn.Conv2DTranspose(
                in_channels=256,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=(1, 1),
            ),
            paddle.nn.BatchNorm2D(num_features=128),
            paddle.nn.ReLU(),
            paddle.nn.Conv2DTranspose(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=(1, 1),
            ),
            paddle.nn.BatchNorm2D(num_features=64),
            paddle.nn.ReLU(),
            paddle.nn.Conv2DTranspose(
                in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=(1, 1)
            ),
            paddle.nn.Tanh(),
        )

    def forward(self, z):
        """Generate images from latent vectors."""
        fake = self.model(z.view(-1, self.z_size, 1, 1))
        if not self.training:
            fake = 255 * (fake.clamp(-1, 1) * 0.5 + 0.5)
            fake = fake.to(paddle.uint8)
        return fake

    def sample(self, num_samples):
        """Sample latent vectors."""
        return paddle.randn(num_samples, self.z_size)


@pytest.mark.skipif(
    not _TORCH_FIDELITY_AVAILABLE, reason="metric requires torch-fidelity"
)
@pytest.mark.parametrize(
    ("argument", "match"),
    [
        (
            {"num_samples": 0},
            "Argument `num_samples` must be a positive integer, but got 0.",
        ),
        ({"conditional": 2}, "Argument `conditional` must be a boolean, but got 2."),
        (
            {"batch_size": 0},
            "Argument `batch_size` must be a positive integer, but got 0.",
        ),
        (
            {"interpolation_method": "wrong"},
            "Argument `interpolation_method` must be one of.*",
        ),
        ({"epsilon": 0}, "Argument `epsilon` must be a positive float, but got 0."),
        (
            {"resize": 0},
            "Argument `resize` must be a positive integer or `None`, but got 0.",
        ),
        (
            {"lower_discard": -1},
            "Argument `lower_discard` must be a float between 0 and 1 or `None`, but got -1",
        ),
        (
            {"upper_discard": 2},
            "Argument `upper_discard` must be a float between 0 and 1 or `None`, but got 2",
        ),
    ],
)
@skip_on_running_out_of_memory()
def test_raises_error_on_wrong_arguments(argument, match):
    """Test that appropriate errors are raised on wrong arguments."""
    with pytest.raises(ValueError, match=match):
        perceptual_path_length(DummyGenerator(128), **argument)
    with pytest.raises(ValueError, match=match):
        PerceptualPathLength(**argument)


class _WrongGenerator1(paddle.nn.Layer):
    pass


class _WrongGenerator2(paddle.nn.Layer):
    sample = 1


class _WrongGenerator3(paddle.nn.Layer):
    def sample(self, n):
        return paddle.randn(n, 2)


class _WrongGenerator4(paddle.nn.Layer):
    def sample(self, n):
        return paddle.randn(n, 2)

    @property
    def num_classes(self):
        return [10, 10]


@pytest.mark.parametrize(
    ("generator", "errortype", "match"),
    [
        (
            _WrongGenerator1(),
            NotImplementedError,
            "The generator must have a `sample` method.*",
        ),
        (
            _WrongGenerator2(),
            ValueError,
            "The generator's `sample` method must be callable.",
        ),
        (
            _WrongGenerator3(),
            AttributeError,
            "The generator must have a `num_classes` attribute when `conditional=True`.",
        ),
        (
            _WrongGenerator4(),
            ValueError,
            "The generator's `num_classes` attribute must be an integer when `conditional=True`.",
        ),
    ],
)
@skip_on_running_out_of_memory()
def test_raises_error_on_wrong_generator(generator, errortype, match):
    """Test that appropriate errors are raised on wrong generator."""
    with pytest.raises(errortype, match=match):
        perceptual_path_length(generator, conditional=True)
    ppl = PerceptualPathLength(conditional=True)
    with pytest.raises(errortype, match=match):
        ppl.update(generator=generator)


@pytest.mark.skipif(
    not _TORCH_FIDELITY_AVAILABLE, reason="metric requires torch-fidelity"
)
@pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires GPU machine")
@skip_on_running_out_of_memory()
def test_compare():
    """Test against torch_fidelity.

    Because it is a sample based metric, the results are not deterministic. Thus we need a large amount of samples to
    even get close to the reference value. Even then we are going to allow a 6% deviation on the mean and 6% deviation
    on the standard deviation.

    """
    generator = DummyGenerator(128)
    compare = torch_fidelity.calculate_metrics(
        input1=torch_fidelity.GenerativeModelModuleWrapper(
            generator, 128, "normal", 10
        ),
        input1_model_num_samples=50000,
        ppl=True,
        ppl_reduction="none",
        input_model_num_classes=0,
        ppl_discard_percentile_lower=None,
        ppl_discard_percentile_higher=None,
    )
    compare = paddle.tensor(compare["perceptual_path_length_raw"])
    result = perceptual_path_length(
        generator,
        num_samples=50000,
        conditional=False,
        lower_discard=None,
        upper_discard=None,
        device="cuda",
    )
    result = result[-1].cpu()
    assert 0.94 * result.mean() <= compare.mean() <= 1.06 * result.mean()
    assert 0.94 * result.std() <= compare.std() <= 1.06 * result.std()
