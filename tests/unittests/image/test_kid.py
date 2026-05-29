import sys

import pickle
from contextlib import nullcontext as does_not_raise

import paddle
import pytest
from unittests._helpers import seed_all

from paddlemetrics.image.kid import KernelInceptionDistance
from paddlemetrics.utils.imports import _TORCH_FIDELITY_AVAILABLE

seed_all(42)


@pytest.mark.skipif(
    not _TORCH_FIDELITY_AVAILABLE, reason="metric requires torch-fidelity"
)
def test_no_train():
    """Assert that metric never leaves evaluation mode."""

    class MyModel(paddle.nn.Layer):
        def __init__(self) -> None:
            super().__init__()
            self.metric = KernelInceptionDistance()

        def forward(self, x):
            return x

    model = MyModel()
    model.train()
    assert model.training
    assert (
        not model.metric.inception.training
    ), "FID metric was changed to training mode which should not happen"


@pytest.mark.skipif(
    not _TORCH_FIDELITY_AVAILABLE, reason="metric requires torch-fidelity"
)
def test_kid_pickle():
    """Assert that we can initialize the metric and pickle it."""
    metric = KernelInceptionDistance()
    assert metric
    pickled_metric = pickle.dumps(metric)
    metric = pickle.loads(pickled_metric)


def test_kid_raises_errors_and_warnings():
    """Test that expected warnings and errors are raised."""
    with pytest.warns(
        UserWarning,
        match="Metric `Kernel Inception Distance` will save all extracted features in buffer. For large datasets this may lead to large memory footprint.",
    ):
        KernelInceptionDistance()
    if _TORCH_FIDELITY_AVAILABLE:
        with pytest.raises(
            ValueError, match="Integer input to argument `feature` must be one of .*"
        ):
            KernelInceptionDistance(feature=2)
    else:
        with pytest.raises(
            ModuleNotFoundError,
            match="Kernel Inception Distance metric requires that `Torch-fidelity` is installed. Either install as `pip install paddlemetrics[image]` or `pip install torch-fidelity`.",
        ):
            KernelInceptionDistance()
    with pytest.raises(TypeError, match="Got unknown input to argument `feature`"):
        KernelInceptionDistance(feature=[1, 2])
    m = KernelInceptionDistance()
    m.update(
        paddle.randint(low=0, high=255, shape=(5, 3, 299, 299), dtype=paddle.uint8),
        real=True,
    )
    m.update(
        paddle.randint(low=0, high=255, shape=(5, 3, 299, 299), dtype=paddle.uint8),
        real=False,
    )
    with pytest.raises(
        ValueError,
        match="Argument `subset_size` should be smaller than the number of samples",
    ):
        m.compute()


@pytest.mark.skipif(
    not _TORCH_FIDELITY_AVAILABLE, reason="metric requires torch-fidelity"
)
def test_kid_extra_parameters():
    """Test that the different input arguments raises expected errors if wrong."""
    with pytest.raises(
        ValueError, match="Argument `subsets` expected to be integer larger than 0"
    ):
        KernelInceptionDistance(subsets=-1)
    with pytest.raises(
        ValueError, match="Argument `subset_size` expected to be integer larger than 0"
    ):
        KernelInceptionDistance(subset_size=-1)
    with pytest.raises(
        ValueError, match="Argument `degree` expected to be integer larger than 0"
    ):
        KernelInceptionDistance(degree=-1)
    with pytest.raises(
        ValueError,
        match="Argument `gamma` expected to be `None` or float larger than 0",
    ):
        KernelInceptionDistance(gamma=-1)
    with pytest.raises(
        ValueError, match="Argument `coef` expected to be float larger than 0"
    ):
        KernelInceptionDistance(coef=-1)


class _DummyFeatureExtractor(paddle.nn.Layer):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = paddle.nn.Flatten()
        self.extractor = paddle.nn.Linear(3 * 299 * 299, 64)

    def __call__(self, img) -> paddle.Tensor:
        img = (img / 125.5).float()
        return self.extractor(self.flatten(img))


@pytest.mark.skipif(
    not _TORCH_FIDELITY_AVAILABLE, reason="metric requires torch-fidelity"
)
@pytest.mark.parametrize("feature", [64, 192, 768, 2048, _DummyFeatureExtractor()])
def test_kid_same_input(feature):
    """Test that the metric works."""
    metric = KernelInceptionDistance(feature=feature, subsets=5, subset_size=2)
    for _ in range(2):
        img = paddle.randint(
            low=0, high=255, shape=(10, 3, 299, 299), dtype=paddle.uint8
        )
        metric.update(img, real=True)
        metric.update(img, real=False)
    assert paddle.allclose(
        x=paddle.concat(metric.real_features, axis=0),
        y=paddle.concat(metric.fake_features, axis=0),
    ).item()
    mean, std = metric.compute()
    assert mean != 0.0
    assert std >= 0.0


class _ImgDataset(paddle.io.Dataset):
    def __init__(self, imgs) -> None:
        self.imgs = imgs

    def __getitem__(self, idx) -> paddle.Tensor:
        return self.imgs[idx]

    def __len__(self) -> int:
        return self.imgs.shape[0]


@pytest.mark.skipif(
    not paddle.cuda.is_available(), reason="test is too slow without gpu"
)
@pytest.mark.skipif(
    not _TORCH_FIDELITY_AVAILABLE, reason="metric requires torch-fidelity"
)
def test_compare_kid(tmpdir, feature=2048):
    """Check that the hole pipeline give the same result as torch-fidelity."""
    from torch_fidelity import calculate_metrics

    metric = KernelInceptionDistance(feature=feature, subsets=1, subset_size=100).cuda()
    img1 = paddle.randint(low=0, high=180, shape=(100, 3, 299, 299), dtype=paddle.uint8)
    img2 = paddle.randint(
        low=100, high=255, shape=(100, 3, 299, 299), dtype=paddle.uint8
    )
    batch_size = 10
    for i in range(img1.shape[0] // batch_size):
        metric.update(img1[batch_size * i : batch_size * (i + 1)].cuda(), real=True)
    for i in range(img2.shape[0] // batch_size):
        metric.update(img2[batch_size * i : batch_size * (i + 1)].cuda(), real=False)
    torch_fid = calculate_metrics(
        input1=_ImgDataset(img1),
        input2=_ImgDataset(img2),
        kid=True,
        feature_layer_fid=str(feature),
        batch_size=batch_size,
        kid_subsets=1,
        kid_subset_size=100,
        save_cpu_ram=True,
    )
    tm_mean, tm_std = metric.compute()
    assert paddle.allclose(
        x=tm_mean.cpu(),
        y=paddle.tensor([torch_fid["kernel_inception_distance_mean"]]),
        atol=0.001,
    ).item()
    assert paddle.allclose(
        x=tm_std.cpu(),
        y=paddle.tensor([torch_fid["kernel_inception_distance_std"]]),
        atol=0.001,
    ).item()


@pytest.mark.parametrize("reset_real_features", [True, False])
def test_reset_real_features_arg(reset_real_features):
    """Test that `reset_real_features` arg works as expected."""
    metric = KernelInceptionDistance(
        feature=64, reset_real_features=reset_real_features
    )
    metric.update(
        paddle.randint(low=0, high=180, shape=(2, 3, 299, 299), dtype=paddle.uint8),
        real=True,
    )
    metric.update(
        paddle.randint(low=0, high=180, shape=(2, 3, 299, 299), dtype=paddle.uint8),
        real=False,
    )
    assert len(metric.real_features) == 1
    assert list(metric.real_features[0].shape) == [2, 64]
    assert len(metric.fake_features) == 1
    assert list(metric.fake_features[0].shape) == [2, 64]
    metric.reset()
    assert len(metric.fake_features) == 0
    if reset_real_features:
        assert len(metric.real_features) == 0
    else:
        assert len(metric.real_features) == 1
        assert list(metric.real_features[0].shape) == [2, 64]


def test_normalize_arg_true():
    """Test that normalize argument works as expected."""
    img = paddle.rand(2, 3, 299, 299)
    metric = KernelInceptionDistance(normalize=True)
    with does_not_raise():
        metric.update(img, real=True)


def test_normalize_arg_false():
    """Test that normalize argument works as expected."""
    img = paddle.rand(2, 3, 299, 299)
    metric = KernelInceptionDistance(normalize=False)
    with pytest.raises(
        ValueError, match="Expecting image as paddle.Tensor with dtype=paddle.uint8"
    ):
        metric.update(img, real=True)
