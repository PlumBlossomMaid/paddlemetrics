import sys

import pickle
from contextlib import nullcontext as does_not_raise

import paddle
import pytest
from unittests._helpers import seed_all

from paddlemetrics.image.inception import InceptionScore
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
            self.metric = InceptionScore()

        def forward(self, x):
            return x

    model = MyModel()
    model.train()
    assert model.training
    assert (
        not model.metric.inception.training
    ), "InceptionScore metric was changed to training mode which should not happen"


@pytest.mark.skipif(
    not _TORCH_FIDELITY_AVAILABLE, reason="metric requires torch-fidelity"
)
def test_is_pickle():
    """Assert that we can initialize the metric and pickle it."""
    metric = InceptionScore()
    assert metric
    pickled_metric = pickle.dumps(metric)
    metric = pickle.loads(pickled_metric)


def test_is_raises_errors_and_warnings():
    """Test that expected warnings and errors are raised."""
    with pytest.warns(
        UserWarning,
        match="Metric `InceptionScore` will save all extracted features in buffer. For large datasets this may lead to large memory footprint.",
    ):
        InceptionScore()
    if _TORCH_FIDELITY_AVAILABLE:
        with pytest.raises(
            ValueError, match="Integer input to argument `feature` must be one of .*"
        ):
            _ = InceptionScore(feature=2)
    else:
        with pytest.raises(
            ModuleNotFoundError,
            match="InceptionScore metric requires that `Torch-fidelity` is installed. Either install as `pip install paddlemetrics[image-quality]` or `pip install torch-fidelity`.",
        ):
            InceptionScore()
    with pytest.raises(TypeError, match="Got unknown input to argument `feature`"):
        InceptionScore(feature=[1, 2])


@pytest.mark.skipif(
    not _TORCH_FIDELITY_AVAILABLE, reason="metric requires torch-fidelity"
)
def test_is_update_compute():
    """Test that inception score works as expected."""
    metric = InceptionScore()
    for _ in range(2):
        img = paddle.randint(
            low=0, high=255, shape=(10, 3, 299, 299), dtype=paddle.uint8
        )
        metric.update(img)
    mean, std = metric.compute()
    assert mean >= 0.0
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
@pytest.mark.parametrize("compute_on_cpu", [True, False])
def test_compare_is(tmpdir, compute_on_cpu):
    """Check that the hole pipeline give the same result as torch-fidelity."""
    from torch_fidelity import calculate_metrics

    metric = InceptionScore(splits=1, compute_on_cpu=compute_on_cpu).cuda()
    img1 = paddle.randint(low=0, high=255, shape=(100, 3, 299, 299), dtype=paddle.uint8)
    batch_size = 10
    for i in range(img1.shape[0] // batch_size):
        metric.update(img1[batch_size * i : batch_size * (i + 1)].cuda())
    torch_fid = calculate_metrics(
        input1=_ImgDataset(img1),
        isc=True,
        isc_splits=1,
        batch_size=batch_size,
        save_cpu_ram=True,
    )
    tm_mean, _ = metric.compute()
    assert paddle.allclose(
        x=tm_mean.cpu(),
        y=paddle.tensor([torch_fid["inception_score_mean"]]),
        atol=0.001,
    ).item()


def test_normalize_arg_true():
    """Test that normalize argument works as expected."""
    img = paddle.rand(2, 3, 299, 299)
    metric = InceptionScore(normalize=True)
    with does_not_raise():
        metric.update(img)


def test_normalize_arg_false():
    """Test that normalize argument works as expected."""
    img = paddle.rand(2, 3, 299, 299)
    metric = InceptionScore(normalize=False)
    with pytest.raises(
        ValueError, match="Expecting image as paddle.Tensor with dtype=paddle.uint8"
    ):
        metric.update(img)
