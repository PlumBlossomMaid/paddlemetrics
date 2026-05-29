import sys

from contextlib import nullcontext as does_not_raise
from functools import partial

import numpy as np
import paddle
import pytest
from scipy.linalg import sqrtm
from unittests import _reference_cachier
from unittests._helpers import seed_all

from paddlemetrics.image.mifid import (
    MemorizationInformedFrechetInceptionDistance, NoTrainInceptionV3)
from paddlemetrics.utils.imports import _TORCH_FIDELITY_AVAILABLE


@_reference_cachier
def _reference_mifid(preds, target, cosine_distance_eps: float = 0.1):
    """Reference implementation.

    Implementation taken from:
    https://github.com/jybai/generative-memorization-benchmark/blob/main/src/competition_scoring.py

    Adjusted slightly to work with our code. We replace the feature extraction with our own, since we already check in
    FID that we use the correct feature extractor. This saves us from needing to download tensorflow for comparison.

    """

    def normalize_rows(x: np.ndarray):
        return np.nan_to_num(x / np.linalg.norm(x, ord=2, axis=1, keepdims=True))

    def cosine_distance(features1, features2):
        features1_nozero = features1[np.sum(features1, axis=1) != 0]
        features2_nozero = features2[np.sum(features2, axis=1) != 0]
        norm_f1 = normalize_rows(features1_nozero)
        norm_f2 = normalize_rows(features2_nozero)
        d = 1.0 - np.abs(np.matmul(norm_f1, norm_f2.T))
        return np.mean(np.min(d, axis=1))

    def distance_thresholding(d, eps):
        return d if d < eps else 1

    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-06):
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        diff = mu1 - mu2
        covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=0.001):
                m = np.max(np.abs(covmean.imag()))
                raise Exception(f"Imaginary component {m}")
            covmean = covmean.real()
        tr_covmean = np.trace(covmean)
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def calculate_activation_statistics(act):
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma, act

    def calculate_mifid(m1, s1, features1, m2, s2, features2):
        fid = calculate_frechet_distance(m1, s1, m2, s2)
        distance = cosine_distance(features1, features2)
        return fid, distance

    net = NoTrainInceptionV3(name="inception-v3-compat", features_list=[str(768)])
    preds_act = net(preds).numpy()
    target_act = net(target).numpy()
    m1, s1, features1 = calculate_activation_statistics(preds_act)
    m2, s2, features2 = calculate_activation_statistics(target_act)
    fid_private, distance_private = calculate_mifid(
        m1, s1, features1, m2, s2, features2
    )
    distance_private_thresholded = distance_thresholding(
        distance_private, cosine_distance_eps
    )
    return fid_private / (distance_private_thresholded + 1e-15)


@pytest.mark.skipif(
    not _TORCH_FIDELITY_AVAILABLE, reason="metric requires torch-fidelity"
)
def test_no_train():
    """Assert that metric never leaves evaluation mode."""

    class MyModel(paddle.nn.Layer):
        def __init__(self) -> None:
            super().__init__()
            self.metric = MemorizationInformedFrechetInceptionDistance()

        def forward(self, x):
            return x

    model = MyModel()
    model.train()
    assert model.training
    assert (
        not model.metric.inception.training
    ), "MiFID metric was changed to training mode which should not happen"


def test_mifid_raises_errors_and_warnings():
    """Test that expected warnings and errors are raised."""
    if _TORCH_FIDELITY_AVAILABLE:
        with pytest.raises(
            ValueError, match="Integer input to argument `feature` must be one of .*"
        ):
            _ = MemorizationInformedFrechetInceptionDistance(feature=2)
    else:
        with pytest.raises(
            ModuleNotFoundError,
            match="FID metric requires that `Torch-fidelity` is installed. Either install as `pip install paddlemetrics[image-quality]` or `pip install torch-fidelity`.",
        ):
            _ = MemorizationInformedFrechetInceptionDistance()
    with pytest.raises(TypeError, match="Got unknown input to argument `feature`"):
        _ = MemorizationInformedFrechetInceptionDistance(feature=[1, 2])
    with pytest.raises(
        ValueError,
        match="Argument `cosine_distance_eps` expected to be a float greater than 0",
    ):
        _ = MemorizationInformedFrechetInceptionDistance(cosine_distance_eps=-1)
    with pytest.raises(
        ValueError,
        match="Argument `cosine_distance_eps` expected to be a float greater than 0",
    ):
        _ = MemorizationInformedFrechetInceptionDistance(cosine_distance_eps=1.1)


@pytest.mark.skipif(
    not _TORCH_FIDELITY_AVAILABLE, reason="metric requires torch-fidelity"
)
@pytest.mark.parametrize("feature", [64, 192, 768, 2048])
def test_fid_same_input(feature):
    """If real and fake are update on the same data the fid score should be 0."""
    metric = MemorizationInformedFrechetInceptionDistance(feature=feature)
    seed_all(42)
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
    val = metric.compute()
    assert paddle.allclose(x=val, y=paddle.zeros_like(val), atol=0.001).item()


@pytest.mark.skipif(
    not paddle.cuda.is_available(), reason="test is too slow without gpu"
)
@pytest.mark.skipif(
    not _TORCH_FIDELITY_AVAILABLE, reason="metric requires torch-fidelity"
)
@pytest.mark.parametrize("equal_size", [False])
def test_compare_mifid(equal_size):
    """Check that our implementation of MIFID is correct by comparing it to the original implementation."""
    metric = MemorizationInformedFrechetInceptionDistance(feature=768).cuda()
    n, m = 100, 100 if equal_size else 90
    seed_all(42)
    img1 = paddle.randint(low=0, high=180, shape=(n, 3, 299, 299), dtype=paddle.uint8)
    img2 = paddle.randint(low=100, high=255, shape=(m, 3, 299, 299), dtype=paddle.uint8)
    batch_size = 10
    for i in range(n // batch_size):
        metric.update(img1[batch_size * i : batch_size * (i + 1)].cuda(), real=True)
    for i in range(m // batch_size):
        metric.update(img2[batch_size * i : batch_size * (i + 1)].cuda(), real=False)
    compare_val = _reference_mifid(img1, img2)
    tm_res = metric.compute()
    assert paddle.allclose(
        x=tm_res.cpu(), y=paddle.tensor(compare_val, dtype=tm_res.dtype), atol=0.001
    ).item()


@pytest.mark.parametrize("normalize", [True, False])
def test_normalize_arg(normalize):
    """Test that normalize argument works as expected."""
    img = paddle.rand(2, 3, 299, 299)
    metric = MemorizationInformedFrechetInceptionDistance(normalize=normalize)
    context = (
        partial(
            pytest.raises,
            expected_exception=ValueError,
            match="Expecting image as paddle.Tensor with dtype=paddle.uint8",
        )
        if not normalize
        else does_not_raise
    )
    with context():
        metric.update(img, real=True)


def test_mifid_custom_encoder_with_normalize():
    """Test that MIFID works with custom encoder and normalize=True without converting inputs to byte."""
    custom_encoder = paddle.nn.Sequential(
        paddle.nn.Flatten(), paddle.nn.Linear(32 * 32, 512)
    )
    input_tensor = paddle.randn(2, 1, 32, 32)
    mifid = MemorizationInformedFrechetInceptionDistance(
        feature=custom_encoder, normalize=True
    )
    mifid.update(input_tensor, real=True)
    mifid.update(input_tensor, real=False)
    assert len(mifid.real_features) == 1
    assert len(mifid.fake_features) == 1
    result = mifid.compute()
    assert isinstance(result, paddle.Tensor)
