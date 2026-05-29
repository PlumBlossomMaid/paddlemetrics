import time

import paddle
import pytest

from paddlemetrics import MetricCollection
from paddlemetrics.image import (FrechetInceptionDistance, InceptionScore,
                                KernelInceptionDistance,
                                LearnedPerceptualImagePatchSimilarity,
                                StructuralSimilarityIndexMeasure)
from paddlemetrics.wrappers import FeatureShare
from paddlemetrics.wrappers.feature_share import NetworkCache


@pytest.mark.parametrize(
    "metrics",
    [
        [
            FrechetInceptionDistance(feature=64),
            InceptionScore(feature=64),
            KernelInceptionDistance(feature=64),
        ],
        {
            "fid": FrechetInceptionDistance(feature=64),
            "is": InceptionScore(feature=64),
            "kid": KernelInceptionDistance(feature=64),
        },
    ],
)
def test_initialization(metrics):
    """Test that the feature share wrapper can be initialized."""
    fs = FeatureShare(metrics)
    assert isinstance(fs, MetricCollection)
    assert len(fs) == 3


def test_error_on_missing_feature_network():
    """Test that an error is raised when the feature network is missing."""
    with pytest.raises(
        AttributeError,
        match="Tried to extract the network to share from the first metric.*",
    ):
        FeatureShare(
            [StructuralSimilarityIndexMeasure(), FrechetInceptionDistance(feature=64)]
        )
    with pytest.raises(
        AttributeError,
        match="Tried to set the cached network to all metrics, but one of the.*",
    ):
        FeatureShare(
            [FrechetInceptionDistance(feature=64), StructuralSimilarityIndexMeasure()]
        )


def test_warning_on_mixing_networks():
    """Test that a warning is raised when the metrics use different networks."""
    with pytest.warns(
        UserWarning, match="The network to share between the metrics is not.*"
    ):
        FeatureShare(
            [
                FrechetInceptionDistance(feature=64),
                InceptionScore(feature=64),
                LearnedPerceptualImagePatchSimilarity(),
            ]
        )


def test_feature_share_speed():
    """Test that the feature share wrapper is faster than the metric collection."""
    mc = MetricCollection(
        [
            FrechetInceptionDistance(feature=64),
            InceptionScore(feature=64),
            KernelInceptionDistance(feature=64),
        ]
    )
    fs = FeatureShare(
        [
            FrechetInceptionDistance(feature=64),
            InceptionScore(feature=64),
            KernelInceptionDistance(feature=64),
        ]
    )
    x = paddle.randint(low=0, high=255, shape=(1, 3, 64, 64), dtype=paddle.uint8)
    start = time.time()
    for _ in range(10):
        x = paddle.randint(low=0, high=255, shape=(1, 3, 64, 64), dtype=paddle.uint8)
        mc.update(x, real=True)
    end = time.time()
    mc_time = end - start
    start = time.time()
    for _ in range(10):
        x = paddle.randint(low=0, high=255, shape=(1, 3, 64, 64), dtype=paddle.uint8)
        fs.update(x, real=True)
    end = time.time()
    fs_time = end - start
    assert (
        fs_time < mc_time
    ), "The feature share wrapper should be faster than the metric collection."


@pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires GPU machine")
def test_memory():
    """Test that the feature share wrapper uses less memory than the metric collection."""
    base_memory = paddle.cuda.memory_allocated()
    fid = FrechetInceptionDistance(feature=64).cuda()
    inception = InceptionScore(feature=64).cuda()
    kid = KernelInceptionDistance(feature=64, subset_size=5).cuda()
    memory_before_fs = paddle.cuda.memory_allocated()
    assert (
        memory_before_fs > base_memory
    ), "The memory usage should be higher after initializing the metrics."
    paddle.cuda.empty_cache()
    feature_share = FeatureShare([fid, inception, kid]).cuda()
    memory_after_fs = paddle.cuda.memory_allocated()
    assert (
        memory_after_fs > base_memory
    ), "The memory usage should be higher after initializing the feature share wrapper."
    assert (
        memory_after_fs < memory_before_fs
    ), "The memory usage should be higher after initializing the feature share wrapper."
    img1 = paddle.randint(
        low=0, high=255, shape=(50, 3, 220, 220), dtype=paddle.uint8
    ).to("cuda")
    img2 = paddle.randint(
        low=0, high=255, shape=(50, 3, 220, 220), dtype=paddle.uint8
    ).to("cuda")
    feature_share.update(img1, real=True)
    feature_share.update(img2, real=False)
    res = feature_share.compute()
    assert "cuda" in str(res["FrechetInceptionDistance"].place)
    assert "cuda" in str(res["InceptionScore"][0].place)
    assert "cuda" in str(res["InceptionScore"][1].place)
    assert "cuda" in str(res["KernelInceptionDistance"][0].place)
    assert "cuda" in str(res["KernelInceptionDistance"][1].place)


def test_same_result_as_individual():
    """Test that the feature share wrapper gives the same result as the individual metrics."""
    fid = FrechetInceptionDistance(feature=64)
    inception = InceptionScore(feature=64)
    kid = KernelInceptionDistance(feature=64, subset_size=10, subsets=2)
    fs = FeatureShare([fid, inception, kid])
    x = paddle.randint(low=0, high=255, shape=(50, 3, 64, 64), dtype=paddle.uint8)
    fs.update(x, real=True)
    fid.update(x, real=True)
    inception.update(x)
    kid.update(x, real=True)
    x = paddle.randint(low=0, high=255, shape=(50, 3, 64, 64), dtype=paddle.uint8)
    fs.update(x, real=False)
    fid.update(x, real=False)
    inception.update(x)
    kid.update(x, real=False)
    fs_res = fs.compute()
    fid_res = fid.compute()
    inception_res = inception.compute()
    kid_res = kid.compute()
    assert fs_res["FrechetInceptionDistance"] == fid_res
    assert fs_res["InceptionScore"][0] == inception_res[0]
    assert fs_res["InceptionScore"][1] == inception_res[1]
    assert fs_res["KernelInceptionDistance"][0] == kid_res[0]
    assert fs_res["KernelInceptionDistance"][1] == kid_res[1]


def test_network_cache():
    """Test the NetworkCache class."""

    class TestNetwork(paddle.nn.Layer):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, x: paddle.Tensor) -> paddle.Tensor:
            self.calls += 1
            return x

    network = TestNetwork()
    cached_net = NetworkCache(network, max_size=2)
    x1 = paddle.randn(1, 3, 64, 64)
    cached_net(x1)
    assert network.calls == 1
    cached_net(x1)
    assert network.calls == 1
    x2 = paddle.randn(1, 3, 64, 64)
    cached_net(x2)
    assert network.calls == 2
    x3 = paddle.randn(1, 3, 64, 64)
    cached_net(x3)
    assert network.calls == 3


def test_feature_share_initialization_edge_cases():
    """Test edge cases for FeatureShare initialization."""
    with pytest.raises(TypeError, match="max_cache_size should be an integer"):
        FeatureShare([FrechetInceptionDistance(feature=64)], max_cache_size="invalid")

    class BadMetric(FrechetInceptionDistance):
        def __init__(self) -> None:
            super().__init__(feature=64)
            self.feature_network = 123

    with pytest.raises(
        TypeError, match="The `feature_network` attribute must be a string"
    ):
        FeatureShare([BadMetric()])
    fs = FeatureShare(FrechetInceptionDistance(feature=64))
    assert len(fs) == 1
    assert isinstance(fs["FrechetInceptionDistance"], FrechetInceptionDistance)
