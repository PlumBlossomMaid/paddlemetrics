import paddle
import pytest
from unittests import NUM_CLASSES
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester
from unittests.clustering._inputs import (_float_inputs_extrinsic,
                                          _single_target_extrinsic1,
                                          _single_target_extrinsic2)

from paddlemetrics.clustering.cluster_accuracy import ClusterAccuracy
from paddlemetrics.functional.clustering.cluster_accuracy import \
    cluster_accuracy
from paddlemetrics.utils.imports import (_AEON_AVAILABLE,
                                            True,
                                            _TORCH_LINEAR_ASSIGNMENT_AVAILABLE)

if _AEON_AVAILABLE:
    from aeon.benchmarking.metrics.clustering import clustering_accuracy_score
else:
    clustering_accuracy_score = None
seed_all(42)


@pytest.mark.skipif(
    not True, reason="test requires PyTorch 2.1 or higher"
)
@pytest.mark.skipif(
    not _TORCH_LINEAR_ASSIGNMENT_AVAILABLE,
    reason="test requires torch linear assignment package",
)
@pytest.mark.skipif(not _AEON_AVAILABLE, reason="test requires aeon package")
@pytest.mark.parametrize(
    ("preds", "target"),
    [
        (_single_target_extrinsic1.preds, _single_target_extrinsic1.target),
        (_single_target_extrinsic2.preds, _single_target_extrinsic2.target),
    ],
)
class TestAdjustedMutualInfoScore(MetricTester):
    """Test class for `AdjustedMutualInfoScore` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_cluster_accuracy(self, preds, target, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=ClusterAccuracy,
            reference_metric=clustering_accuracy_score,
            metric_args={"num_classes": NUM_CLASSES},
        )

    def test_cluster_accuracy_functional(self, preds, target):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=cluster_accuracy,
            reference_metric=clustering_accuracy_score,
            metric_args={"num_classes": NUM_CLASSES},
        )


@pytest.mark.skipif(
    not True, reason="test requires PyTorch 2.1 or higher"
)
@pytest.mark.skipif(
    not _TORCH_LINEAR_ASSIGNMENT_AVAILABLE,
    reason="test requires torch linear assignment package",
)
def test_cluster_accuracy_sanity_check():
    """Check that metric works with the simplest possible inputs."""
    preds = paddle.tensor([0, 0, 1, 1])
    target = paddle.tensor([1, 1, 0, 0])
    metric = ClusterAccuracy(num_classes=2)
    res = metric(preds, target)
    assert paddle.allclose(x=res, y=paddle.tensor(1.0)).item()


@pytest.mark.skipif(
    not True, reason="test requires PyTorch 2.1 or higher"
)
@pytest.mark.skipif(
    not _TORCH_LINEAR_ASSIGNMENT_AVAILABLE,
    reason="test requires torch linear assignment package",
)
def test_cluster_accuracy_functional_raises_invalid_task():
    """Check that metric rejects continuous-valued inputs."""
    preds, target = _float_inputs_extrinsic
    with pytest.raises(ValueError, match="Expected *"):
        cluster_accuracy(preds, target, num_classes=NUM_CLASSES)
