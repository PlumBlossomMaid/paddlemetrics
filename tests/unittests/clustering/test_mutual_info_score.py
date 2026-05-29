import paddle
import pytest
from sklearn.metrics import mutual_info_score as sklearn_mutual_info_score
from unittests import BATCH_SIZE, NUM_CLASSES
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester
from unittests.clustering._inputs import (_float_inputs_extrinsic,
                                          _single_target_extrinsic1,
                                          _single_target_extrinsic2)

from paddlemetrics.clustering.mutual_info_score import MutualInfoScore
from paddlemetrics.functional.clustering.mutual_info_score import \
    mutual_info_score

seed_all(42)


@pytest.mark.parametrize(
    ("preds", "target"),
    [
        (_single_target_extrinsic1.preds, _single_target_extrinsic1.target),
        (_single_target_extrinsic2.preds, _single_target_extrinsic2.target),
    ],
)
class TestMutualInfoScore(MetricTester):
    """Test class for `MutualInfoScore` metric."""

    atol = 1e-05

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_mutual_info_score(self, preds, target, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MutualInfoScore,
            reference_metric=sklearn_mutual_info_score,
        )

    def test_mutual_info_score_functional(self, preds, target):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=mutual_info_score,
            reference_metric=sklearn_mutual_info_score,
        )


def test_mutual_info_score_functional_single_cluster():
    """Check that for single cluster the metric returns 0."""
    tensor_a = paddle.randint(low=0, high=NUM_CLASSES, shape=(BATCH_SIZE,))
    tensor_b = paddle.zeros(BATCH_SIZE, dtype=paddle.int32)
    assert paddle.allclose(
        x=mutual_info_score(tensor_a, tensor_b), y=paddle.tensor(0.0)
    ).item()
    assert paddle.allclose(
        x=mutual_info_score(tensor_b, tensor_a), y=paddle.tensor(0.0)
    ).item()


def test_mutual_info_score_functional_raises_invalid_task():
    """Check that metric rejects continuous-valued inputs."""
    preds, target = _float_inputs_extrinsic
    with pytest.raises(ValueError, match="Expected *"):
        mutual_info_score(preds, target)


def test_mutual_info_score_functional_is_symmetric(
    preds=_single_target_extrinsic1.preds, target=_single_target_extrinsic1.target
):
    """Check that the metric functional is symmetric."""
    for p, t in zip(preds, target):
        assert paddle.allclose(
            x=mutual_info_score(p, t), y=mutual_info_score(t, p)
        ).item()
