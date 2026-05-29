import paddle
import pytest
from sklearn.metrics import rand_score as sklearn_rand_score
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester
from unittests.clustering._inputs import (_float_inputs_extrinsic,
                                          _single_target_extrinsic1,
                                          _single_target_extrinsic2)

from paddlemetrics.clustering.rand_score import RandScore
from paddlemetrics.functional.clustering.rand_score import rand_score

seed_all(42)


@pytest.mark.parametrize(
    ("preds", "target"),
    [
        (_single_target_extrinsic1.preds, _single_target_extrinsic1.target),
        (_single_target_extrinsic2.preds, _single_target_extrinsic2.target),
    ],
)
class TestRandScore(MetricTester):
    """Test class for `RandScore` metric."""

    atol = 1e-05

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_rand_score(self, preds, target, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=RandScore,
            reference_metric=sklearn_rand_score,
        )

    def test_rand_score_functional(self, preds, target):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=rand_score,
            reference_metric=sklearn_rand_score,
        )


def test_rand_score_functional_raises_invalid_task():
    """Check that metric rejects continuous-valued inputs."""
    preds, target = _float_inputs_extrinsic
    with pytest.raises(ValueError, match="Expected *"):
        rand_score(preds, target)


def test_rand_score_functional_is_symmetric(
    preds=_single_target_extrinsic1.preds, target=_single_target_extrinsic1.target
):
    """Check that the metric functional is symmetric."""
    for p, t in zip(preds, target):
        assert paddle.allclose(x=rand_score(p, t), y=rand_score(t, p)).item()
