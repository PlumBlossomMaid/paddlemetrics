from functools import partial

import paddle
import pytest
from sklearn.metrics import adjusted_mutual_info_score as sklearn_ami
from unittests import BATCH_SIZE, NUM_CLASSES
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester
from unittests.clustering._inputs import (_float_inputs_extrinsic,
                                          _single_target_extrinsic1,
                                          _single_target_extrinsic2)

from paddlemetrics.clustering.adjusted_mutual_info_score import \
    AdjustedMutualInfoScore
from paddlemetrics.functional.clustering.adjusted_mutual_info_score import \
    adjusted_mutual_info_score

seed_all(42)
ATOL = 1e-05


@pytest.mark.parametrize(
    ("preds", "target"),
    [
        (_single_target_extrinsic1.preds, _single_target_extrinsic1.target),
        (_single_target_extrinsic2.preds, _single_target_extrinsic2.target),
    ],
)
@pytest.mark.parametrize("average_method", ["min", "arithmetic", "geometric", "max"])
class TestAdjustedMutualInfoScore(MetricTester):
    """Test class for `AdjustedMutualInfoScore` metric."""

    atol = ATOL

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_adjusted_mutual_info_score(self, preds, target, average_method, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=AdjustedMutualInfoScore,
            reference_metric=partial(sklearn_ami, average_method=average_method),
            metric_args={"average_method": average_method},
        )

    def test_adjusted_mutual_info_score_functional(self, preds, target, average_method):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=adjusted_mutual_info_score,
            reference_metric=partial(sklearn_ami, average_method=average_method),
            average_method=average_method,
        )


@pytest.mark.parametrize("average_method", ["min", "geometric", "arithmetic", "max"])
def test_adjusted_mutual_info_score_functional_single_cluster(average_method):
    """Check that for single cluster the metric returns 0."""
    tensor_a = paddle.randint(low=0, high=NUM_CLASSES, shape=(BATCH_SIZE,))
    tensor_b = paddle.zeros((BATCH_SIZE,), dtype=paddle.int32)
    assert paddle.allclose(
        x=adjusted_mutual_info_score(tensor_a, tensor_b, average_method),
        y=paddle.tensor(0.0),
        atol=ATOL,
    ).item()
    assert paddle.allclose(
        x=adjusted_mutual_info_score(tensor_b, tensor_a, average_method),
        y=paddle.tensor(0.0),
        atol=ATOL,
    ).item()


@pytest.mark.parametrize("average_method", ["min", "geometric", "arithmetic", "max"])
def test_adjusted_mutual_info_score_functional_raises_invalid_task(average_method):
    """Check that metric rejects continuous-valued inputs."""
    preds, target = _float_inputs_extrinsic
    with pytest.raises(ValueError, match="Expected *"):
        adjusted_mutual_info_score(preds, target, average_method)


@pytest.mark.parametrize("average_method", ["min", "geometric", "arithmetic", "max"])
def test_adjusted_mutual_info_score_functional_is_symmetric(
    average_method,
    preds=_single_target_extrinsic1.preds,
    target=_single_target_extrinsic1.target,
):
    """Check that the metric functional is symmetric."""
    for p, t in zip(preds, target):
        assert paddle.allclose(
            x=adjusted_mutual_info_score(p, t, average_method),
            y=adjusted_mutual_info_score(t, p, average_method),
            atol=1e-06,
        ).item()
