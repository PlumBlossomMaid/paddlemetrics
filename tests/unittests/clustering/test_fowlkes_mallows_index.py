import pytest
from sklearn.metrics import \
    fowlkes_mallows_score as sklearn_fowlkes_mallows_score
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester
from unittests.clustering._inputs import (_single_target_extrinsic1,
                                          _single_target_extrinsic2)

from paddlemetrics.clustering import FowlkesMallowsIndex
from paddlemetrics.functional.clustering import fowlkes_mallows_index

seed_all(42)


@pytest.mark.parametrize(
    ("preds", "target"),
    [
        (_single_target_extrinsic1.preds, _single_target_extrinsic1.target),
        (_single_target_extrinsic2.preds, _single_target_extrinsic2.target),
    ],
)
class TestFowlkesMallowsIndex(MetricTester):
    """Test class for `FowlkesMallowsIndex` metric."""

    atol = 1e-05

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_fowlkes_mallows_index(self, preds, target, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=FowlkesMallowsIndex,
            reference_metric=sklearn_fowlkes_mallows_score,
        )

    def test_fowlkes_mallows_index_functional(self, preds, target):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=fowlkes_mallows_index,
            reference_metric=sklearn_fowlkes_mallows_score,
        )
