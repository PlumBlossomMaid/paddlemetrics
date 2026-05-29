import pytest
from sklearn.metrics import \
    calinski_harabasz_score as sklearn_calinski_harabasz_score
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester
from unittests.clustering._inputs import (_single_target_intrinsic1,
                                          _single_target_intrinsic2)

from paddlemetrics.clustering.calinski_harabasz_score import \
    CalinskiHarabaszScore
from paddlemetrics.functional.clustering.calinski_harabasz_score import \
    calinski_harabasz_score

seed_all(42)


@pytest.mark.parametrize(
    ("data", "labels"),
    [
        (_single_target_intrinsic1.data, _single_target_intrinsic1.labels),
        (_single_target_intrinsic2.data, _single_target_intrinsic2.labels),
    ],
)
class TestCalinskiHarabaszScore(MetricTester):
    """Test class for `CalinskiHarabaszScore` metric."""

    atol = 1e-05

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_calinski_harabasz_score(self, data, labels, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=data,
            target=labels,
            metric_class=CalinskiHarabaszScore,
            reference_metric=sklearn_calinski_harabasz_score,
        )

    def test_calinski_harabasz_score_functional(self, data, labels):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=data,
            target=labels,
            metric_functional=calinski_harabasz_score,
            reference_metric=sklearn_calinski_harabasz_score,
        )
