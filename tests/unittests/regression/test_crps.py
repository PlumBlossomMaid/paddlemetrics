import paddle
import pytest
from properscoring import crps_ensemble
from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

from paddlemetrics.functional.regression.crps import \
    continuous_ranked_probability_score
from paddlemetrics.regression.crps import ContinuousRankedProbabilityScore

seed_all(42)
_input_10ensemble = _Input(
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE, 10),
    target=paddle.rand(NUM_BATCHES, BATCH_SIZE),
)
_input2_5ensemble = _Input(
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE, 5),
    target=paddle.rand(NUM_BATCHES, BATCH_SIZE),
)


def _reference_implementation(preds, target):
    sk_preds = preds.numpy()
    sk_target = target.numpy()
    return crps_ensemble(sk_target, sk_preds).mean()


@pytest.mark.parametrize(
    ("preds", "target"),
    [
        (_input2_5ensemble.preds, _input2_5ensemble.target),
        (_input_10ensemble.preds, _input_10ensemble.target),
    ],
)
class TestContinuousRankedProbabilityScore(MetricTester):
    """Test class for `ContinuousRankedProbabilityScore` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_continuous_ranked_probability_score(self, preds, target, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=ContinuousRankedProbabilityScore,
            reference_metric=_reference_implementation,
        )

    def test_continuous_ranked_probability_score_functional(self, preds, target):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=continuous_ranked_probability_score,
            reference_metric=_reference_implementation,
        )


def test_error_on_different_shape(metric_class=ContinuousRankedProbabilityScore):
    """Test that error is raised on different shapes of input."""
    metric = metric_class()
    with pytest.raises(
        RuntimeError,
        match="Predictions and targets are expected to have the same shape",
    ):
        metric(paddle.randn(100, 5), paddle.randn(50))


def test_error_on_single_ensemble_member():
    """Test that error is raised on single ensemble member."""
    metric = ContinuousRankedProbabilityScore()
    with pytest.raises(
        ValueError, match="CRPS requires at least 2 ensemble members, but.*"
    ):
        metric(paddle.randn(100, 1), paddle.randn(100))
