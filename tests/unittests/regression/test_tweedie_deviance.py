from functools import partial

import paddle
import pytest
from sklearn.metrics import mean_tweedie_deviance
from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

from paddlemetrics.functional.regression.tweedie_deviance import \
    tweedie_deviance_score
from paddlemetrics.regression.tweedie_deviance import TweedieDevianceScore

seed_all(42)
_single_target_inputs1 = _Input(
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE),
    target=paddle.rand(NUM_BATCHES, BATCH_SIZE),
)
_single_target_inputs2 = _Input(
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE),
    target=paddle.rand(NUM_BATCHES, BATCH_SIZE),
)
_multi_target_inputs = _Input(
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE, 5),
    target=paddle.rand(NUM_BATCHES, BATCH_SIZE, 5),
)


def _reference_sklearn_deviance(
    preds: paddle.Tensor, targets: paddle.Tensor, power: float
):
    sk_preds = preds.view(-1).numpy()
    sk_target = targets.view(-1).numpy()
    return mean_tweedie_deviance(sk_target, sk_preds, power=power)


@pytest.mark.parametrize("power", [-0.5, 0, 1, 1.5, 2, 3])
@pytest.mark.parametrize(
    ("preds", "target"),
    [
        (_single_target_inputs2.preds, _single_target_inputs2.target),
        (_single_target_inputs1.preds, _single_target_inputs1.target),
        (_multi_target_inputs.preds, _multi_target_inputs.target),
    ],
)
class TestDevianceScore(MetricTester):
    """Test class for `TweedieDevianceScore` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_deviance_scores_class(self, ddp, preds, target, power):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            TweedieDevianceScore,
            partial(_reference_sklearn_deviance, power=power),
            metric_args={"power": power},
        )

    def test_deviance_scores_functional(self, preds, target, power):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            tweedie_deviance_score,
            partial(_reference_sklearn_deviance, power=power),
            metric_args={"power": power},
        )

    def test_deviance_scores_differentiability(self, preds, target, power):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            preds,
            target,
            metric_module=TweedieDevianceScore,
            metric_functional=tweedie_deviance_score,
        )

    def test_deviance_scores_half_cpu(self, preds, target, power):
        """Test dtype support of the metric on CPU."""
        if power in [1, 2]:
            pytest.skip(
                "Tweedie Deviance Score half + cpu does not work for power=[1,2] due to missing support in paddle.log"
            )
        metric_args = {"power": power}
        self.run_precision_test_cpu(
            preds,
            target,
            metric_module=TweedieDevianceScore,
            metric_functional=tweedie_deviance_score,
            metric_args=metric_args,
        )

    @pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires cuda")
    def test_deviance_scores_half_gpu(self, preds, target, power):
        """Test dtype support of the metric on GPU."""
        metric_args = {"power": power}
        self.run_precision_test_gpu(
            preds,
            target,
            metric_module=TweedieDevianceScore,
            metric_functional=tweedie_deviance_score,
            metric_args=metric_args,
        )


def test_error_on_different_shape(metric_class=TweedieDevianceScore):
    """Test that error is raised on different shapes of input."""
    metric = metric_class()
    with pytest.raises(
        RuntimeError,
        match="Predictions and targets are expected to have the same shape",
    ):
        metric(paddle.randn(100), paddle.randn(50))


def test_error_on_invalid_inputs(metric_class=TweedieDevianceScore):
    """Test that error is raised on wrong argument combinations."""
    with pytest.raises(
        ValueError, match="Deviance Score is not defined for power=0.5."
    ):
        metric_class(power=0.5)
    metric = metric_class(power=1)
    with pytest.raises(
        ValueError,
        match="For power=1, 'preds' has to be strictly positive and 'targets' cannot be negative.",
    ):
        metric(paddle.tensor([-1.0, 2.0, 3.0]), paddle.rand(3))
    with pytest.raises(
        ValueError,
        match="For power=1, 'preds' has to be strictly positive and 'targets' cannot be negative.",
    ):
        metric(paddle.rand(3), paddle.tensor([-1.0, 2.0, 3.0]))
    metric = metric_class(power=2)
    with pytest.raises(
        ValueError,
        match="For power=2, both 'preds' and 'targets' have to be strictly positive.",
    ):
        metric(paddle.tensor([-1.0, 2.0, 3.0]), paddle.rand(3))
    with pytest.raises(
        ValueError,
        match="For power=2, both 'preds' and 'targets' have to be strictly positive.",
    ):
        metric(paddle.rand(3), paddle.tensor([-1.0, 2.0, 3.0]))


def test_corner_case_for_power_at_1(metric_class=TweedieDevianceScore):
    """Test that corner case for power=1.0 produce valid result."""
    metric = TweedieDevianceScore()
    targets = paddle.tensor([0, 1, 0, 1])
    preds = paddle.tensor([0.1, 0.1, 0.1, 0.1])
    val = metric(preds, targets)
    assert val != 0.0
    assert not paddle.isnan(val)
