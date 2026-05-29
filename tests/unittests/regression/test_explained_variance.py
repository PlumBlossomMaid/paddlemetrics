from functools import partial

import paddle
import pytest
from sklearn.metrics import explained_variance_score
from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

from paddlemetrics.functional import explained_variance
from paddlemetrics.regression import ExplainedVariance

seed_all(42)
NUM_TARGETS = 5
_single_target_inputs = _Input(
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE),
    target=paddle.rand(NUM_BATCHES, BATCH_SIZE),
)
_multi_target_inputs = _Input(
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE, NUM_TARGETS),
    target=paddle.rand(NUM_BATCHES, BATCH_SIZE, NUM_TARGETS),
)


def _single_target_ref_metric(preds, target, sk_fn=explained_variance_score):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()
    return sk_fn(sk_target, sk_preds)


def _multi_target_ref_metric(preds, target, sk_fn=explained_variance_score):
    sk_preds = preds.view(-1, NUM_TARGETS).numpy()
    sk_target = target.view(-1, NUM_TARGETS).numpy()
    return sk_fn(sk_target, sk_preds)


@pytest.mark.parametrize(
    "multioutput", ["raw_values", "uniform_average", "variance_weighted"]
)
@pytest.mark.parametrize(
    ("preds", "target", "ref_metric"),
    [
        (
            _single_target_inputs.preds,
            _single_target_inputs.target,
            _single_target_ref_metric,
        ),
        (
            _multi_target_inputs.preds,
            _multi_target_inputs.target,
            _multi_target_ref_metric,
        ),
    ],
)
class TestExplainedVariance(MetricTester):
    """Test class for `ExplainedVariance` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_explained_variance(self, multioutput, preds, target, ref_metric, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            ExplainedVariance,
            partial(
                ref_metric,
                sk_fn=partial(explained_variance_score, multioutput=multioutput),
            ),
            metric_args={"multioutput": multioutput},
        )

    def test_explained_variance_functional(
        self, multioutput, preds, target, ref_metric
    ):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            explained_variance,
            partial(
                ref_metric,
                sk_fn=partial(explained_variance_score, multioutput=multioutput),
            ),
            metric_args={"multioutput": multioutput},
        )

    def test_explained_variance_differentiability(
        self, multioutput, preds, target, ref_metric
    ):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=ExplainedVariance,
            metric_functional=explained_variance,
            metric_args={"multioutput": multioutput},
        )

    def test_explained_variance_half_cpu(self, multioutput, preds, target, ref_metric):
        """Test dtype support of the metric on CPU."""
        self.run_precision_test_cpu(
            preds, target, ExplainedVariance, explained_variance
        )

    @pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires cuda")
    def test_explained_variance_half_gpu(self, multioutput, preds, target, ref_metric):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds, target, ExplainedVariance, explained_variance
        )


def test_error_on_different_shape(metric_class=ExplainedVariance):
    """Test that error is raised on different shapes of input."""
    metric = metric_class()
    with pytest.raises(
        RuntimeError,
        match="Predictions and targets are expected to have the same shape",
    ):
        metric(paddle.randn(100), paddle.randn(50))
