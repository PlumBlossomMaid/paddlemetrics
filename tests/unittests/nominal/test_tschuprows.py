import itertools

import paddle
import pandas as pd
import pytest
from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests._helpers.testers import MetricTester

from paddlemetrics.functional.nominal.tschuprows import (tschuprows_t,
                                                        tschuprows_t_matrix)
from paddlemetrics.nominal.tschuprows import TschuprowsT

NUM_CLASSES = 4
_input_default = _Input(
    preds=paddle.randint(low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE)),
    target=paddle.randint(low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE)),
)
_input_logits = _Input(
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES),
    target=paddle.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES),
)


@pytest.fixture
def tschuprows_matrix_input():
    """Define input in matrix format for the metric."""
    return paddle.concat(
        [
            paddle.randint(
                low=0,
                high=NUM_CLASSES,
                shape=(NUM_BATCHES * BATCH_SIZE, 1),
                dtype=paddle.float32,
            ),
            paddle.randint(
                low=0,
                high=NUM_CLASSES + 2,
                shape=(NUM_BATCHES * BATCH_SIZE, 1),
                dtype=paddle.float32,
            ),
            paddle.randint(
                low=0, high=2, shape=(NUM_BATCHES * BATCH_SIZE, 1), dtype=paddle.float32
            ),
        ], axis=-1,
    )


def _reference_pd_tschuprows_t(preds, target):
    try:
        from scipy.stats.contingency import association
    except ImportError:
        pytest.skip("test requires scipy package to be installed")
    preds = preds.argmax(1) if preds.ndim == 2 else preds
    target = target.argmax(1) if target.ndim == 2 else target
    preds, target = preds.numpy().astype(int), target.numpy().astype(int)
    observed_values = pd.crosstab(preds, target)
    t = association(observed=observed_values, method="tschuprow")
    return paddle.tensor(t)


def _reference_pd_tschuprows_t_matrix(matrix):
    num_variables = matrix.shape[1]
    tschuprows_t_matrix_value = paddle.ones(num_variables, num_variables)
    for i, j in itertools.combinations(range(num_variables), 2):
        x, y = matrix[:, i], matrix[:, j]
        tschuprows_t_matrix_value[i, j] = tschuprows_t_matrix_value[
            j, i
        ] = _reference_pd_tschuprows_t(x, y)
    return tschuprows_t_matrix_value


@pytest.mark.parametrize(
    ("preds", "target"),
    [
        (_input_default.preds, _input_default.target),
        (_input_logits.preds, _input_logits.target),
    ],
)
class TestTschuprowsT(MetricTester):
    """Test class for `TschuprowsT` metric."""

    atol = 1e-05

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_tschuprows_ta(self, ddp, preds, target):
        """Test class implementation of metric."""
        metric_args = {"bias_correction": False, "num_classes": NUM_CLASSES}
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=TschuprowsT,
            reference_metric=_reference_pd_tschuprows_t,
            metric_args=metric_args,
        )

    def test_tschuprows_t_functional(self, preds, target):
        """Test functional implementation of metric."""
        metric_args = {"bias_correction": False}
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=tschuprows_t,
            reference_metric=_reference_pd_tschuprows_t,
            metric_args=metric_args,
        )

    def test_tschuprows_t_differentiability(self, preds, target):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        metric_args = {"bias_correction": False, "num_classes": NUM_CLASSES}
        self.run_differentiability_test(
            preds,
            target,
            metric_module=TschuprowsT,
            metric_functional=tschuprows_t,
            metric_args=metric_args,
        )


def test_tschuprows_t_matrix(tschuprows_matrix_input):
    """Test matrix version of metric works as expected."""
    tm_score = tschuprows_t_matrix(tschuprows_matrix_input, bias_correction=False)
    reference_score = _reference_pd_tschuprows_t_matrix(tschuprows_matrix_input)
    assert paddle.allclose(x=tm_score, y=reference_score).item()
