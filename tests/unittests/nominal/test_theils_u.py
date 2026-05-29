import itertools
from functools import partial

import paddle
import pytest
from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests._helpers.testers import MetricTester

from paddlemetrics.functional.nominal.theils_u import theils_u, theils_u_matrix
from paddlemetrics.nominal import TheilsU

NUM_CLASSES = 4
_input_default = _Input(
    preds=paddle.randint(low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE)),
    target=paddle.randint(low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE)),
)
_preds = paddle.randint(
    low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE), dtype=paddle.float32
)
_preds[0, 0] = float("nan")
_preds[-1, -1] = float("nan")
_target = paddle.randint(
    low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE), dtype=paddle.float32
)
_target[1, 0] = float("nan")
_target[-1, 0] = float("nan")
_input_with_nans = _Input(preds=_preds, target=_target)
_input_logits = _Input(
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES),
    target=paddle.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES),
)


@pytest.fixture
def theils_u_matrix_input():
    """Define input in matrix format for the metric."""
    matrix = paddle.concat(
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
    matrix[0, 0] = float("nan")
    matrix[-1, -1] = float("nan")
    return matrix


def _reference_dython_theils_u(preds, target, nan_strategy, nan_replace_value):
    try:
        from dython.nominal import theils_u as dython_theils_u
    except ImportError:
        pytest.skip("Test requires `dython` package to be installed.")
    preds = preds.argmax(1) if preds.ndim == 2 else preds
    target = target.argmax(1) if target.ndim == 2 else target
    v = dython_theils_u(
        preds.numpy(),
        target.numpy(),
        nan_strategy=nan_strategy,
        nan_replace_value=nan_replace_value,
    )
    return paddle.tensor(v)


def _reference_dython_theils_u_matrix(matrix, nan_strategy, nan_replace_value):
    num_variables = matrix.shape[1]
    theils_u_matrix_value = paddle.ones(num_variables, num_variables)
    for i, j in itertools.combinations(range(num_variables), 2):
        x, y = matrix[:, i], matrix[:, j]
        theils_u_matrix_value[i, j] = _reference_dython_theils_u(
            x, y, nan_strategy, nan_replace_value
        )
        theils_u_matrix_value[j, i] = _reference_dython_theils_u(
            y, x, nan_strategy, nan_replace_value
        )
    return theils_u_matrix_value


@pytest.mark.parametrize(
    ("preds", "target"),
    [
        (_input_default.preds, _input_default.target),
        (_input_with_nans.preds, _input_with_nans.target),
        (_input_logits.preds, _input_logits.target),
    ],
)
@pytest.mark.parametrize(
    ("nan_strategy", "nan_replace_value"), [("replace", 0.0), ("drop", None)]
)
class TestTheilsU(MetricTester):
    """Test class for `TheilsU` metric."""

    atol = 1e-05

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_theils_u(self, ddp, preds, target, nan_strategy, nan_replace_value):
        """Test class implementation of metric."""
        metric_args = {
            "nan_strategy": nan_strategy,
            "nan_replace_value": nan_replace_value,
            "num_classes": NUM_CLASSES,
        }
        reference_metric = partial(
            _reference_dython_theils_u,
            nan_strategy=nan_strategy,
            nan_replace_value=nan_replace_value,
        )
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=TheilsU,
            reference_metric=reference_metric,
            metric_args=metric_args,
        )

    def test_theils_u_functional(self, preds, target, nan_strategy, nan_replace_value):
        """Test functional implementation of metric."""
        metric_args = {
            "nan_strategy": nan_strategy,
            "nan_replace_value": nan_replace_value,
        }
        reference_metric = partial(
            _reference_dython_theils_u,
            nan_strategy=nan_strategy,
            nan_replace_value=nan_replace_value,
        )
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=theils_u,
            reference_metric=reference_metric,
            metric_args=metric_args,
        )

    def test_theils_u_differentiability(
        self, preds, target, nan_strategy, nan_replace_value
    ):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        metric_args = {
            "nan_strategy": nan_strategy,
            "nan_replace_value": nan_replace_value,
            "num_classes": NUM_CLASSES,
        }
        self.run_differentiability_test(
            preds,
            target,
            metric_module=TheilsU,
            metric_functional=theils_u,
            metric_args=metric_args,
        )


@pytest.mark.parametrize(
    ("nan_strategy", "nan_replace_value"), [("replace", 1.0), ("drop", None)]
)
def test_theils_u_matrix(theils_u_matrix_input, nan_strategy, nan_replace_value):
    """Test matrix version of metric works as expected."""
    tm_score = theils_u_matrix(theils_u_matrix_input, nan_strategy, nan_replace_value)
    reference_score = _reference_dython_theils_u_matrix(
        theils_u_matrix_input, nan_strategy, nan_replace_value
    )
    assert paddle.allclose(x=tm_score, y=reference_score, atol=1e-06).item()
