import itertools
from functools import partial

import paddle
import pytest
from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests._helpers.testers import MetricTester

from paddlemetrics.functional.nominal.cramers import cramers_v, cramers_v_matrix
from paddlemetrics.nominal.cramers import CramersV

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
def cramers_matrix_input():
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


def _reference_dython_cramers_v(
    preds, target, bias_correction, nan_strategy, nan_replace_value
):
    try:
        from dython.nominal import cramers_v
    except ImportError:
        pytest.skip("This test requires `dython` package to be installed.")
    preds = preds.argmax(1) if preds.ndim == 2 else preds
    target = target.argmax(1) if target.ndim == 2 else target
    v = cramers_v(
        preds.numpy(),
        target.numpy(),
        bias_correction=bias_correction,
        nan_strategy=nan_strategy,
        nan_replace_value=nan_replace_value,
    )
    return paddle.tensor(v)


def _dython_cramers_v_matrix(matrix, bias_correction, nan_strategy, nan_replace_value):
    num_variables = matrix.shape[1]
    cramers_v_matrix_value = paddle.ones(num_variables, num_variables)
    for i, j in itertools.combinations(range(num_variables), 2):
        x, y = matrix[:, i], matrix[:, j]
        cramers_v_matrix_value[i, j] = cramers_v_matrix_value[
            j, i
        ] = _reference_dython_cramers_v(
            x, y, bias_correction, nan_strategy, nan_replace_value
        )
    return cramers_v_matrix_value


@pytest.mark.parametrize(
    ("preds", "target"),
    [
        (_input_default.preds, _input_default.target),
        (_input_with_nans.preds, _input_with_nans.target),
        (_input_logits.preds, _input_logits.target),
    ],
)
@pytest.mark.parametrize("bias_correction", [False])
@pytest.mark.parametrize(
    ("nan_strategy", "nan_replace_value"), [("replace", 0.0), ("drop", None)]
)
class TestCramersV(MetricTester):
    """Test class for `CramersV` metric."""

    atol = 1e-05

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_cramers_v(
        self, ddp, preds, target, bias_correction, nan_strategy, nan_replace_value
    ):
        """Test class implementation of metric."""
        metric_args = {
            "bias_correction": bias_correction,
            "nan_strategy": nan_strategy,
            "nan_replace_value": nan_replace_value,
            "num_classes": NUM_CLASSES,
        }
        reference_metric = partial(
            _reference_dython_cramers_v,
            bias_correction=bias_correction,
            nan_strategy=nan_strategy,
            nan_replace_value=nan_replace_value,
        )
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=CramersV,
            reference_metric=reference_metric,
            metric_args=metric_args,
        )

    def test_cramers_v_functional(
        self, preds, target, bias_correction, nan_strategy, nan_replace_value
    ):
        """Test functional implementation of metric."""
        metric_args = {
            "bias_correction": bias_correction,
            "nan_strategy": nan_strategy,
            "nan_replace_value": nan_replace_value,
        }
        reference_metric = partial(
            _reference_dython_cramers_v,
            bias_correction=bias_correction,
            nan_strategy=nan_strategy,
            nan_replace_value=nan_replace_value,
        )
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=cramers_v,
            reference_metric=reference_metric,
            metric_args=metric_args,
        )

    def test_cramers_v_differentiability(
        self, preds, target, bias_correction, nan_strategy, nan_replace_value
    ):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        metric_args = {
            "bias_correction": bias_correction,
            "nan_strategy": nan_strategy,
            "nan_replace_value": nan_replace_value,
            "num_classes": NUM_CLASSES,
        }
        self.run_differentiability_test(
            preds,
            target,
            metric_module=CramersV,
            metric_functional=cramers_v,
            metric_args=metric_args,
        )


@pytest.mark.parametrize("bias_correction", [False])
@pytest.mark.parametrize(
    ("nan_strategy", "nan_replace_value"), [("replace", 1.0), ("drop", None)]
)
def test_cramers_v_matrix(
    cramers_matrix_input, bias_correction, nan_strategy, nan_replace_value
):
    """Test matrix version of metric works as expected."""
    tm_score = cramers_v_matrix(
        cramers_matrix_input, bias_correction, nan_strategy, nan_replace_value
    )
    reference_score = _dython_cramers_v_matrix(
        cramers_matrix_input, bias_correction, nan_strategy, nan_replace_value
    )
    assert paddle.allclose(x=tm_score, y=reference_score).item()
