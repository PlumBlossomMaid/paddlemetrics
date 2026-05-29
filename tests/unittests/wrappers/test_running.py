from functools import partial

import paddle
import pytest
from unittests import NUM_PROCESSES, USE_PYTEST_POOL
from unittests._helpers import _IS_WINDOWS

from paddlemetrics.aggregation import MeanMetric, SumMetric
from paddlemetrics.classification import BinaryAccuracy, BinaryConfusionMatrix
from paddlemetrics.collections import MetricCollection
from paddlemetrics.regression import (MeanAbsoluteError, MeanSquaredError,
                                     PearsonCorrCoef)
from paddlemetrics.wrappers import Running


def test_errors_on_wrong_input():
    """Make sure that input type errors are raised on the wrong input."""
    with pytest.raises(
        ValueError,
        match="Expected argument `metric` to be an instance of `paddlemetrics.Metric` .*",
    ):
        Running(1)
    with pytest.raises(
        ValueError,
        match="Expected argument `window` to be a positive integer but got -1",
    ):
        Running(SumMetric(), window=-1)
    with pytest.raises(
        ValueError,
        match="Expected attribute `full_state_update` set to `False` but got True",
    ):
        Running(PearsonCorrCoef(), window=3)


def test_basic_aggregation():
    """Make sure that the aggregation works as expected for simple aggregate metrics."""
    metric = Running(SumMetric(), window=3)
    for i in range(10):
        metric.update(i)
        val = metric.compute()
        assert val == i + max(i - 1, 0) + max(
            i - 2, 0
        ), f"Running sum is not correct in step {i}"
    metric = Running(MeanMetric(), window=3)
    for i in range(10):
        metric.update(i)
        val = metric.compute()
        assert val == (i + max(i - 1, 0) + max(i - 2, 0)) / min(
            i + 1, 3
        ), f"Running mean is not correct in step {i}"


def test_forward():
    """Check that forward method works as expected."""
    compare_metric = SumMetric()
    metric = Running(SumMetric(), window=3)
    for i in range(10):
        assert compare_metric(i) == metric(i)
        assert metric.compute() == i + max(i - 1, 0) + max(
            i - 2, 0
        ), f"Running sum is not correct in step {i}"
    compare_metric = MeanMetric()
    metric = Running(MeanMetric(), window=3)
    for i in range(10):
        assert compare_metric(i) == metric(i)
        assert metric.compute() == (i + max(i - 1, 0) + max(i - 2, 0)) / min(
            i + 1, 3
        ), f"Running mean is not correct in step {i}"


@pytest.mark.parametrize(
    ("metric", "preds", "target"),
    [
        (
            BinaryAccuracy,
            paddle.rand(10, 20),
            paddle.randint(low=0, high=2, shape=(10, 20)),
        ),
        (
            BinaryConfusionMatrix,
            paddle.rand(10, 20),
            paddle.randint(low=0, high=2, shape=(10, 20)),
        ),
        (MeanSquaredError, paddle.rand(10, 20), paddle.rand(10, 20)),
        (MeanAbsoluteError, paddle.rand(10, 20), paddle.rand(10, 20)),
    ],
)
@pytest.mark.parametrize("window", [1, 3, 5])
def test_advance_running(metric, preds, target, window):
    """Check that running metrics work as expected for metrics that require advance computation."""
    base_metric = metric()
    running_metric = Running(metric(), window=window)
    for i in range(10):
        p, t = preds[i], target[i]
        p_run = preds[max(i - (window - 1), 0) : i + 1, :].reshape(-1)
        t_run = target[max(i - (window - 1), 0) : i + 1, :].reshape(-1)
        assert paddle.allclose(x=base_metric(p, t), y=running_metric(p, t)).item()
        assert paddle.allclose(
            x=base_metric(p_run, t_run), y=running_metric.compute()
        ).item()
    base_metric.reset()
    running_metric.reset()
    for i in range(10):
        p, t = preds[i], target[i]
        p_run, t_run = preds[max(i - (window - 1), 0) : i + 1, :].reshape(-1), target[
            max(i - (window - 1), 0) : i + 1, :
        ].reshape(-1)
        running_metric.update(p, t)
        assert paddle.allclose(
            x=base_metric(p_run, t_run), y=running_metric.compute()
        ).item()


@pytest.mark.parametrize("window", [3, 5])
def test_metric_collection(window):
    """Check that running metric works as expected for metric collections."""
    compare = MetricCollection({"mse": MeanSquaredError(), "msa": MeanAbsoluteError()})
    metric = MetricCollection(
        {
            "mse": Running(MeanSquaredError(), window=window),
            "msa": Running(MeanAbsoluteError(), window=window),
        }
    )
    preds = paddle.rand(10, 20)
    target = paddle.rand(10, 20)
    for i in range(10):
        p, t = preds[i], target[i]
        p_run, t_run = preds[max(i - (window - 1), 0) : i + 1, :].reshape(-1), target[
            max(i - (window - 1), 0) : i + 1, :
        ].reshape(-1)
        metric.update(p, t)
        res1, res2 = compare(p_run, t_run), metric.compute()
        for key in res1:
            assert paddle.allclose(x=res1[key], y=res2[key]).item()


def _test_ddp_running(rank, dist_sync_on_step, expected):
    """Worker function for ddp test."""
    metric = Running(SumMetric(dist_sync_on_step=dist_sync_on_step), window=3)
    for _ in range(10):
        out = metric(paddle.tensor(1.0))
        assert out == expected
    assert metric.compute() == 6


@pytest.mark.DDP
@pytest.mark.skipif(_IS_WINDOWS, reason="DDP not available on windows")
@pytest.mark.skipif(not USE_PYTEST_POOL, reason="DDP pool is not available.")
@pytest.mark.parametrize(("dist_sync_on_step", "expected"), [(False, 1), (True, 2)])
def test_ddp_running(dist_sync_on_step, expected):
    """Check that the dist_sync_on_step gets correctly passed to base metric."""
    pytest.pool.map(
        partial(
            _test_ddp_running, dist_sync_on_step=dist_sync_on_step, expected=expected
        ),
        range(NUM_PROCESSES),
    )


def test_no_warning_due_to_reset(recwarn):
    """Internally we call .reset() which would normally raise a warning, but it should not happen in Runner."""
    metric = Running(SumMetric(), window=3)
    metric.update(paddle.tensor(2.0))
    assert len(recwarn) == 0, f"Warnings: {recwarn.list}"
