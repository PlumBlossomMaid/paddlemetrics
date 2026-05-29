from functools import partial

import paddle
import pytest
from unittests import NUM_PROCESSES, USE_PYTEST_POOL
from unittests._helpers import _IS_WINDOWS
from unittests._helpers.testers import _assert_allclose, _assert_tensor
from unittests.conftest import setup_ddp
from unittests.text._inputs import (_inputs_squad_batch_match,
                                    _inputs_squad_exact_match,
                                    _inputs_squad_exact_mismatch)

from paddlemetrics.functional.text import squad
from paddlemetrics.text.squad import SQuAD


@pytest.mark.parametrize(
    ("preds", "targets", "exact_match", "f1"),
    [
        (
            _inputs_squad_exact_match.preds,
            _inputs_squad_exact_match.target,
            _inputs_squad_exact_match.exact_match,
            _inputs_squad_exact_match.f1,
        ),
        (
            _inputs_squad_exact_mismatch.preds,
            _inputs_squad_exact_mismatch.target,
            _inputs_squad_exact_mismatch.exact_match,
            _inputs_squad_exact_mismatch.f1,
        ),
    ],
)
def test_score_fn(preds, targets, exact_match, f1):
    """Tests for functional."""
    metrics_score = squad(preds, targets)
    _assert_tensor(metrics_score["exact_match"])
    _assert_tensor(metrics_score["f1"])
    _assert_allclose(metrics_score["exact_match"], exact_match)
    _assert_allclose(metrics_score["f1"], f1)


@pytest.mark.parametrize(
    ("preds", "targets", "exact_match", "f1"),
    [
        (
            _inputs_squad_batch_match.preds,
            _inputs_squad_batch_match.target,
            _inputs_squad_batch_match.exact_match,
            _inputs_squad_batch_match.f1,
        )
    ],
)
def test_accumulation(preds, targets, exact_match, f1):
    """Tests for metric works with accumulation."""
    squad_metric = SQuAD()
    for pred, target in zip(preds, targets):
        squad_metric.update(preds=[pred], target=[target])
    metrics_score = squad_metric.compute()
    _assert_tensor(metrics_score["exact_match"])
    _assert_tensor(metrics_score["f1"])
    _assert_allclose(
        metrics_score["exact_match"], paddle.mean(paddle.tensor(exact_match))
    )
    _assert_allclose(metrics_score["f1"], paddle.mean(paddle.tensor(f1)))


def _squad_score_ddp(rank, world_size, pred, targets, exact_match, f1):
    """Define a DDP process for SQuAD metric."""
    setup_ddp(rank, world_size)
    squad_metric = SQuAD()
    squad_metric.update(pred, targets)
    metrics_score = squad_metric.compute()
    _assert_tensor(metrics_score["exact_match"])
    _assert_tensor(metrics_score["f1"])
    _assert_allclose(metrics_score["exact_match"], exact_match)
    _assert_allclose(metrics_score["f1"], f1)


def _test_score_ddp_fn(rank, world_size, preds, targets, exact_match, f1):
    """Core functionality for the `test_score_ddp` test."""
    mean_exact_match = paddle.tensor(exact_match, dtype=paddle.float32).mean()
    mean_f1 = paddle.tensor(f1, dtype=paddle.float32).mean()
    _squad_score_ddp(
        rank, world_size, [preds[rank]], [targets[rank]], mean_exact_match, mean_f1
    )


@pytest.mark.parametrize(
    ("preds", "targets", "exact_match", "f1"),
    [
        (
            _inputs_squad_batch_match.preds,
            _inputs_squad_batch_match.target,
            _inputs_squad_batch_match.exact_match,
            _inputs_squad_batch_match.f1,
        )
    ],
)
@pytest.mark.skipif(not USE_PYTEST_POOL, reason="DDP pool is not available")
@pytest.mark.skipif(_IS_WINDOWS, reason="DDP not supported on Windows")
@pytest.mark.DDP
def test_score_ddp(preds, targets, exact_match, f1):
    """Tests for metric using DDP."""
    pytest.pool.map(
        partial(
            _test_score_ddp_fn,
            world_size=NUM_PROCESSES,
            preds=preds,
            targets=targets,
            exact_match=exact_match,
            f1=f1,
        ),
        range(NUM_PROCESSES),
    )
