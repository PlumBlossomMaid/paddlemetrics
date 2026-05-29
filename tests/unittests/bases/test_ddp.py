import os
from copy import deepcopy
from functools import partial

import paddle
from paddle import Tensor
import pytest
from unittests import NUM_PROCESSES, USE_PYTEST_POOL
from unittests._helpers import _IS_WINDOWS, seed_all
from unittests._helpers.testers import (DummyListMetric, DummyMetric,
                                        DummyMetricSum)
from unittests.conftest import setup_ddp

from paddlemetrics import Metric
from paddlemetrics.utils.exceptions import PaddleMetricsUserError

seed_all(42)

def gather_all_tensors(result: Tensor, group: Optional[Any] = None) -> List[Tensor]:
    """Gather all tensors from several ddp processes onto a list that is broadcast to all processes.

    Works on tensors that have the same number of dimensions, but where each dimension may differ. In this case
    tensors are padded, gathered and then trimmed to secure equal workload for all processes.

    Args:
        result: the value to sync
        group: the process group to gather results from. Defaults to all processes (world)

    Return:
        list with size equal to the process group where element i corresponds to result tensor from process i

    """
    if group is None:
        group = paddle.distributed.group.WORLD

    # convert tensors to contiguous format
    result = result.contiguous()

    world_size = paddle.distributed.get_world_size(group)
    paddle.distributed.barrier(group=group)

    # if the tensor is scalar, things are easy
    if result.ndim == 0:
        return _simple_gather_all_tensors(result, group, world_size)

    # 1. Gather sizes of all tensors
    local_size = paddle.to_tensor(result.shape, device=result.place)
    local_sizes = [paddle.zeros_like(local_size) for _ in range(world_size)]
    paddle.distributed.all_gather(local_sizes, local_size, group=group)
    max_size = paddle.stack(local_sizes).max(dim=0).values
    all_sizes_equal = all(all(ls == max_size) for ls in local_sizes)

    # 2. If shapes are all the same, then do a simple gather:
    if all_sizes_equal:
        return _simple_gather_all_tensors(result, group, world_size)

    # 3. If not, we need to pad each local tensor to maximum size, gather and then truncate
    with paddle.no_grad():
        pad_dims = []
        pad_by = (max_size - local_size).detach().cpu()
        for val in reversed(pad_by):
            pad_dims.append(0)
            pad_dims.append(val.item())
        result_padded = paddle.nn.functional.pad(result, pad_dims)
        gathered_result = [paddle.zeros_like(result_padded) for _ in range(world_size)]
        paddle.distributed.all_gather(gathered_result, result_padded, group)
        for idx, item_size in enumerate(local_sizes):
            slice_param = [slice(dim_size) for dim_size in item_size]
            gathered_result[idx] = gathered_result[idx][tuple(slice_param)]
    # to propagate autograd graph from local rank
    gathered_result[paddle.distributed.get_rank(group)] = result
    return gathered_result



def _test_ddp_sum(rank: int, worldsize: int = NUM_PROCESSES) -> None:
    dummy = DummyMetric()
    dummy._reductions = {"foo": paddle.sum}
    dummy.foo = paddle.tensor(1)
    dummy._sync_dist()
    assert dummy.foo == worldsize


def _test_ddp_cat(rank: int, worldsize: int = NUM_PROCESSES) -> None:
    dummy = DummyMetric()
    dummy._reductions = {"foo": paddle.cat}
    dummy.foo = [paddle.tensor([1])]
    dummy._sync_dist()
    assert paddle.all(paddle.eq(dummy.foo, paddle.tensor([1, 1])))


def _test_ddp_sum_cat(rank: int, worldsize: int = NUM_PROCESSES) -> None:
    dummy = DummyMetric()
    dummy._reductions = {"foo": paddle.cat, "bar": paddle.sum}
    dummy.foo = [paddle.tensor([1])]
    dummy.bar = paddle.tensor(1)
    dummy._sync_dist()
    assert paddle.all(paddle.eq(dummy.foo, paddle.tensor([1, 1])))
    assert dummy.bar == worldsize


def _test_ddp_gather_uneven_tensors(rank: int, worldsize: int = NUM_PROCESSES) -> None:
    tensor = paddle.ones(rank)
    result = gather_all_tensors(tensor)
    assert len(result) == worldsize
    for idx in range(worldsize):
        assert (result[idx] == paddle.ones_like(result[idx])).all()


def _test_ddp_gather_uneven_tensors_multidim(
    rank: int, worldsize: int = NUM_PROCESSES
) -> None:
    tensor = paddle.ones(rank + 1, 2 - rank)
    result = gather_all_tensors(tensor)
    assert len(result) == worldsize
    for idx in range(worldsize):
        val = result[idx]
        assert (val == paddle.ones_like(val)).all()


def _test_ddp_compositional_tensor(rank: int, worldsize: int = NUM_PROCESSES) -> None:
    dummy = DummyMetricSum()
    dummy._reductions = {"x": paddle.sum}
    dummy = dummy.clone() + dummy.clone()
    dummy.update(paddle.tensor(1))
    val = dummy.compute()
    assert val == 2 * worldsize


@pytest.mark.DDP
@pytest.mark.skipif(_IS_WINDOWS, reason="DDP not available on windows")
@pytest.mark.skipif(not USE_PYTEST_POOL, reason="DDP pool is not available.")
@pytest.mark.parametrize(
    "process",
    [
        _test_ddp_cat,
        _test_ddp_sum,
        _test_ddp_sum_cat,
        _test_ddp_gather_uneven_tensors,
        _test_ddp_gather_uneven_tensors_multidim,
        _test_ddp_compositional_tensor,
    ],
)
def test_ddp(process):
    """Test ddp functions."""
    pytest.pool.map(process, range(NUM_PROCESSES))


def _test_ddp_gather_all_autograd_same_shape(
    rank: int, worldsize: int = NUM_PROCESSES
) -> None:
    """Test that ddp gather preserves local rank's autograd graph for same-shaped tensors across ranks."""
    setup_ddp(rank, worldsize)
    x = (rank + 1) * paddle.ones(10, requires_grad=True)
    a, b = paddle.randn(1), paddle.randn(1)
    y = a * x + b
    result = gather_all_tensors(y)
    assert len(result) == worldsize
    grad = paddle.grad(outputs=result[rank].sum(), inputs=x)[0]
    assert paddle.allclose(x=grad, y=a * paddle.ones_like(x)).item()


def _test_ddp_gather_all_autograd_different_shape(
    rank: int, worldsize: int = NUM_PROCESSES
) -> None:
    """Test that ddp gather preserves local rank's autograd graph for differently-shaped tensors across ranks."""
    setup_ddp(rank, worldsize)
    x = (rank + 1) * paddle.ones(rank + 1, 2 - rank, requires_grad=True)
    a, b = paddle.randn(1), paddle.randn(1)
    y = a * x + b
    result = gather_all_tensors(y)
    assert len(result) == worldsize
    grad = paddle.grad(outputs=result[rank].sum(), inputs=x)[0]
    assert paddle.allclose(x=grad, y=a * paddle.ones_like(x)).item()


@pytest.mark.DDP
@pytest.mark.skipif(_IS_WINDOWS, reason="DDP not available on windows")
@pytest.mark.skipif(not USE_PYTEST_POOL, reason="DDP pool is not available.")
@pytest.mark.parametrize(
    "process",
    [
        _test_ddp_gather_all_autograd_same_shape,
        _test_ddp_gather_all_autograd_different_shape,
    ],
)
def test_ddp_autograd(process):
    """Test ddp functions for autograd compatibility."""
    pytest.pool.map(process, range(NUM_PROCESSES))


def _test_non_contiguous_tensors(rank):
    class DummyCatMetric(Metric):
        full_state_update = True

        def __init__(self) -> None:
            super().__init__()
            self.add_state("x", default=[], dist_reduce_fx=None)

        def update(self, x):
            self.x.append(x)

        def compute(self):
            x = paddle.concat(self.x, axis=0)
            return x.sum()

    metric = DummyCatMetric()
    metric.update(paddle.randn(10, 5)[:, 0])


@pytest.mark.DDP
@pytest.mark.skipif(_IS_WINDOWS, reason="DDP not available on windows")
@pytest.mark.skipif(not USE_PYTEST_POOL, reason="DDP pool is not available.")
def test_non_contiguous_tensors():
    """Test that gather_all operation works for non-contiguous tensors."""
    pytest.pool.map(_test_non_contiguous_tensors, range(NUM_PROCESSES))


def _test_state_dict_is_synced(rank, tmpdir):
    class DummyCatMetric(Metric):
        full_state_update = True

        def __init__(self) -> None:
            super().__init__()
            self.add_state("x", paddle.tensor(0), dist_reduce_fx=paddle.sum)
            self.add_state("c", paddle.tensor(0), dist_reduce_fx=paddle.sum)

        def update(self, x):
            self.x += x
            self.c += 1

        def compute(self):
            return self.x // self.c

        def __repr__(self) -> str:
            return f"DummyCatMetric(x={self.x}, c={self.c})"

    metric = DummyCatMetric()
    metric.persistent(True)

    def verify_metric(metric, i, world_size):
        state_dict = metric.state_dict()
        exp_sum = i * (i + 1) / 2
        assert state_dict["x"] == exp_sum * world_size
        assert metric.x == exp_sum * world_size
        assert metric.c == (i + 1) * world_size
        assert state_dict["c"] == metric.c

    steps = 5
    for i in range(steps):
        if metric._is_synced:
            with pytest.raises(
                PaddleMetricsUserError,
                match="The Metric shouldn't be synced when performing",
            ):
                metric(i)
            metric.unsync()
        metric(i)
        verify_metric(metric, i, 1)
        metric.sync()
        assert metric._is_synced
        with pytest.raises(
            PaddleMetricsUserError, match="The Metric has already been synced."
        ):
            metric.sync()
        verify_metric(metric, i, 2)
        metric.unsync()
        assert not metric._is_synced
        with pytest.raises(
            PaddleMetricsUserError, match="The Metric has already been un-synced."
        ):
            metric.unsync()
        with metric.sync_context():
            assert metric._is_synced
            verify_metric(metric, i, 2)
        with metric.sync_context(should_unsync=False):
            assert metric._is_synced
            verify_metric(metric, i, 2)
        assert metric._is_synced
        metric.unsync()
        assert not metric._is_synced
        metric.sync()
        cache = metric._cache
        metric._cache = None
        with pytest.raises(
            PaddleMetricsUserError,
            match="The internal cache should exist to unsync the Metric.",
        ):
            metric.unsync()
        metric._cache = cache

    def reload_state_dict(state_dict, expected_x, expected_c):
        metric = DummyCatMetric()
        metric.load_state_dict(state_dict)
        assert metric.x == expected_x
        assert metric.c == expected_c

    reload_state_dict(deepcopy(metric.state_dict()), 20, 10)
    metric.unsync()
    reload_state_dict(deepcopy(metric.state_dict()), 10, 5)
    metric.sync()
    filepath = os.path.join(tmpdir, f"weights-{rank}.pt")
    paddle.save(obj=metric.state_dict(), path=filepath)
    metric.unsync()
    with metric.sync_context():
        paddle.save(obj=metric.state_dict(), path=filepath)


@pytest.mark.DDP
@pytest.mark.skipif(_IS_WINDOWS, reason="DDP not available on windows")
@pytest.mark.skipif(not USE_PYTEST_POOL, reason="DDP pool is not available.")
def test_state_dict_is_synced(tmpdir):
    """Tests that metrics are synced while creating the state dict but restored after to continue accumulation."""
    pytest.pool.map(
        partial(_test_state_dict_is_synced, tmpdir=tmpdir), range(NUM_PROCESSES)
    )


def _test_sync_on_compute_tensor_state(rank, sync_on_compute):
    dummy = DummyMetricSum(sync_on_compute=sync_on_compute)
    dummy.update(paddle.tensor(rank + 1))
    val = dummy.compute()
    if sync_on_compute:
        assert val == 3
    else:
        assert val == rank + 1


def _test_sync_on_compute_list_state(rank, sync_on_compute):
    dummy = DummyListMetric(sync_on_compute=sync_on_compute)
    dummy.update(paddle.tensor(rank + 1))
    val = dummy.compute()
    if sync_on_compute:
        assert val.sum() == 3
        assert (
            paddle.allclose(x=val, y=paddle.tensor([1, 2])).item()
            or paddle.allclose(x=val, y=paddle.tensor([2, 1])).item()
        )
    else:
        assert val == [paddle.tensor(rank + 1)]


@pytest.mark.DDP
@pytest.mark.skipif(_IS_WINDOWS, reason="DDP not available on windows")
@pytest.mark.skipif(not USE_PYTEST_POOL, reason="DDP pool is not available.")
@pytest.mark.parametrize("sync_on_compute", [True, False])
@pytest.mark.parametrize(
    "test_func", [_test_sync_on_compute_list_state, _test_sync_on_compute_tensor_state]
)
def test_sync_on_compute(sync_on_compute, test_func):
    """Test that synchronization of states can be enabled and disabled for compute."""
    pytest.pool.map(
        partial(test_func, sync_on_compute=sync_on_compute), range(NUM_PROCESSES)
    )


def _test_sync_with_empty_lists(rank):
    dummy = DummyListMetric()
    val = dummy.compute()
    assert paddle.allclose(x=val, y=paddle.tensor([])).item()


@pytest.mark.DDP
@pytest.mark.skipif(
    not True, reason="test only works on newer torch versions"
)
@pytest.mark.skipif(_IS_WINDOWS, reason="DDP not available on windows")
@pytest.mark.skipif(not USE_PYTEST_POOL, reason="DDP pool is not available.")
def test_sync_with_empty_lists():
    """Test that synchronization of states can be enabled and disabled for compute."""
    pytest.pool.map(_test_sync_with_empty_lists, range(NUM_PROCESSES))


def _test_sync_with_unequal_size_lists(rank):
    """Test that synchronization of list states work even when some ranks have not received any data yet."""
    dummy = DummyListMetric()
    if rank == 0:
        dummy.update(paddle.zeros(2))
    assert paddle.all(dummy.compute() == paddle.tensor([0.0, 0.0]))


@pytest.mark.DDP
@pytest.mark.skipif(
    not True, reason="test only works on newer torch versions"
)
@pytest.mark.skipif(_IS_WINDOWS, reason="DDP not available on windows")
@pytest.mark.skipif(not USE_PYTEST_POOL, reason="DDP pool is not available.")
def test_sync_with_unequal_size_lists():
    """Test that synchronization of states can be enabled and disabled for compute."""
    pytest.pool.map(_test_sync_with_unequal_size_lists, range(NUM_PROCESSES))
