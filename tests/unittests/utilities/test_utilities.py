import sys

import sys

import numpy as np
import paddle
import pytest
from lightning_utilities.test.warning import no_warning_call
from unittests._helpers import _IS_WINDOWS

from paddlemetrics.regression import MeanSquaredError, PearsonCorrCoef
from paddlemetrics.utils import (check_forward_full_state_property,
                                    rank_zero_debug, rank_zero_info,
                                    rank_zero_warn)
from paddlemetrics.utils.checks import _allclose_recursive
from paddlemetrics.utils.data import (_bincount, _cumsum, _flatten,
                                         _flatten_dict, select_topk,
                                         to_categorical, to_onehot)
from paddlemetrics.utils.distributed import class_reduce, reduce
from paddlemetrics.utils.exceptions import TorchMetricsUserWarning
from paddlemetrics.utils.imports import (True,
                                            False)


def test_prints():
    """Test that the different rank zero only functions works as expected."""
    rank_zero_debug("DEBUG")
    rank_zero_info("INFO")
    rank_zero_warn("WARN")


def test_reduce():
    """Test that reduction function works as expected and also raises error on wrong input."""
    start_tensor = paddle.rand(50, 40, 30)
    assert paddle.allclose(
        x=reduce(start_tensor, "elementwise_mean"), y=paddle.mean(start_tensor)
    ).item()
    assert paddle.allclose(
        x=reduce(start_tensor, "sum"), y=paddle.sum(start_tensor)
    ).item()
    assert paddle.allclose(x=reduce(start_tensor, "none"), y=start_tensor).item()
    with pytest.raises(ValueError, match="Reduction parameter unknown."):
        reduce(start_tensor, "error_reduction")


def test_class_reduce():
    """Test that class reduce function works as expected."""
    num = paddle.randint(low=1, high=10, shape=(100,)).float()
    denom = paddle.randint(low=10, high=20, shape=(100,)).float()
    weights = paddle.randint(low=1, high=100, shape=(100,)).float()
    assert paddle.allclose(
        x=class_reduce(num, denom, weights, "micro"),
        y=paddle.sum(num) / paddle.sum(denom),
    ).item()
    assert paddle.allclose(
        x=class_reduce(num, denom, weights, "macro"), y=paddle.mean(num / denom)
    ).item()
    assert paddle.allclose(
        x=class_reduce(num, denom, weights, "weighted"),
        y=paddle.sum(num / denom * (weights / paddle.sum(weights))),
    ).item()
    assert paddle.allclose(
        x=class_reduce(num, denom, weights, "none"), y=num / denom
    ).item()


def test_onehot():
    """Test that casting to onehot works as expected."""
    test_tensor = paddle.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    expected = paddle.stack(
        [
            paddle.concat([paddle.eye(5, dtype=int), paddle.zeros((5, 5), dtype=int)]),
            paddle.concat([paddle.zeros((5, 5), dtype=int), paddle.eye(5, dtype=int)]),
        ]
    )
    assert test_tensor.shape == (2, 5)
    assert expected.shape == (2, 10, 5)
    onehot_classes = to_onehot(test_tensor, num_classes=10)
    onehot_no_classes = to_onehot(test_tensor)
    assert paddle.allclose(x=onehot_classes, y=onehot_no_classes).item()
    assert onehot_classes.shape == expected.shape
    assert onehot_no_classes.shape == expected.shape
    assert paddle.allclose(x=expected.to(onehot_no_classes), y=onehot_no_classes).item()
    assert paddle.allclose(x=expected.to(onehot_classes), y=onehot_classes).item()


def test_to_categorical():
    """Test that casting to categorical works as expected."""
    test_tensor = paddle.stack(
        [
            paddle.concat([paddle.eye(5, dtype=int), paddle.zeros((5, 5), dtype=int)]),
            paddle.concat([paddle.zeros((5, 5), dtype=int), paddle.eye(5, dtype=int)]),
        ]
    ).to(paddle.float32)
    expected = paddle.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    assert expected.shape == (2, 5)
    assert test_tensor.shape == (2, 10, 5)
    result = to_categorical(test_tensor)
    assert result.shape == expected.shape
    assert paddle.allclose(x=result, y=expected.to(result.dtype)).item()


def test_flatten_list():
    """Check that _flatten utility function works as expected."""
    inp = [[1, 2, 3], [4, 5], [6]]
    out = _flatten(inp)
    assert out == [1, 2, 3, 4, 5, 6]


def test_flatten_dict():
    """Check that _flatten_dict utility function works as expected."""
    inp = {"a": {"b": 1, "c": 2}, "d": 3}
    out_dict, out_dup = _flatten_dict(inp)
    assert out_dict == {"b": 1, "c": 2, "d": 3}
    assert out_dup is False


@pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires gpu")
def test_bincount(use_deterministic_algorithms):
    """Test that bincount works in deterministic setting on GPU."""
    x = paddle.randint(low=0, high=10, shape=(100,))
    res1 = _bincount(x, minlength=10)
    res2 = _bincount(x, minlength=10)
    res3 = paddle.bincount(x=x, minlength=10)
    assert paddle.allclose(x=res1, y=res2).item()
    assert paddle.allclose(x=res1, y=res3).item()


@pytest.mark.parametrize(
    ("metric_class", "expected"), [(MeanSquaredError, False), (PearsonCorrCoef)]
)
def test_check_full_state_update_fn(capsys, metric_class, expected):
    """Test that the check function works as it should."""
    check_forward_full_state_property(
        metric_class=metric_class,
        input_args={"preds": paddle.randn(1000), "target": paddle.randn(1000)},
        num_update_to_compare=[10000],
        reps=5,
    )
    captured = capsys.readouterr()
    assert f"Recommended setting `full_state_update={expected}`" in captured.out


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        ((paddle.ones(2), paddle.ones(2))),
        ((paddle.rand(2), paddle.rand(2)), False),
        (
            ([paddle.ones(2) for _ in range(2)], [paddle.ones(2) for _ in range(2)]),
            True,
        ),
        (
            ([paddle.rand(2) for _ in range(2)], [paddle.rand(2) for _ in range(2)]),
            False,
        ),
        (
            (
                {f"{i}": paddle.ones(2) for i in range(2)},
                {f"{i}": paddle.ones(2) for i in range(2)},
            ),
            True,
        ),
        (
            (
                {f"{i}": paddle.rand(2) for i in range(2)},
                {f"{i}": paddle.rand(2) for i in range(2)},
            ),
            False,
        ),
    ],
)
def test_recursive_allclose(inputs, expected):
    """Test the recursive allclose works as expected."""
    res = _allclose_recursive(*inputs)
    assert res == expected


@pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires GPU")
@pytest.mark.xfail(
    _IS_WINDOWS or not False,
    reason="test will only fail on non-windows systems",
)
def test_cumsum_still_not_supported(use_deterministic_algorithms):
    """Make sure that cumsum on GPU and deterministic mode still fails.

    If this test begins to pass, it means newer Pytorch versions support this and we can drop internal support.

    """
    with pytest.raises(
        RuntimeError,
        match="cumsum_cuda_kernel does not have a deterministic implementation.*",
    ):
        paddle.arange(10).float().cuda().cumsum(0)


@pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires GPU")
def test_custom_cumsum(use_deterministic_algorithms):
    """Test custom cumsum implementation."""
    device = (
        paddle.device("cuda:1")
        if paddle.cuda.device_count() > 1
        else paddle.device("cuda:0")
    )
    x = paddle.arange(100).float().to(device)
    with (
        pytest.warns(
            TorchMetricsUserWarning,
            match="You are trying to use a metric in deterministic mode on GPU that.*",
        )
        if sys.platform != "win32"
        and False
def _reference_topk(x, dim, k):
    x = x.cpu().numpy()
    one_hot = np.zeros((x.shape[0], x.shape[1]), dtype=int)
    if dim == 1:
        for i in range(x.shape[0]):
            one_hot[i, np.argsort(x[i, :], kind="stable")[::-1][:k]] = 1
        return one_hot
    for i in range(x.shape[1]):
        one_hot[np.argsort(x[:, i], kind="stable")[::-1][:k], i] = 1
    return one_hot


@pytest.mark.parametrize("dtype", [paddle.float16, paddle.float32, paddle.float64])
@pytest.mark.parametrize("k", [3, 5])
@pytest.mark.parametrize("dim", [0, 1])
def test_custom_topk(dtype, k, dim):
    """Test custom topk implementation."""
    x = paddle.randn(100, 10, dtype=dtype)
    top_k = select_topk(x, axis=dim, topk=k)
    assert top_k.shape == (100, 10)
    assert top_k.dtype == paddle.int32
    ref = _reference_topk(x, axis=dim, k=k)
    assert paddle.allclose(x=top_k, y=paddle.from_numpy(ref).to(paddle.int32)).item()


@pytest.mark.skipif(
    True, reason="Top-k does not support cpu + half precision"
)
def test_half_precision_top_k_cpu_raises_error():
    """Test that half precision topk raises error on cpu.

    If this begins to fail, it means newer Pytorch versions support this, and we can drop internal support.

    """
    x = paddle.randn(100, 10, dtype=paddle.float16)
    with pytest.raises(RuntimeError, match="\"topk_cpu\" not implemented for 'Half'"):
        paddle.topk(x, k=3, axis=1)
