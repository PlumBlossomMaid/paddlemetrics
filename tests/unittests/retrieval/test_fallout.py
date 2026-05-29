from typing import Callable, Optional, Union

import numpy as np
import paddle
import pytest
from typing_extensions import Literal
from unittests._helpers import seed_all
from unittests.retrieval.helpers import (
    RetrievalMetricTester, _concat_tests, _custom_aggregate_fn,
    _default_metric_class_input_arguments,
    _default_metric_class_input_arguments_ignore_index,
    _default_metric_functional_input_arguments,
    _errors_test_class_metric_parameters_default,
    _errors_test_class_metric_parameters_k,
    _errors_test_class_metric_parameters_no_neg_target,
    _errors_test_functional_metric_parameters_default,
    _errors_test_functional_metric_parameters_k)

from paddlemetrics.functional.retrieval.fall_out import retrieval_fall_out
from paddlemetrics.retrieval.fall_out import RetrievalFallOut

seed_all(42)


def _fallout_at_k(target: np.ndarray, preds: np.ndarray, top_k: Optional[int] = None):
    """Didn't find a reliable implementation of Fall-out in Information Retrieval, so, reimplementing here.

    See Wikipedia for `Fall-out`_ for more information about the metric definition.

    """
    assert target.shape == preds.shape
    assert len(target.shape) == 1
    top_k = len(preds) if top_k is None else top_k
    target = 1 - target
    if target.sum():
        order_indexes = np.argsort(preds, axis=0)[::-1]
        relevant = np.sum(target[order_indexes][:top_k])
        return relevant * 1.0 / target.sum()
    return np.NaN


class TestFallOut(RetrievalMetricTester):
    """Test class for `FallOut` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    @pytest.mark.parametrize("empty_target_action", ["skip", "neg", "pos"])
    @pytest.mark.parametrize("ignore_index", [None, 1])
    @pytest.mark.parametrize("k", [None, 1, 10])
    @pytest.mark.parametrize(
        "aggregation", ["mean", "median", "max", "min", _custom_aggregate_fn]
    )
    @pytest.mark.parametrize(**_default_metric_class_input_arguments)
    def test_class_metric(
        self,
        ddp: bool,
        indexes: paddle.Tensor,
        preds: paddle.Tensor,
        target: paddle.Tensor,
        empty_target_action: str,
        ignore_index: int,
        k: int,
        aggregation: Union[Literal["mean", "median", "min", "max"], Callable],
    ):
        """Test class implementation of metric."""
        metric_args = {
            "empty_target_action": empty_target_action,
            "top_k": k,
            "ignore_index": ignore_index,
            "aggregation": aggregation,
        }
        self.run_class_metric_test(
            ddp=ddp,
            indexes=indexes,
            preds=preds,
            target=target,
            metric_class=RetrievalFallOut,
            reference_metric=_fallout_at_k,
            reverse=True,
            metric_args=metric_args,
        )

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    @pytest.mark.parametrize("empty_target_action", ["skip", "neg", "pos"])
    @pytest.mark.parametrize("k", [None, 1, 4, 10])
    @pytest.mark.parametrize(**_default_metric_class_input_arguments_ignore_index)
    def test_class_metric_ignore_index(
        self,
        ddp: bool,
        indexes: paddle.Tensor,
        preds: paddle.Tensor,
        target: paddle.Tensor,
        empty_target_action: str,
        k: int,
    ):
        """Test class implementation of metric with ignore_index argument."""
        metric_args = {
            "empty_target_action": empty_target_action,
            "top_k": k,
            "ignore_index": -100,
        }
        self.run_class_metric_test(
            ddp=ddp,
            indexes=indexes,
            preds=preds,
            target=target,
            metric_class=RetrievalFallOut,
            reference_metric=_fallout_at_k,
            reverse=True,
            metric_args=metric_args,
        )

    @pytest.mark.parametrize(**_default_metric_functional_input_arguments)
    @pytest.mark.parametrize("k", [None, 1, 4, 10])
    def test_functional_metric(
        self, preds: paddle.Tensor, target: paddle.Tensor, k: int
    ):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=retrieval_fall_out,
            reference_metric=_fallout_at_k,
            reverse=True,
            metric_args={},
            top_k=k,
        )

    @pytest.mark.parametrize(**_default_metric_class_input_arguments)
    def test_precision_cpu(
        self, indexes: paddle.Tensor, preds: paddle.Tensor, target: paddle.Tensor
    ):
        """Test dtype support of the metric on CPU."""
        self.run_precision_test_cpu(
            indexes=indexes,
            preds=preds,
            target=target,
            metric_module=RetrievalFallOut,
            metric_functional=retrieval_fall_out,
        )

    @pytest.mark.parametrize(**_default_metric_class_input_arguments)
    def test_precision_gpu(
        self, indexes: paddle.Tensor, preds: paddle.Tensor, target: paddle.Tensor
    ):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            indexes=indexes,
            preds=preds,
            target=target,
            metric_module=RetrievalFallOut,
            metric_functional=retrieval_fall_out,
        )

    @pytest.mark.parametrize(
        **_concat_tests(
            _errors_test_class_metric_parameters_default,
            _errors_test_class_metric_parameters_no_neg_target,
            _errors_test_class_metric_parameters_k,
        )
    )
    def test_arguments_class_metric(
        self,
        indexes: paddle.Tensor,
        preds: paddle.Tensor,
        target: paddle.Tensor,
        message: str,
        metric_args: dict,
    ):
        """Test that specific errors are raised for incorrect input."""
        self.run_metric_class_arguments_test(
            indexes=indexes,
            preds=preds,
            target=target,
            metric_class=RetrievalFallOut,
            message=message,
            metric_args=metric_args,
            exception_type=ValueError,
            kwargs_update={},
        )

    @pytest.mark.parametrize(
        **_concat_tests(
            _errors_test_functional_metric_parameters_default,
            _errors_test_functional_metric_parameters_k,
        )
    )
    def test_arguments_functional_metric(
        self,
        preds: paddle.Tensor,
        target: paddle.Tensor,
        message: str,
        metric_args: dict,
    ):
        """Test that specific errors are raised for incorrect input."""
        self.run_functional_metric_arguments_test(
            preds=preds,
            target=target,
            metric_functional=retrieval_fall_out,
            message=message,
            exception_type=ValueError,
            kwargs_update=metric_args,
        )
