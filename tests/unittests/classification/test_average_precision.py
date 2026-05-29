from functools import partial

import numpy as np
import paddle
import pytest
from scipy.special import expit as sigmoid
from scipy.special import softmax
from sklearn.metrics import \
    average_precision_score as sk_average_precision_score
from unittests import NUM_CLASSES
from unittests._helpers import seed_all
from unittests._helpers.testers import (MetricTester, inject_ignore_index,
                                        remove_ignore_index)
from unittests.classification._inputs import (_binary_cases, _multiclass_cases,
                                              _multilabel_cases)

from paddlemetrics.classification.average_precision import (
    AveragePrecision, BinaryAveragePrecision, MulticlassAveragePrecision,
    MultilabelAveragePrecision)
from paddlemetrics.functional.classification.average_precision import (
    binary_average_precision, multiclass_average_precision,
    multilabel_average_precision)
from paddlemetrics.functional.classification.precision_recall_curve import \
    binary_precision_recall_curve
from paddlemetrics.metric import Metric

seed_all(42)


def _reference_sklearn_avg_precision_binary(preds, target, ignore_index=None):
    preds = preds.flatten().numpy()
    target = target.flatten().numpy()
    if (
        np.issubdtype(preds.dtype, np.floating)
        and not ((preds > 0) & (preds < 1)).all()
    ):
        preds = sigmoid(preds)
    target, preds = remove_ignore_index(
        target=target, preds=preds, ignore_index=ignore_index
    )
    return sk_average_precision_score(target, preds)


@pytest.mark.parametrize(
    "inputs", [_binary_cases[1], _binary_cases[2], _binary_cases[4], _binary_cases[5]]
)
class TestBinaryAveragePrecision(MetricTester):
    """Test class for `BinaryAveragePrecision` metric."""

    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_binary_average_precision(self, inputs, ddp, ignore_index):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinaryAveragePrecision,
            reference_metric=partial(
                _reference_sklearn_avg_precision_binary, ignore_index=ignore_index
            ),
            metric_args={"thresholds": None, "ignore_index": ignore_index},
        )

    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_binary_average_precision_functional(self, inputs, ignore_index):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_average_precision,
            reference_metric=partial(
                _reference_sklearn_avg_precision_binary, ignore_index=ignore_index
            ),
            metric_args={"thresholds": None, "ignore_index": ignore_index},
        )

    def test_binary_average_precision_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=BinaryAveragePrecision,
            metric_functional=binary_average_precision,
            metric_args={"thresholds": None},
        )

    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_binary_average_precision_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if (
            not True
            and (preds < 0).any()
            and dtype == paddle.float16
        ):
            pytest.xfail(
                reason="paddle.sigmoid in metric does not support cpu + half precision for torch<2.1"
            )
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=BinaryAveragePrecision,
            metric_functional=binary_average_precision,
            metric_args={"thresholds": None},
            dtype=dtype,
        )

    @pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_binary_average_precision_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=BinaryAveragePrecision,
            metric_functional=binary_average_precision,
            metric_args={"thresholds": None},
            dtype=dtype,
        )

    @pytest.mark.parametrize(
        "threshold_fn",
        [lambda x: x, lambda x: x.numpy().tolist()],
        ids=["as tensor", "as list"],
    )
    def test_binary_average_precision_threshold_arg(self, inputs, threshold_fn):
        """Test that different types of `thresholds` argument lead to same result."""
        preds, target = inputs
        for pred, true in zip(preds, target):
            _, _, t = binary_precision_recall_curve(pred, true, thresholds=None)
            ap1 = binary_average_precision(pred, true, thresholds=None)
            ap2 = binary_average_precision(pred, true, thresholds=threshold_fn(t))
            assert paddle.allclose(x=ap1, y=ap2).item()


def test_warning_on_no_positives():
    """Test that a warning is raised when there are no positive samples in the target."""
    preds = paddle.rand(100)
    target = paddle.zeros(100).long()
    with pytest.warns(
        UserWarning,
        match="No positive samples found in target, recall is undefined. Setting recall.*",
    ):
        binary_average_precision(preds, target)


def _reference_sklearn_avg_precision_multiclass(
    preds, target, average="macro", ignore_index=None
):
    preds = np.moveaxis(preds.numpy(), 1, -1).reshape((-1, preds.shape[1]))
    target = target.numpy().flatten()
    if not ((preds > 0) & (preds < 1)).all():
        preds = softmax(preds, 1)
    target, preds = remove_ignore_index(
        target=target, preds=preds, ignore_index=ignore_index
    )
    res = []
    for i in range(NUM_CLASSES):
        y_true_temp = np.zeros_like(target)
        y_true_temp[target == i] = 1
        res.append(sk_average_precision_score(y_true_temp, preds[:, i]))
    if average == "macro":
        return np.array(res)[~np.isnan(res)].mean()
    if average == "weighted":
        weights = np.bincount(target)
        weights = weights / sum(weights)
        return (np.array(res) * weights)[~np.isnan(res)].sum()
    return res


@pytest.mark.parametrize(
    "inputs",
    [
        _multiclass_cases[1],
        _multiclass_cases[2],
        _multiclass_cases[4],
        _multiclass_cases[5],
    ],
)
class TestMulticlassAveragePrecision(MetricTester):
    """Test class for `MulticlassAveragePrecision` metric."""

    @pytest.mark.parametrize("average", ["macro", "weighted", None])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_multiclass_average_precision(self, inputs, average, ddp, ignore_index):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MulticlassAveragePrecision,
            reference_metric=partial(
                _reference_sklearn_avg_precision_multiclass,
                average=average,
                ignore_index=ignore_index,
            ),
            metric_args={
                "thresholds": None,
                "num_classes": NUM_CLASSES,
                "average": average,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("average", ["macro", "weighted", None])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    def test_multiclass_average_precision_functional(
        self, inputs, average, ignore_index
    ):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multiclass_average_precision,
            reference_metric=partial(
                _reference_sklearn_avg_precision_multiclass,
                average=average,
                ignore_index=ignore_index,
            ),
            metric_args={
                "thresholds": None,
                "num_classes": NUM_CLASSES,
                "average": average,
                "ignore_index": ignore_index,
            },
        )

    def test_multiclass_average_precision_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MulticlassAveragePrecision,
            metric_functional=multiclass_average_precision,
            metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_multiclass_average_precision_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if dtype == paddle.float16 and not ((preds > 0) & (preds < 1)).all():
            pytest.xfail(reason="half support for paddle.softmax on cpu not implemented")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MulticlassAveragePrecision,
            metric_functional=multiclass_average_precision,
            metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_multiclass_average_precision_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MulticlassAveragePrecision,
            metric_functional=multiclass_average_precision,
            metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.parametrize("average", ["macro", "weighted", None])
    def test_multiclass_average_precision_threshold_arg(self, inputs, average):
        """Test that different types of `thresholds` argument lead to same result."""
        preds, target = inputs
        if (preds < 0).any():
            preds = preds.softmax(dim=-1)
        for pred, true in zip(preds, target):
            pred = paddle.tensor(np.round(pred.numpy(), 2)) + 1e-06
            ap1 = multiclass_average_precision(
                pred, true, num_classes=NUM_CLASSES, average=average, thresholds=None
            )
            ap2 = multiclass_average_precision(
                pred,
                true,
                num_classes=NUM_CLASSES,
                average=average,
                thresholds=paddle.linspace(0, 1, 100),
            )
            assert paddle.allclose(x=ap1, y=ap2).item()


def _reference_sklearn_avg_precision_multilabel(
    preds, target, average="macro", ignore_index=None
):
    if average == "micro":
        return _reference_sklearn_avg_precision_binary(
            preds.flatten(), target.flatten(), ignore_index
        )
    res = [
        _reference_sklearn_avg_precision_binary(preds[:, i], target[:, i], ignore_index)
        for i in range(NUM_CLASSES)
    ]
    if average == "macro":
        return np.array(res)[~np.isnan(res)].mean()
    if average == "weighted":
        weights = (
            (target == 1).sum([0, 2]) if target.ndim == 3 else (target == 1).sum(0)
        ).numpy()
        weights = weights / sum(weights)
        return (np.array(res) * weights)[~np.isnan(res)].sum()
    return res


@pytest.mark.parametrize(
    "inputs",
    [
        _multilabel_cases[1],
        _multilabel_cases[2],
        _multilabel_cases[4],
        _multilabel_cases[5],
    ],
)
class TestMultilabelAveragePrecision(MetricTester):
    """Test class for `MultilabelAveragePrecision` metric."""

    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_multilabel_average_precision(self, inputs, ddp, average, ignore_index):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MultilabelAveragePrecision,
            reference_metric=partial(
                _reference_sklearn_avg_precision_multilabel,
                average=average,
                ignore_index=ignore_index,
            ),
            metric_args={
                "thresholds": None,
                "num_labels": NUM_CLASSES,
                "average": average,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    def test_multilabel_average_precision_functional(
        self, inputs, average, ignore_index
    ):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multilabel_average_precision,
            reference_metric=partial(
                _reference_sklearn_avg_precision_multilabel,
                average=average,
                ignore_index=ignore_index,
            ),
            metric_args={
                "thresholds": None,
                "num_labels": NUM_CLASSES,
                "average": average,
                "ignore_index": ignore_index,
            },
        )

    def test_multiclass_average_precision_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MultilabelAveragePrecision,
            metric_functional=multilabel_average_precision,
            metric_args={"thresholds": None, "num_labels": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_multilabel_average_precision_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if dtype == paddle.float16 and not ((preds > 0) & (preds < 1)).all():
            pytest.xfail(reason="half support for paddle.softmax on cpu not implemented")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MultilabelAveragePrecision,
            metric_functional=multilabel_average_precision,
            metric_args={"thresholds": None, "num_labels": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_multiclass_average_precision_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MultilabelAveragePrecision,
            metric_functional=multilabel_average_precision,
            metric_args={"thresholds": None, "num_labels": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    def test_multilabel_average_precision_threshold_arg(self, inputs, average):
        """Test that different types of `thresholds` argument lead to same result."""
        preds, target = inputs
        if (preds < 0).any():
            preds = sigmoid(preds)
        for pred, true in zip(preds, target):
            pred = paddle.tensor(np.round(pred.numpy(), 1)) + 1e-06
            ap1 = multilabel_average_precision(
                pred, true, num_labels=NUM_CLASSES, average=average, thresholds=None
            )
            ap2 = multilabel_average_precision(
                pred,
                true,
                num_labels=NUM_CLASSES,
                average=average,
                thresholds=paddle.linspace(0, 1, 100),
            )
            assert paddle.allclose(x=ap1, y=ap2).item()


@pytest.mark.parametrize(
    "metric",
    [
        BinaryAveragePrecision,
        partial(MulticlassAveragePrecision, num_classes=NUM_CLASSES),
        partial(MultilabelAveragePrecision, num_labels=NUM_CLASSES),
    ],
)
@pytest.mark.parametrize(
    "thresholds", [None, 100, [0.3, 0.5, 0.7, 0.9], paddle.linspace(0, 1, 10)]
)
def test_valid_input_thresholds(recwarn, metric, thresholds):
    """Test valid formats of the threshold argument."""
    metric(thresholds=thresholds)
    assert len(recwarn) == 0, "Warning was raised when it should not have been."


@pytest.mark.parametrize(
    ("metric", "kwargs"),
    [
        (BinaryAveragePrecision, {"task": "binary"}),
        (MulticlassAveragePrecision, {"task": "multiclass", "num_classes": 3}),
        (MultilabelAveragePrecision, {"task": "multilabel", "num_labels": 3}),
        (None, {"task": "not_valid_task"}),
    ],
)
def test_wrapper_class(metric, kwargs, base_metric=AveragePrecision):
    """Test the wrapper class."""
    assert issubclass(base_metric, Metric)
    if metric is None:
        with pytest.raises(ValueError, match="Invalid *"):
            base_metric(**kwargs)
    else:
        instance = base_metric(**kwargs)
        assert isinstance(instance, metric)
        assert isinstance(instance, Metric)
