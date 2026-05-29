from functools import partial

import numpy as np
import paddle
import pytest
from scipy.special import expit as sigmoid
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import jaccard_score as sk_jaccard_index
from unittests import NUM_CLASSES, THRESHOLD
from unittests._helpers.testers import (MetricTester, inject_ignore_index,
                                        remove_ignore_index)
from unittests.classification._inputs import (_binary_cases, _multiclass_cases,
                                              _multilabel_cases)

from paddlemetrics.classification.jaccard import (BinaryJaccardIndex,
                                                 JaccardIndex,
                                                 MulticlassJaccardIndex,
                                                 MultilabelJaccardIndex)
from paddlemetrics.functional.classification.jaccard import (
    _jaccard_index_reduce, binary_jaccard_index, multiclass_jaccard_index,
    multilabel_jaccard_index)
from paddlemetrics.metric import Metric


def _reference_sklearn_jaccard_index_binary(
    preds, target, ignore_index=None, zero_division=0
):
    preds = preds.view(-1).numpy()
    target = target.view(-1).numpy()
    if np.issubdtype(preds.dtype, np.floating):
        if not ((preds > 0) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= THRESHOLD).astype(np.uint8)
    target, preds = remove_ignore_index(
        target=target, preds=preds, ignore_index=ignore_index
    )
    return sk_jaccard_index(y_true=target, y_pred=preds, zero_division=zero_division)


@pytest.mark.parametrize("inputs", _binary_cases)
class TestBinaryJaccardIndex(MetricTester):
    """Test class for `BinaryJaccardIndex` metric."""

    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("zero_division", [0, 1])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_binary_jaccard_index(self, inputs, ddp, ignore_index, zero_division):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinaryJaccardIndex,
            reference_metric=partial(
                _reference_sklearn_jaccard_index_binary,
                ignore_index=ignore_index,
                zero_division=zero_division,
            ),
            metric_args={
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "zero_division": zero_division,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("zero_division", [0, 1])
    def test_binary_jaccard_index_functional(self, inputs, ignore_index, zero_division):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_jaccard_index,
            reference_metric=partial(
                _reference_sklearn_jaccard_index_binary,
                ignore_index=ignore_index,
                zero_division=zero_division,
            ),
            metric_args={
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "zero_division": zero_division,
            },
        )

    def test_binary_jaccard_index_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=BinaryJaccardIndex,
            metric_functional=binary_jaccard_index,
            metric_args={"threshold": THRESHOLD},
        )

    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_binary_jaccard_index_dtype_cpu(self, inputs, dtype):
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
            metric_module=BinaryJaccardIndex,
            metric_functional=binary_jaccard_index,
            metric_args={"threshold": THRESHOLD},
            dtype=dtype,
        )

    @pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_binary_jaccard_index_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=BinaryJaccardIndex,
            metric_functional=binary_jaccard_index,
            metric_args={"threshold": THRESHOLD},
            dtype=dtype,
        )


def _reference_sklearn_jaccard_index_multiclass(
    preds, target, ignore_index=None, average="macro", zero_division=0
):
    preds = preds.numpy()
    target = target.numpy()
    if np.issubdtype(preds.dtype, np.floating):
        preds = np.argmax(preds, axis=1)
    preds = preds.flatten()
    target = target.flatten()
    target, preds = remove_ignore_index(
        target=target, preds=preds, ignore_index=ignore_index
    )
    if average is None:
        return sk_jaccard_index(
            y_true=target,
            y_pred=preds,
            average=average,
            labels=list(range(NUM_CLASSES)),
            zero_division=zero_division,
        )
    if ignore_index is not None and 0 <= ignore_index < NUM_CLASSES:
        labels = [i for i in range(NUM_CLASSES) if i != ignore_index]
        res = sk_jaccard_index(
            y_true=target,
            y_pred=preds,
            average=average,
            labels=labels,
            zero_division=zero_division,
        )
        return np.insert(res, ignore_index, 0) if average is None else res
    return sk_jaccard_index(
        y_true=target, y_pred=preds, average=average, zero_division=zero_division
    )


@pytest.mark.parametrize("inputs", _multiclass_cases)
class TestMulticlassJaccardIndex(MetricTester):
    """Test class for `MulticlassJaccardIndex` metric."""

    @pytest.mark.parametrize("average", ["macro", "micro", "weighted", None])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("zero_division", [0, 1])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_multiclass_jaccard_index(
        self, inputs, ddp, ignore_index, average, zero_division
    ):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MulticlassJaccardIndex,
            reference_metric=partial(
                _reference_sklearn_jaccard_index_multiclass,
                ignore_index=ignore_index,
                average=average,
                zero_division=zero_division,
            ),
            metric_args={
                "num_classes": NUM_CLASSES,
                "ignore_index": ignore_index,
                "average": average,
                "zero_division": zero_division,
            },
        )

    @pytest.mark.parametrize("average", ["macro", "micro", "weighted", None])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("zero_division", [0, 1])
    def test_multiclass_jaccard_index_functional(
        self, inputs, ignore_index, average, zero_division
    ):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multiclass_jaccard_index,
            reference_metric=partial(
                _reference_sklearn_jaccard_index_multiclass,
                ignore_index=ignore_index,
                average=average,
                zero_division=zero_division,
            ),
            metric_args={
                "num_classes": NUM_CLASSES,
                "ignore_index": ignore_index,
                "average": average,
                "zero_division": zero_division,
            },
        )

    def test_multiclass_jaccard_index_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MulticlassJaccardIndex,
            metric_functional=multiclass_jaccard_index,
            metric_args={"num_classes": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_multiclass_jaccard_index_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MulticlassJaccardIndex,
            metric_functional=multiclass_jaccard_index,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_multiclass_jaccard_index_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MulticlassJaccardIndex,
            metric_functional=multiclass_jaccard_index,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )


def _reference_sklearn_jaccard_index_multilabel(
    preds, target, ignore_index=None, average="macro", zero_division=0
):
    preds = preds.numpy()
    target = target.numpy()
    if np.issubdtype(preds.dtype, np.floating):
        if not ((preds > 0) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= THRESHOLD).astype(np.uint8)
    preds = np.moveaxis(preds, 1, -1).reshape((-1, preds.shape[1]))
    target = np.moveaxis(target, 1, -1).reshape((-1, target.shape[1]))
    if ignore_index is None:
        return sk_jaccard_index(
            y_true=target, y_pred=preds, average=average, zero_division=zero_division
        )
    if average == "micro":
        return _reference_sklearn_jaccard_index_binary(
            paddle.tensor(preds),
            paddle.tensor(target),
            ignore_index,
            zero_division=zero_division,
        )
    scores, weights = [], []
    for i in range(preds.shape[1]):
        pred, true = preds[:, i], target[:, i]
        true, pred = remove_ignore_index(
            target=true, preds=pred, ignore_index=ignore_index
        )
        confmat = sk_confusion_matrix(true, pred, labels=[0, 1])
        scores.append(sk_jaccard_index(true, pred, zero_division=zero_division))
        weights.append(confmat[1, 0] + confmat[1, 1])
    scores = np.stack(scores, axis=0)
    weights = np.stack(weights, axis=0)
    if average is None or average == "none":
        return scores
    if average == "macro":
        return scores.mean()
    return (scores * weights / weights.sum()).sum()


@pytest.mark.parametrize("inputs", _multilabel_cases)
class TestMultilabelJaccardIndex(MetricTester):
    """Test class for `MultilabelJaccardIndex` metric."""

    @pytest.mark.parametrize("average", ["macro", "micro", "weighted", None])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("zero_division", [0, 1])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_multilabel_jaccard_index(
        self, inputs, ddp, ignore_index, average, zero_division
    ):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MultilabelJaccardIndex,
            reference_metric=partial(
                _reference_sklearn_jaccard_index_multilabel,
                ignore_index=ignore_index,
                average=average,
                zero_division=zero_division,
            ),
            metric_args={
                "num_labels": NUM_CLASSES,
                "ignore_index": ignore_index,
                "average": average,
                "zero_division": zero_division,
            },
        )

    @pytest.mark.parametrize("average", ["macro", "micro", "weighted", None])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("zero_division", [0, 1])
    def test_multilabel_jaccard_index_functional(
        self, inputs, ignore_index, average, zero_division
    ):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multilabel_jaccard_index,
            reference_metric=partial(
                _reference_sklearn_jaccard_index_multilabel,
                ignore_index=ignore_index,
                average=average,
                zero_division=zero_division,
            ),
            metric_args={
                "num_labels": NUM_CLASSES,
                "ignore_index": ignore_index,
                "average": average,
                "zero_division": zero_division,
            },
        )

    def test_multilabel_jaccard_index_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MultilabelJaccardIndex,
            metric_functional=multilabel_jaccard_index,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
        )

    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_multilabel_jaccard_index_dtype_cpu(self, inputs, dtype):
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
            metric_module=MultilabelJaccardIndex,
            metric_functional=multilabel_jaccard_index,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
            dtype=dtype,
        )

    @pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_multilabel_jaccard_index_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MultilabelJaccardIndex,
            metric_functional=multilabel_jaccard_index,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
            dtype=dtype,
        )


def test_corner_case():
    """Issue: https://github.com/Lightning-AI/paddlemetrics/issues/1693."""
    target = paddle.tensor([0, 1, 0, 0])
    preds = paddle.tensor([0, 1, 0, 1])
    metric = MulticlassJaccardIndex(num_classes=3, average="none")
    res = metric(preds, target)
    assert paddle.allclose(x=res, y=paddle.tensor([2.0 / 3.0, 0.5, 0.0])).item()
    metric = MulticlassJaccardIndex(num_classes=3, average="macro")
    res = metric(preds, target)
    assert paddle.allclose(x=res, y=paddle.tensor(0.5833333)).item()
    target = paddle.tensor([0, 1])
    pred = paddle.tensor([0, 1])
    out = paddle.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0]).float()
    res = multiclass_jaccard_index(pred, target, num_classes=10)
    assert paddle.allclose(x=res, y=paddle.ones_like(res)).item()
    res = multiclass_jaccard_index(pred, target, num_classes=10, average="none")
    assert paddle.allclose(x=res, y=out).item()


def test_jaccard_index_zero_division():
    """Issue: https://github.com/Lightning-AI/paddlemetrics/issues/2658."""
    confmat = paddle.tensor([[4, 0], [0, 0]])
    result = _jaccard_index_reduce(confmat, average="binary", zero_division=0.0)
    assert result == 0.0, f"Expected 0.0, but got {result}"
    result = _jaccard_index_reduce(confmat, average="binary", zero_division=1.0)
    assert result == 1.0, f"Expected 1.0, but got {result}"
    confmat = paddle.tensor([[2, 1], [1, 1]])
    result = _jaccard_index_reduce(confmat, average="binary", zero_division=0.0)
    expected = 1 / 3
    assert paddle.isclose(
        result, paddle.tensor(expected)
    ), f"Expected {expected}, but got {result}"


@pytest.mark.parametrize(
    ("metric", "kwargs"),
    [
        (BinaryJaccardIndex, {"task": "binary"}),
        (MulticlassJaccardIndex, {"task": "multiclass", "num_classes": 3}),
        (MultilabelJaccardIndex, {"task": "multilabel", "num_labels": 3}),
        (None, {"task": "not_valid_task"}),
    ],
)
def test_wrapper_class(metric, kwargs, base_metric=JaccardIndex):
    """Test the wrapper class."""
    assert issubclass(base_metric, Metric)
    if metric is None:
        with pytest.raises(ValueError, match="Invalid *"):
            base_metric(**kwargs)
    else:
        instance = base_metric(**kwargs)
        assert isinstance(instance, metric)
        assert isinstance(instance, Metric)
