from functools import partial

import numpy as np
import paddle
import pytest
from scipy.special import expit as sigmoid
from sklearn.metrics import coverage_error as sk_coverage_error
from sklearn.metrics import \
    label_ranking_average_precision_score as sk_label_ranking
from sklearn.metrics import label_ranking_loss as sk_label_ranking_loss
from unittests import NUM_CLASSES
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester, inject_ignore_index
from unittests.classification._inputs import _multilabel_cases

from paddlemetrics.classification.ranking import (
    MultilabelCoverageError, MultilabelRankingAveragePrecision,
    MultilabelRankingLoss)
from paddlemetrics.functional.classification.ranking import (
    multilabel_coverage_error, multilabel_ranking_average_precision,
    multilabel_ranking_loss)

seed_all(42)


def _reference_sklearn_ranking(preds, target, fn, ignore_index):
    preds = preds.numpy()
    target = target.numpy()
    if (
        np.issubdtype(preds.dtype, np.floating)
        and not ((preds > 0) & (preds < 1)).all()
    ):
        preds = sigmoid(preds)
    preds = np.moveaxis(preds, 1, -1).reshape((-1, preds.shape[1]))
    target = np.moveaxis(target, 1, -1).reshape((-1, target.shape[1]))
    if ignore_index is not None:
        idx = target == ignore_index
        target[idx] = -1
    return fn(target, preds)


@pytest.mark.parametrize(
    ("metric", "functional_metric", "ref_metric"),
    [
        (MultilabelCoverageError, multilabel_coverage_error, sk_coverage_error),
        (
            MultilabelRankingAveragePrecision,
            multilabel_ranking_average_precision,
            sk_label_ranking,
        ),
        (MultilabelRankingLoss, multilabel_ranking_loss, sk_label_ranking_loss),
    ],
)
@pytest.mark.parametrize(
    "inputs",
    [
        _multilabel_cases[1],
        _multilabel_cases[2],
        _multilabel_cases[4],
        _multilabel_cases[5],
    ],
)
class TestMultilabelRanking(MetricTester):
    """Test class for `MultilabelRanking` metric."""

    @pytest.mark.parametrize("ignore_index", [None])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_multilabel_ranking(
        self, inputs, metric, functional_metric, ref_metric, ddp, ignore_index
    ):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=metric,
            reference_metric=partial(
                _reference_sklearn_ranking, fn=ref_metric, ignore_index=ignore_index
            ),
            metric_args={"num_labels": NUM_CLASSES, "ignore_index": ignore_index},
        )

    @pytest.mark.parametrize("ignore_index", [None])
    def test_multilabel_ranking_functional(
        self, inputs, metric, functional_metric, ref_metric, ignore_index
    ):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=functional_metric,
            reference_metric=partial(
                _reference_sklearn_ranking, fn=ref_metric, ignore_index=ignore_index
            ),
            metric_args={"num_labels": NUM_CLASSES, "ignore_index": ignore_index},
        )

    def test_multilabel_ranking_differentiability(
        self, inputs, metric, functional_metric, ref_metric
    ):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=metric,
            metric_functional=functional_metric,
            metric_args={"num_labels": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_multilabel_ranking_dtype_cpu(
        self, inputs, metric, functional_metric, ref_metric, dtype
    ):
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
        if (
            dtype == paddle.float16
            and functional_metric == multilabel_ranking_average_precision
        ):
            pytest.xfail(
                reason="multilabel_ranking_average_precision requires paddle.unique which is not implemented for half"
            )
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=metric,
            metric_functional=functional_metric,
            metric_args={"num_labels": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_multilabel_ranking_dtype_gpu(
        self, inputs, metric, functional_metric, ref_metric, dtype
    ):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=metric,
            metric_functional=functional_metric,
            metric_args={"num_labels": NUM_CLASSES},
            dtype=dtype,
        )
