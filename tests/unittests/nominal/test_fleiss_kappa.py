import sys

from functools import partial

import numpy as np
import paddle
import pytest
from statsmodels.stats.inter_rater import fleiss_kappa as sk_fleiss_kappa
from unittests import BATCH_SIZE, NUM_BATCHES, NUM_CLASSES
from unittests._helpers.testers import MetricTester

from paddlemetrics.functional.nominal.fleiss_kappa import fleiss_kappa
from paddlemetrics.nominal.fleiss_kappa import FleissKappa

NUM_RATERS = 20
NUM_CATEGORIES = NUM_CLASSES


def _reference_fleiss_kappa(preds, target, mode):
    if mode == "probs":
        counts = np.zeros((preds.shape[0], preds.shape[1]))
        preds = preds.argmax(dim=1)
        for participant in range(preds.shape[0]):
            for rater in range(preds.shape[1]):
                counts[participant, preds[participant, rater]] += 1
        return sk_fleiss_kappa(counts)
    return sk_fleiss_kappa(preds)


def wrapped_fleiss_kappa(preds, target, mode):
    """Wrapped function for `fleiss_kappa` to support testing framework."""
    return fleiss_kappa(preds, mode)


class WrappedFleissKappa(FleissKappa):
    """Wrapped class for `FleissKappa` to support testing framework."""

    def update(self, preds, target):
        """Update function."""
        super().update(preds)


def _random_counts(high, size):
    """Generate random counts matrix that is fully ranked.

    Interface is similar to paddle.randint.

    """
    x = paddle.randint(low=0, high=high, shape=size)
    x_sum = x.sum(-1)
    x_total = x_sum._max()
    x[:, :, -1] = x_total - (x_sum - x[:, :, -1])
    return x


@pytest.mark.parametrize(
    ("preds", "target", "mode"),
    [
        (
            _random_counts(
                high=NUM_RATERS, size=(NUM_BATCHES, BATCH_SIZE, NUM_CATEGORIES)
            ),
            _random_counts(
                high=NUM_RATERS, size=(NUM_BATCHES, BATCH_SIZE, NUM_CATEGORIES)
            ),
            "counts",
        ),
        (
            paddle.randn(NUM_BATCHES, BATCH_SIZE, NUM_CATEGORIES, NUM_RATERS),
            paddle.randn(NUM_BATCHES, BATCH_SIZE, NUM_CATEGORIES, NUM_RATERS),
            "probs",
        ),
    ],
)
class TestFleissKappa(MetricTester):
    """Test class for `FleissKappa` metric."""

    atol = 1e-05

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_fleiss_kappa(self, ddp, preds, target, mode):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=WrappedFleissKappa,
            reference_metric=partial(_reference_fleiss_kappa, mode=mode),
            metric_args={"mode": mode},
        )

    def test_fleiss_kappa_functional(self, preds, target, mode):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=wrapped_fleiss_kappa,
            reference_metric=partial(_reference_fleiss_kappa, mode=mode),
            metric_args={"mode": mode},
        )

    def test_fleiss_kappa_differentiability(self, preds, target, mode):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            preds,
            target,
            metric_module=WrappedFleissKappa,
            metric_functional=wrapped_fleiss_kappa,
            metric_args={"mode": mode},
        )
