from typing import Union

import pytest
from unittests._helpers import seed_all
from unittests.text._helpers import TextTester
from unittests.text._inputs import (_inputs_error_rate_batch_size_1,
                                    _inputs_error_rate_batch_size_2)

from paddlemetrics.functional.text.mer import match_error_rate
from paddlemetrics.text.mer import MatchErrorRate

seed_all(42)


def _reference_jiwer_mer(preds: Union[str, list[str]], target: Union[str, list[str]]):
    try:
        from jiwer import mer
    except ImportError:
        pytest.skip("test requires jiwer package to be installed")
    return mer(target, preds)


@pytest.mark.parametrize(
    ("preds", "targets"),
    [
        (_inputs_error_rate_batch_size_1.preds, _inputs_error_rate_batch_size_1.target),
        (_inputs_error_rate_batch_size_2.preds, _inputs_error_rate_batch_size_2.target),
    ],
)
class TestMatchErrorRate(TextTester):
    """Test class for `MatchErrorRate` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_mer_class(self, ddp, preds, targets):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            targets=targets,
            metric_class=MatchErrorRate,
            reference_metric=_reference_jiwer_mer,
        )

    def test_mer_functional(self, preds, targets):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=match_error_rate,
            reference_metric=_reference_jiwer_mer,
        )

    def test_mer_differentiability(self, preds, targets):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            preds=preds,
            targets=targets,
            metric_module=MatchErrorRate,
            metric_functional=match_error_rate,
        )
