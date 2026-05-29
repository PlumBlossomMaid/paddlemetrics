from typing import Union

import pytest
from unittests.text._helpers import TextTester
from unittests.text._inputs import (_inputs_error_rate_batch_size_1,
                                    _inputs_error_rate_batch_size_2)

from paddlemetrics.functional.text.cer import char_error_rate
from paddlemetrics.text.cer import CharErrorRate


def _reference_jiwer_cer(preds: Union[str, list[str]], target: Union[str, list[str]]):
    try:
        from jiwer import cer
    except ImportError:
        pytest.skip("test requires jiwer package to be installed.")
    return cer(target, preds)


@pytest.mark.parametrize(
    ("preds", "targets"),
    [
        (_inputs_error_rate_batch_size_1.preds, _inputs_error_rate_batch_size_1.target),
        (_inputs_error_rate_batch_size_2.preds, _inputs_error_rate_batch_size_2.target),
    ],
)
class TestCharErrorRate(TextTester):
    """Test class for character error rate."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_cer_class(self, ddp, preds, targets):
        """Test modular version of cer."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            targets=targets,
            metric_class=CharErrorRate,
            reference_metric=_reference_jiwer_cer,
        )

    def test_cer_functional(self, preds, targets):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=char_error_rate,
            reference_metric=_reference_jiwer_cer,
        )

    def test_cer_differentiability(self, preds, targets):
        """Test differentiability of cer metric."""
        self.run_differentiability_test(
            preds=preds,
            targets=targets,
            metric_module=CharErrorRate,
            metric_functional=char_error_rate,
        )
