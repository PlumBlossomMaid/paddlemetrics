from functools import partial
from typing import Any

import paddle
import pytest
from unittests import _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

from paddlemetrics.functional.image.tv import total_variation
from paddlemetrics.image.tv import TotalVariation

seed_all(42)


class TotalVariationTester(TotalVariation):
    """Tester class for `TotalVariation` metric overriding its update method."""

    def update(self, img, *args: Any):
        """Update metric."""
        super().update(img=img)


def _total_variaion_wrapped(preds, target, reduction="mean"):
    return total_variation(preds, reduction)


def _reference_kornia_tv(preds, target, reduction):
_inputs = []
for size, channel, dtype in [
    (12, 3, paddle.float32),
    (13, 3, paddle.float32),
    (14, 3, paddle.float64),
    (15, 3, paddle.float64),
]:
    preds = paddle.rand(2, 4, channel, size, size, dtype=dtype)
    target = paddle.rand(2, 4, channel, size, size, dtype=dtype)
    _inputs.append(_Input(preds=preds, target=target))


@pytest.mark.parametrize(("preds", "target"), [(i.preds, i.target) for i in _inputs])
@pytest.mark.parametrize("reduction", ["sum", "mean", None])
class TestTotalVariation(MetricTester):
    """Test class for `TotalVariation` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_total_variation(self, preds, target, reduction, ddp):
        """Test class implementation of metric."""
        if reduction is None and ddp:
            pytest.skip(
                "reduction=None and ddp=True runs out of memory on CI hardware, but it does work"
            )
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            TotalVariationTester,
            partial(_reference_kornia_tv, reduction=reduction),
            metric_args={"reduction": reduction},
        )

    def test_total_variation_functional(self, preds, target, reduction):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            _total_variaion_wrapped,
            partial(_reference_kornia_tv, reduction=reduction),
            metric_args={"reduction": reduction},
        )

    def test_sam_half_cpu(self, preds, target, reduction):
        """Test for half precision on CPU."""
        self.run_precision_test_cpu(
            preds, target, TotalVariationTester, _total_variaion_wrapped
        )

    @pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires cuda")
    def test_sam_half_gpu(self, preds, target, reduction):
        """Test for half precision on GPU."""
        self.run_precision_test_gpu(
            preds, target, TotalVariationTester, _total_variaion_wrapped
        )


def test_correct_args():
    """That that arguments have the right type and sizes."""
    with pytest.raises(ValueError, match="Expected argument `reduction`.*"):
        _ = TotalVariation(reduction="diff")
    with pytest.raises(RuntimeError, match="Expected input `img` to.*"):
        _ = total_variation(paddle.randn(1, 2, 3))
