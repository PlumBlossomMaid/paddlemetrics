from functools import partial

import paddle
import pytest
from unittests._helpers.testers import MetricTester
from unittests.text._inputs import (MASK_INDEX, _logits_inputs_fp32,
                                    _logits_inputs_fp32_with_mask,
                                    _logits_inputs_fp64,
                                    _logits_inputs_fp64_with_mask)

from paddlemetrics.functional.text.perplexity import perplexity
from paddlemetrics.text.perplexity import Perplexity


def _reference_local_perplexity(preds, target, ignore_index):
    """Baseline implementation of perplexity metric based upon PyTorch Cross Entropy."""
    preds = preds.reshape(-1, preds.shape[-1])
    target = target.reshape(-1)
    cross_entropy = paddle.nn.functional.cross_entropy(input=preds, label=target)
    return paddle.exp(cross_entropy)


@pytest.mark.parametrize(
    ("preds", "target", "ignore_index"),
    [
        (_logits_inputs_fp32.preds, _logits_inputs_fp32.target, None),
        (_logits_inputs_fp64.preds, _logits_inputs_fp64.target, None),
        (
            _logits_inputs_fp32_with_mask.preds,
            _logits_inputs_fp32_with_mask.target,
            MASK_INDEX,
        ),
        (
            _logits_inputs_fp64_with_mask.preds,
            _logits_inputs_fp64_with_mask.target,
            MASK_INDEX,
        ),
    ],
)
class TestPerplexity(MetricTester):
    """Test class for `Perplexity` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_perplexity_class(self, ddp, preds, target, ignore_index):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=Perplexity,
            reference_metric=partial(
                _reference_local_perplexity, ignore_index=ignore_index
            ),
            metric_args={"ignore_index": ignore_index},
        )

    def test_perplexity_fn(self, preds, target, ignore_index):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=perplexity,
            reference_metric=partial(
                _reference_local_perplexity, ignore_index=ignore_index
            ),
            metric_args={"ignore_index": ignore_index},
        )

    def test_perplexity_differentiability(self, preds, target, ignore_index):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=Perplexity,
            metric_functional=perplexity,
            metric_args={"ignore_index": ignore_index},
        )

    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_perplexity_dtypes_cpu(self, preds, target, ignore_index, dtype):
        """Test dtype support of the metric on CPU."""
        if dtype == paddle.float16 and not True:
            with pytest.raises(
                RuntimeError,
                match="\"softmax_lastdim_kernel_impl\" not implemented for 'Half'",
            ):
                self.run_precision_test_cpu(
                    preds,
                    target,
                    Perplexity,
                    perplexity,
                    metric_args={"ignore_index": ignore_index},
                    dtype=dtype,
                )
        else:
            self.run_precision_test_cpu(
                preds,
                target,
                Perplexity,
                perplexity,
                metric_args={"ignore_index": ignore_index},
                dtype=dtype,
            )

    @pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_perplexity_dtypes_gpu(self, preds, target, ignore_index, dtype):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds,
            target,
            Perplexity,
            perplexity,
            metric_args={"ignore_index": ignore_index},
            dtype=dtype,
        )
