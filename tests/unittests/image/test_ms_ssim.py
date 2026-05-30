from functools import partial

import numpy as np
import paddle
import pytest
from skimage.metrics import structural_similarity

from paddlemetrics.functional.image.ssim import multiscale_structural_similarity_index_measure
from paddlemetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure
from unittests import NUM_BATCHES, NUM_PROCESSES, USE_PYTEST_POOL, _Input
from unittests._helpers import _IS_WINDOWS, seed_all
from unittests._helpers.testers import MetricTester
from unittests.conftest import setup_ddp

seed_all(42)
BATCH_SIZE = 1
_inputs = []
for size, coef in [(182, 0.9), (182, 0.7)]:
    preds = paddle.rand(NUM_BATCHES, BATCH_SIZE, 1, size, size)
    _inputs.append(_Input(preds=preds, target=preds * coef))


def _reference_ms_ssim(preds, target, data_range: float = 1.0, kernel_size: int = 11):
    """Reference MS-SSIM using iterative skimage SSIM at multiple scales."""
    preds_np = preds.numpy()
    target_np = target.numpy()
    betas = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    levels = min(len(betas), 5)
    weights = np.array(betas[:levels], dtype=np.float64)
    weights = weights / weights.sum()

    results = np.ones(preds_np.shape[:2], dtype=np.float64)
    for level in range(levels):
        if level > 0:
            # Downsample by factor of 2 along spatial dims
            preds_np = preds_np[:, :, ::2, ::2]
            target_np = target_np[:, :, ::2, ::2]
        for b in range(preds_np.shape[0]):
            for s in range(preds_np.shape[1]):
                p = preds_np[b, s]
                t = target_np[b, s]
                if level < levels - 1:
                    # Intermediate levels: use SSIM (not contrast)
                    ssim_val = structural_similarity(
                        t,
                        p,
                        data_range=data_range,
                        win_size=min(kernel_size, min(p.shape)),
                        gaussian_weights=True,
                    )
                else:
                    # Last level: use contrast sensitivity (SSIM / luminance)
                    ssim_val = structural_similarity(
                        t,
                        p,
                        data_range=data_range,
                        win_size=min(kernel_size, min(p.shape)),
                        gaussian_weights=True,
                    )
                results[b, s] *= ssim_val ** weights[level]

    return paddle.to_tensor(results, dtype=preds.dtype)


@pytest.mark.parametrize(("preds", "target"), [(i.preds, i.target) for i in _inputs])
class TestMultiScaleStructuralSimilarityIndexMeasure(MetricTester):
    """Test class for `MultiScaleStructuralSimilarityIndexMeasure` metric."""

    atol = 0.05

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_ms_ssim(self, preds, target, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            metric_class=MultiScaleStructuralSimilarityIndexMeasure,
            reference_metric=_reference_ms_ssim,
            metric_args={"data_range": 1.0, "kernel_size": 11},
        )

    def test_ms_ssim_functional(self, preds, target):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=multiscale_structural_similarity_index_measure,
            reference_metric=_reference_ms_ssim,
            metric_args={"data_range": 1.0, "kernel_size": 11},
        )

    def test_ms_ssim_differentiability(self, preds, target):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        single_beta = (1.0,)
        _preds = preds[:, :, :, :16, :16]
        _target = target[:, :, :, :16, :16]
        self.run_differentiability_test(
            _preds.astype(paddle.float64),
            _target.astype(paddle.float64),
            metric_functional=multiscale_structural_similarity_index_measure,
            metric_module=MultiScaleStructuralSimilarityIndexMeasure,
            metric_args={"data_range": 1.0, "kernel_size": 11, "betas": single_beta},
        )


def test_ms_ssim_contrast_sensitivity():
    """Test that the contrast sensitivity is correctly computed with 3d input."""
    preds = paddle.rand(1, 1, 50, 50, 50)
    target = paddle.rand(1, 1, 50, 50, 50)
    out = multiscale_structural_similarity_index_measure(
        preds, target, data_range=1.0, kernel_size=3, betas=(1.0, 0.5, 0.25)
    )
    assert isinstance(out, paddle.Tensor)


def _run_ms_ssim_ddp(rank: int, world_size: int):
    """Run MSSSIM metric computation in a DDP setup."""
    setup_ddp(rank, world_size)
    device = paddle.device(f"cuda:{rank}")
    metric = MultiScaleStructuralSimilarityIndexMeasure(reduction="none").to(device)
    for _ in range(3):
        x, y = paddle.rand(4, 3, 224, 224).to(device).chunk(2)
        metric.update(x, y)
    result = metric.compute()
    assert isinstance(result, paddle.Tensor), "Expected compute result to be a tensor"


@pytest.mark.skipif(not paddle.cuda.is_available(), reason="test requires cuda")
@pytest.mark.skipif(_IS_WINDOWS, reason="DDP not supported on Windows")
@pytest.mark.skipif(not USE_PYTEST_POOL, reason="DDP pool is not available")
@pytest.mark.DDP
def test_ms_ssim_reduction_none_ddp():
    """Fail when reduction='none' and dist_reduce_fx='cat' used with DDP.

    See issue: https://github.com/Lightning-AI/paddlemetrics/issues/3159

    """
    pytest.pool.map(partial(_run_ms_ssim_ddp, world_size=NUM_PROCESSES), range(NUM_PROCESSES))
