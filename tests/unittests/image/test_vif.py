from functools import partial

import numpy as np
import paddle
import pytest
from sewar.full_ref import vifp
from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

from paddlemetrics.functional.image.vif import visual_information_fidelity
from paddlemetrics.image.vif import VisualInformationFidelity

seed_all(42)
_inputs = [
    _Input(
        preds=paddle.randint(
            low=0,
            high=255,
            shape=(NUM_BATCHES, BATCH_SIZE, channels, 41, 41),
            dtype=paddle.float32,
        ),
        target=paddle.randint(
            low=0,
            high=255,
            shape=(NUM_BATCHES, BATCH_SIZE, channels, 41, 41),
            dtype=paddle.float32,
        ),
    )
    for channels in [1, 3]
]


def _reference_sewar_vif(preds, target, sigma_nsq=2, reduction="mean"):
    preds = paddle.moveaxis(x=preds, source=1, destination=-1)
    target = paddle.moveaxis(x=target, source=1, destination=-1)
    preds = preds.cpu().numpy()
    target = target.cpu().numpy()
    vif = [
        vifp(GT=target[batch], P=preds[batch], sigma_nsq=sigma_nsq)
        for batch in range(preds.shape[0])
    ]
    if reduction == "none":
        return np.array(vif)
    return np.mean(vif)


@pytest.mark.parametrize(
    ("preds", "target"), [(inputs.preds, inputs.target) for inputs in _inputs]
)
class TestVIF(MetricTester):
    """Test class for `VisualInformationFidelity` metric."""

    atol = 1e-06

    @pytest.mark.parametrize("reduction", ["mean", "none"])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_vif(self, preds, target, reduction, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            metric_class=VisualInformationFidelity,
            reference_metric=partial(_reference_sewar_vif, reduction=reduction),
            metric_args={"reduction": reduction},
            check_ddp_sorting=True,
        )

    @pytest.mark.parametrize("reduction", ["mean", "none"])
    def test_vif_functional(self, preds, target, reduction):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=visual_information_fidelity,
            reference_metric=partial(_reference_sewar_vif, reduction=reduction),
            metric_args={"reduction": reduction},
        )
