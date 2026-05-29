from unittest import mock

import paddle
from lightning_utilities import module_available

if module_available("lightning"):
    pass
from integrations.lightning.boring_model import BoringModel

from paddlemetrics import MetricCollection
from paddlemetrics.aggregation import SumMetric
from paddlemetrics.classification import (BinaryAccuracy,
                                         BinaryAveragePrecision,
                                         MulticlassAccuracy)
from paddlemetrics.regression import MeanAbsoluteError, MeanSquaredError
from paddlemetrics.utils.prints import rank_zero_only
from paddlemetrics.wrappers import (ClasswiseWrapper, MinMaxMetric,
                                   MultitaskWrapper)

class DiffMetric(SumMetric):
    """DiffMetric inherited from `SumMetric` by overriding its `update` method."""

    def update(self, value):
        """Update state."""
        super().update(-value)


def test_metric_lightning(tmpdir):
    """Test that including a metric inside a lightning module calculates a simple sum correctly."""

    class TestModel(BoringModel):
        def __init__(self) -> None:
            super().__init__()
            self.metric = SumMetric()
            self.register_buffer("sum", paddle.tensor(0.0))

        def training_step(self, batch, batch_idx):
            x = batch
            self.metric(x.sum())
            self.sum += x.sum()
            return self.step(x)

        def on_training_epoch_end(self):
            if not paddle.allclose(x=self.sum, y=self.metric.compute()).item():
                raise ValueError("Sum and computed value must be equal")
            self.sum = 0.0
            self.metric.reset()

    model = TestModel()
    model.val_dataloader = None
def test_metrics_reset(tmpdir):
    """Tests that metrics are reset correctly after the end of the train/val/test epoch.

    Taken from: `Metric Test for Reset`_

    """

    class TestModel(BoringModel):
        def __init__(self) -> None:
            super().__init__()
            self.layer = paddle.nn.Linear(32, 1)
            for stage in ["train", "val", "test"]:
                acc = BinaryAccuracy()
                acc.reset = mock.Mock(side_effect=acc.reset)
                ap = BinaryAveragePrecision()
                ap.reset = mock.Mock(side_effect=ap.reset)
                self.add_module(f"acc_{stage}", acc)
                self.add_module(f"ap_{stage}", ap)

        def forward(self, x):
            return self.layer(x)

        def _step(self, stage, batch):
            labels = (batch.detach().sum(1) > 0).float()
            logits = self.forward(batch)
            loss = paddle.nn.functional.binary_cross_entropy_with_logits(
                logit=logits, label=labels.unsqueeze(1)
            )
            probs = paddle.sigmoid(logits.detach())
            self.log(f"loss/{stage}", loss)
            acc = self._modules[f"acc_{stage}"]
            ap = self._modules[f"ap_{stage}"]
            labels_int = labels.to(paddle.long)
            acc(probs.flatten(), labels_int)
            ap(probs.flatten(), labels_int)
            acc.reset.reset_mock()
            ap.reset.reset_mock()
            self.log(f"{stage}/accuracy", acc)
            self.log(f"{stage}/ap", ap)
            return loss

        def training_step(self, batch, batch_idx):
            return self._step("train", batch)

        def validation_step(self, batch, batch_idx):
            return self._step("val", batch)

        def test_step(self, batch, batch_idx):
            return self._step("test", batch)

        def _assert_epoch_end(self, stage):
            acc = self._modules[f"acc_{stage}"]
            ap = self._modules[f"ap_{stage}"]
            acc.reset.asset_not_called()
            ap.reset.assert_not_called()

        def on_train_epoch_end(self):
            self._assert_epoch_end("train")

        def on_validation_epoch_end(self):
            self._assert_epoch_end("val")

        def on_test_epoch_end(self):
            self._assert_epoch_end("test")

    def _assert_called(model, stage):
        acc = model._modules[f"acc_{stage}"]
        ap = model._modules[f"ap_{stage}"]
        acc.reset.assert_called_once()
        acc.reset.reset_mock()
        ap.reset.assert_called_once()
        ap.reset.reset_mock()

    model = TestModel()
def test_metric_lightning_log(tmpdir):
    """Test logging a metric object and that the metric state gets reset after each epoch."""

    class TestModel(BoringModel):
        def __init__(self) -> None:
            super().__init__()
            self.metric_update = SumMetric()
            self.metric_update_step = SumMetric()
            self.metric_update_epoch = SumMetric()
            self.metric_forward = SumMetric()
            self.metric_forward_step = SumMetric()
            self.metric_forward_epoch = SumMetric()
            self.compo_update = SumMetric() + SumMetric()
            self.compo_update_step = SumMetric() + SumMetric()
            self.compo_update_epoch = SumMetric() + SumMetric()
            self.compo_forward = SumMetric() + SumMetric()
            self.compo_forward_step = SumMetric() + SumMetric()
            self.compo_forward_epoch = SumMetric() + SumMetric()
            self.sum = []

        def training_step(self, batch, batch_idx):
            x = batch
            s = x.sum()
            for metric in [
                self.metric_update,
                self.metric_update_step,
                self.metric_update_epoch,
            ]:
                metric.update(s)
            for metric in [
                self.metric_forward,
                self.metric_forward_step,
                self.metric_forward_epoch,
            ]:
                _ = metric(s)
            for metric in [
                self.compo_update,
                self.compo_update_step,
                self.compo_update_epoch,
            ]:
                metric.update(s)
            for metric in [
                self.compo_forward,
                self.compo_forward_step,
                self.compo_forward_epoch,
            ]:
                _ = metric(s)
            self.sum.append(s)
            self.log("metric_update", self.metric_update)
            self.log(
                "metric_update_step",
                self.metric_update_step,
                on_epoch=False,
                on_step=True,
            )
            self.log(
                "metric_update_epoch",
                self.metric_update_epoch,
                on_epoch=True,
                on_step=False,
            )
            self.log("metric_forward", self.metric_forward)
            self.log(
                "metric_forward_step",
                self.metric_forward_step,
                on_epoch=False,
                on_step=True,
            )
            self.log(
                "metric_forward_epoch",
                self.metric_forward_epoch,
                on_epoch=True,
                on_step=False,
            )
            self.log("compo_update", self.compo_update)
            self.log(
                "compo_update_step",
                self.compo_update_step,
                on_epoch=False,
                on_step=True,
            )
            self.log(
                "compo_update_epoch",
                self.compo_update_epoch,
                on_epoch=True,
                on_step=False,
            )
            self.log("compo_forward", self.compo_forward)
            self.log(
                "compo_forward_step",
                self.compo_forward_step,
                on_epoch=False,
                on_step=True,
            )
            self.log(
                "compo_forward_epoch",
                self.compo_forward_epoch,
                on_epoch=True,
                on_step=False,
            )
            return self.step(x)

    model = TestModel()

def test_metric_collection_lightning_log(tmpdir):
    """Test that MetricCollection works with Lightning modules."""

    class TestModel(BoringModel):
        def __init__(self) -> None:
            super().__init__()
            self.metric = MetricCollection([SumMetric(), DiffMetric()])
            self.register_buffer("sum", paddle.tensor(0.0))
            self.register_buffer("diff", paddle.tensor(0.0))

        def training_step(self, batch, batch_idx):
            x = batch
            metric_vals = self.metric(x.sum())
            self.sum += x.sum()
            self.diff -= x.sum()
            self.log_dict({f"{k}_step": v for k, v in metric_vals.items()})
            return self.step(x)

        def on_train_epoch_end(self):
            metric_vals = self.metric.compute()
            self.log_dict({f"{k}_epoch": v for k, v in metric_vals.items()})

    model = TestModel()
def test_task_wrapper_lightning_logging(tmpdir):
    """Test that MultiTaskWrapper works with Lightning modules."""

    class TestModel(BoringModel):
        def __init__(self) -> None:
            super().__init__()
            self.multitask = MultitaskWrapper(
                {"classification": BinaryAccuracy(), "regression": MeanSquaredError()}
            )
            self.multitask_collection = MultitaskWrapper(
                {
                    "classification": MetricCollection(
                        [BinaryAccuracy(), BinaryAveragePrecision()]
                    ),
                    "regression": MetricCollection(
                        [MeanSquaredError(), MeanAbsoluteError()]
                    ),
                }
            )
            self.accuracy = BinaryAccuracy()
            self.mse = MeanSquaredError()

        def training_step(self, batch, batch_idx):
            preds = paddle.rand(10)
            target = paddle.rand(10)
            self.multitask(
                {"classification": preds, "regression": preds},
                {"classification": target.round().int(), "regression": target},
            )
            self.multitask_collection(
                {"classification": preds, "regression": preds},
                {"classification": target.round().int(), "regression": target},
            )
            self.accuracy(preds.round(), target.round())
            self.mse(preds, target)
            self.log("accuracy", self.accuracy, on_epoch=True)
            self.log("mse", self.mse, on_epoch=True)
            self.log_dict(self.multitask, on_epoch=True)
            self.log_dict(self.multitask_collection, on_epoch=True)
            return self.step(batch)

    model = TestModel()
def test_scriptable(tmpdir):
    """Test that lightning modules can still be scripted even if metrics cannot."""

    class TestModel(BoringModel):
        def __init__(self) -> None:
            super().__init__()
            self.metric = SumMetric()
            self.register_buffer("sum", paddle.tensor(0.0))

        def training_step(self, batch, batch_idx):
            x = batch
            self.metric(x.sum())
            self.sum += x.sum()
            self.log("sum", self.metric, on_epoch=True, on_step=False)
            return self.step(x)

    model = TestModel()
def test_dtype_in_pl_module_transfer(tmpdir):
    """Test that metric states don't change dtype when .half() or .float() is called on the LightningModule."""

def test_collection_classwise_lightning_integration(tmpdir):
    """Check the integration of ClasswiseWrapper, MetricCollection and LightningModule.

    See issue: https://github.com/Lightning-AI/paddlemetrics/issues/2683

    """

    class TestModel(BoringModel):
        def __init__(self) -> None:
            super().__init__()
            self.train_metrics = MetricCollection(
                {
                    "macro_accuracy": MulticlassAccuracy(
                        num_classes=5, average="macro"
                    ),
                    "classwise_accuracy": ClasswiseWrapper(
                        MulticlassAccuracy(num_classes=5, average=None)
                    ),
                },
                prefix="train_",
            )
            self.val_metrics = self.train_metrics.clone(prefix="val_")

        def training_step(self, batch, batch_idx):
            loss = self(batch).sum()
            preds = paddle.randint(low=0, high=5, shape=(100,))
            target = paddle.randint(low=0, high=5, shape=(100,))
            batch_values = self.train_metrics(preds, target)
            self.log_dict(batch_values, on_step=True, on_epoch=False)
            return {"loss": loss}

        def validation_step(self, batch, batch_idx):
            preds = paddle.randint(low=0, high=5, shape=(100,))
            target = paddle.randint(low=0, high=5, shape=(100,))
            self.val_metrics.update(preds, target)

        def on_validation_epoch_end(self):
            self.log_dict(self.val_metrics.compute(), on_step=False, on_epoch=True)
            self.val_metrics.reset()

    model = TestModel()
def test_collection_minmax_lightning_integration(tmpdir):
    """Check the integration of MinMaxWrapper, MetricCollection and LightningModule.

    See issue: https://github.com/Lightning-AI/paddlemetrics/issues/2763

    """

    class TestModel(BoringModel):
        def __init__(self) -> None:
            super().__init__()
            self.train_metrics = MetricCollection(
                {
                    "macro_accuracy": MinMaxMetric(
                        MulticlassAccuracy(num_classes=5, average="macro")
                    ),
                    "weighted_accuracy": MinMaxMetric(
                        MulticlassAccuracy(num_classes=5, average="weighted")
                    ),
                },
                prefix="train_",
            )
            self.val_metrics = self.train_metrics.clone(prefix="val_")

        def training_step(self, batch, batch_idx):
            loss = self(batch).sum()
            preds = paddle.randint(low=0, high=5, shape=(100,))
            target = paddle.randint(low=0, high=5, shape=(100,))
            batch_values = self.train_metrics(preds, target)
            self.log_dict(batch_values, on_step=True, on_epoch=False)
            return {"loss": loss}

        def validation_step(self, batch, batch_idx):
            preds = paddle.randint(low=0, high=5, shape=(100,))
            target = paddle.randint(low=0, high=5, shape=(100,))
            self.val_metrics.update(preds, target)

        def on_validation_epoch_end(self):
            self.log_dict(self.val_metrics.compute(), on_step=False, on_epoch=True)
            self.val_metrics.reset()

    model = TestModel()