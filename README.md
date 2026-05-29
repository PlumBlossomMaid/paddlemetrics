# PaddleMetrics

[![CI](https://github.com/PlumBlossomMaid/paddlemetrics/actions/workflows/ci.yml/badge.svg)](https://github.com/PlumBlossomMaid/paddlemetrics/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PaddlePaddle](https://img.shields.io/badge/PaddlePaddle-2.5%2B-7B00DB)](https://www.paddlepaddle.org.cn/)

Machine learning metrics for [PaddlePaddle](https://www.paddlepaddle.org.cn/), providing a comprehensive, production-ready metric library with built-in distributed support.

> Ported from [TorchMetrics](https://github.com/Lightning-AI/torchmetrics) to the PaddlePaddle ecosystem.

## Features

- **100+ metrics** across classification, regression, image, text, retrieval, clustering, segmentation, and more
- **Declarative state management** via `add_state()` — automatic reset, device transfer, and distributed sync
- **Distributed synchronization** — seamless multi-GPU metric aggregation
- **Metric composition** — combine metrics with `+`, `-`, `*`, `/` operators
- **MetricCollection** — batch manage multiple metrics with compute groups optimization
- **Serialization** — `state_dict` / `set_state_dict` for checkpoint saving
- **Pickle support** — full serialization for multiprocessing and persistence

## Installation

```bash
pip install -e .
```

With optional dependencies for specific metric domains:

```bash
pip install -e ".[classification]"   # scikit-learn, scipy
pip install -e ".[regression]"       # scipy
pip install -e ".[image]"            # scipy, scikit-image
pip install -e ".[text]"             # nltk, regex, sentencepiece
pip install -e ".[detection]"        # pycocotools
pip install -e ".[audio]"            # pesq, pystoi, librosa
pip install -e ".[nominal]"          # scipy, pandas
pip install -e ".[visual]"           # matplotlib
pip install -e ".[tests]"            # pytest, scikit-learn, cloudpickle
pip install -e ".[dev]"              # all of the above + ruff
```

## Quick Start

### Single Metric

```python
import paddle
from paddlemetrics.classification import Accuracy

accuracy = Accuracy(task="binary")
preds = paddle.to_tensor([0.9, 0.2, 0.8, 0.1])
target = paddle.to_tensor([1, 0, 1, 0])

# Three-step usage
accuracy.update(preds, target)
print(accuracy.compute())  # tensor(1.)

# Or one-step with forward()
batch_acc = accuracy(preds, target)
```

### MetricCollection

```python
from paddlemetrics import MetricCollection
from paddlemetrics.classification import Accuracy, Precision, Recall

metrics = MetricCollection({
    "accuracy": Accuracy(task="binary"),
    "precision": Precision(task="binary"),
    "recall": Recall(task="binary"),
})

results = metrics(preds, target)
print(results)  # {'accuracy': ..., 'precision': ..., 'recall': ...}
```

### Custom Metric

```python
import paddle
from paddlemetrics import Metric

class BinaryRecall(Metric):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.add_state("tp", default=paddle.zeros([1]), dist_reduce_fx="sum")
        self.add_state("fn", default=paddle.zeros([1]), dist_reduce_fx="sum")

    def update(self, preds, targets):
        preds = (preds > self.threshold).cast(preds.dtype)
        targets = targets.cast(preds.dtype)
        self.tp += ((preds == 1) & (targets == 1)).sum()
        self.fn += ((preds == 0) & (targets == 1)).sum()

    def compute(self):
        return self.tp / (self.tp + self.fn + 1e-7)
```

### Metric Composition

```python
precision = Precision(task="binary")
recall = Recall(task="binary")

# F1 = 2 * (precision * recall) / (precision + recall)
f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
```

## Available Metrics

| Domain | Metrics |
|--------|---------|
| **Classification** | Accuracy, AUROC, AveragePrecision, CalibrationError, CohenKappa, ConfusionMatrix, EER, ExactMatch, F1Score, FBetaScore, HammingDistance, HingeLoss, JaccardIndex, LogAUC, MatthewsCorrCoef, NegativePredictiveValue, Precision, Recall, ROC, Specificity, StatScores, and more |
| **Regression** | MSE, MAE, MAPE, MSLE, R2, PearsonCorrCoef, SpearmanCorrCoef, ConcordanceCorrCoef, CosineSimilarity, KLDivergence, JensenShannonDivergence, LogCosh, Minkowski, ExplainedVariance, KendallRankCorrCoef, TweedieDeviance, WMAPE, and more |
| **Image** | SSIM, MS-SSIM, PSNR, PSNRB, TV, SAM, ERGAS, UQI, VIF, D\_lambda, D\_s, RASE, RMSE\_SW, SCC, QNR |
| **Text** | BLEU, ROUGE, WER, CER, CHRFScore, EditDistance, EED, MER, Perplexity, SacreBLEU, SQuAD, TER, WIL, WIP |
| **Retrieval** | MAP, MRR, NDCG, Precision, Recall, FallOut, HitRate, RPrecision, AUROC |
| **Clustering** | ARI, NMI, Homogeneity, Completeness, VMeasure, ClusterAccuracy, CalinskiHarabasz, DaviesBouldin, DunnIndex, FowlkesMallows, RandScore |
| **Nominal** | CramersV, FleissKappa, PearsonsContingency, TheilsU, TschuprowsT |
| **Segmentation** | DiceScore, GeneralizedDiceScore, HausdorffDistance, MeanIoU |
| **Audio** | SI-SNR, SI-SDR, SDR, SNR, PESQ, STOI, PIT (requires `pesq`, `pystoi`, `librosa`) |
| **Detection** | PanopticQuality, IoU, GIoU, DIoU, CIoU, mAP (requires `pycocotools`) |
| **Wrappers** | BootStrapper, ClasswiseWrapper, MinMaxMetric, MetricTracker, MultitaskWrapper, Running |

## API Compatibility

PaddleMetrics follows the same API design as TorchMetrics:

| Feature | TorchMetrics | PaddleMetrics |
|---------|-------------|---------------|
| Base class | `torch.nn.Module` | `paddle.nn.Layer` |
| State registration | `add_state()` | `add_state()` (alias: `declare()`) |
| Forward | `forward()` | `forward()` |
| Collection | `MetricCollection` | `MetricCollection` |
| Composition | `+`, `-`, `*`, `/` | `+`, `-`, `*`, `/` |
| Serialization | `state_dict()` | `state_dict()` |

## Testing

```bash
# Run core tests
pytest tests/unittests/bases/ -v -m "not DDP"

# Run with coverage
pytest tests/unittests/bases/ --cov=paddlemetrics -m "not DDP"
```

## License

[Apache License 2.0](LICENSE)

## Acknowledgments

This project is a port of [TorchMetrics](https://github.com/Lightning-AI/torchmetrics) by Lightning AI, adapted for the PaddlePaddle ecosystem.
