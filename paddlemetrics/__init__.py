"""PaddleMetrics - Machine learning metrics for PaddlePaddle."""
import logging as __logging
import os

_logger = __logging.getLogger("paddlemetrics")
_logger.addHandler(__logging.StreamHandler())
_logger.setLevel(__logging.INFO)

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

from paddlemetrics.__about__ import __version__
from paddlemetrics.metric import Metric, CompositionalMetric

__all__ = [
    "__version__",
    "Metric",
    "CompositionalMetric",
]

# Core modules
from paddlemetrics.collections import MetricCollection
from paddlemetrics.aggregation import (
    CatMetric, MaxMetric, MeanMetric, MinMetric, RunningMean, RunningSum, SumMetric,
)

__all__ += [
    "MetricCollection",
    "CatMetric", "MaxMetric", "MeanMetric", "MinMetric",
    "RunningMean", "RunningSum", "SumMetric",
]


def __getattr__(name: str):
    """Lazy imports for domain-specific metrics to avoid cascading import errors."""
    _lazy_imports = {
        # Classification
        "Accuracy": ("paddlemetrics.classification", "Accuracy"),
        "AUROC": ("paddlemetrics.classification", "AUROC"),
        "ROC": ("paddlemetrics.classification", "ROC"),
        "Precision": ("paddlemetrics.classification", "Precision"),
        "Recall": ("paddlemetrics.classification", "Recall"),
        "F1Score": ("paddlemetrics.classification", "F1Score"),
        "FBetaScore": ("paddlemetrics.classification", "FBetaScore"),
        "ConfusionMatrix": ("paddlemetrics.classification", "ConfusionMatrix"),
        "AveragePrecision": ("paddlemetrics.classification", "AveragePrecision"),
        "CalibrationError": ("paddlemetrics.classification", "CalibrationError"),
        "CohenKappa": ("paddlemetrics.classification", "CohenKappa"),
        "EER": ("paddlemetrics.classification", "EER"),
        "ExactMatch": ("paddlemetrics.classification", "ExactMatch"),
        "HammingDistance": ("paddlemetrics.classification", "HammingDistance"),
        "HingeLoss": ("paddlemetrics.classification", "HingeLoss"),
        "JaccardIndex": ("paddlemetrics.classification", "JaccardIndex"),
        "LogAUC": ("paddlemetrics.classification", "LogAUC"),
        "MatthewsCorrCoef": ("paddlemetrics.classification", "MatthewsCorrCoef"),
        "NegativePredictiveValue": ("paddlemetrics.classification", "NegativePredictiveValue"),
        "PrecisionAtFixedRecall": ("paddlemetrics.classification", "PrecisionAtFixedRecall"),
        "PrecisionRecallCurve": ("paddlemetrics.classification", "PrecisionRecallCurve"),
        "RecallAtFixedPrecision": ("paddlemetrics.classification", "RecallAtFixedPrecision"),
        "SensitivityAtSpecificity": ("paddlemetrics.classification", "SensitivityAtSpecificity"),
        "Specificity": ("paddlemetrics.classification", "Specificity"),
        "SpecificityAtSensitivity": ("paddlemetrics.classification", "SpecificityAtSensitivity"),
        "StatScores": ("paddlemetrics.classification", "StatScores"),
        # Regression
        "MeanSquaredError": ("paddlemetrics.regression", "MeanSquaredError"),
        "MeanAbsoluteError": ("paddlemetrics.regression", "MeanAbsoluteError"),
        "R2Score": ("paddlemetrics.regression", "R2Score"),
        "PearsonCorrCoef": ("paddlemetrics.regression", "PearsonCorrCoef"),
        "SpearmanCorrCoef": ("paddlemetrics.regression", "SpearmanCorrCoef"),
        "CosineSimilarity": ("paddlemetrics.regression", "CosineSimilarity"),
        "KLDivergence": ("paddlemetrics.regression", "KLDivergence"),
        "LogCoshError": ("paddlemetrics.regression", "LogCoshError"),
        "MeanAbsolutePercentageError": ("paddlemetrics.regression", "MeanAbsolutePercentageError"),
        "MeanSquaredLogError": ("paddlemetrics.regression", "MeanSquaredLogError"),
        "MinkowskiDistance": ("paddlemetrics.regression", "MinkowskiDistance"),
        "ConcordanceCorrCoef": ("paddlemetrics.regression", "ConcordanceCorrCoef"),
        "ExplainedVariance": ("paddlemetrics.regression", "ExplainedVariance"),
        "KendallRankCorrCoef": ("paddlemetrics.regression", "KendallRankCorrCoef"),
        "SymmetricMeanAbsolutePercentageError": ("paddlemetrics.regression", "SymmetricMeanAbsolutePercentageError"),
        "TweedieDevianceScore": ("paddlemetrics.regression", "TweedieDevianceScore"),
        "WeightedMeanAbsolutePercentageError": ("paddlemetrics.regression", "WeightedMeanAbsolutePercentageError"),
        # Image
        "StructuralSimilarityIndexMeasure": ("paddlemetrics.image", "StructuralSimilarityIndexMeasure"),
        "PeakSignalNoiseRatio": ("paddlemetrics.image", "PeakSignalNoiseRatio"),
        "TotalVariation": ("paddlemetrics.image", "TotalVariation"),
        # Text
        "BLEUScore": ("paddlemetrics.text", "BLEUScore"),
        "WordErrorRate": ("paddlemetrics.text", "WordErrorRate"),
        "CharErrorRate": ("paddlemetrics.text", "CharErrorRate"),
        "ROUGEScore": ("paddlemetrics.text", "ROUGEScore"),
        # Retrieval
        "RetrievalMAP": ("paddlemetrics.retrieval", "RetrievalMAP"),
        "RetrievalNormalizedDCG": ("paddlemetrics.retrieval", "RetrievalNormalizedDCG"),
        # Clustering
        "AdjustedRandScore": ("paddlemetrics.clustering", "AdjustedRandScore"),
        "NormalizedMutualInfoScore": ("paddlemetrics.clustering", "NormalizedMutualInfoScore"),
        # Nominal
        "CramersV": ("paddlemetrics.nominal", "CramersV"),
        "FleissKappa": ("paddlemetrics.nominal", "FleissKappa"),
        # Segmentation
        "DiceScore": ("paddlemetrics.segmentation", "DiceScore"),
        "MeanIoU": ("paddlemetrics.segmentation", "MeanIoU"),
        # Wrappers
        "BootStrapper": ("paddlemetrics.wrappers", "BootStrapper"),
        "ClasswiseWrapper": ("paddlemetrics.wrappers", "ClasswiseWrapper"),
        "MinMaxMetric": ("paddlemetrics.wrappers", "MinMaxMetric"),
        "MetricTracker": ("paddlemetrics.wrappers", "MetricTracker"),
        "MultitaskWrapper": ("paddlemetrics.wrappers", "MultitaskWrapper"),
        "Running": ("paddlemetrics.wrappers", "Running"),
    }
    if name in _lazy_imports:
        module_path, attr_name = _lazy_imports[name]
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, attr_name)
    raise AttributeError(f"module 'paddlemetrics' has no attribute {name!r}")


# Functional (lazy)
def __dir__():
    return __all__ + list(_lazy_imports.keys()) if '_lazy_imports' in dir() else __all__
