from collections.abc import Sequence
from functools import partial

import paddle
import pytest
from unittests.text._helpers import TextTester
from unittests.text._inputs import (
    _inputs_multiple_references, _inputs_single_sentence_multiple_references)

from paddlemetrics.functional.text.ter import translation_edit_rate
from paddlemetrics.text.ter import TranslationEditRate


def _reference_sacrebleu_ter(
    preds: Sequence[str],
    target: Sequence[Sequence[str]],
    normalized: bool,
    no_punct: bool,
    asian_support: bool,
    case_sensitive: bool,
) -> paddle.Tensor:
    try:
        from sacrebleu.metrics import TER
    except ImportError:
        pytest.skip("test requires sacrebleu package to be installed")
    sacrebleu_ter = TER(
        normalized=normalized,
        no_punct=no_punct,
        asian_support=asian_support,
        case_sensitive=case_sensitive,
    )
    target = [[tgt[i] for tgt in target] for i in range(len(target[0]))]
    sacrebleu_ter = sacrebleu_ter.corpus_score(preds, target).score / 100
    return paddle.tensor(sacrebleu_ter)


@pytest.mark.parametrize(
    ("normalize", "no_punctuation", "asian_support", "lowercase"),
    [
        (False, False, False, False),
        (True, False, False, False),
        (False, False, False),
        (False, False, False),
        (False, False, False),
        (True),
    ],
)
@pytest.mark.parametrize(
    ("preds", "targets"),
    [(_inputs_multiple_references.preds, _inputs_multiple_references.target)],
)
class TestTER(TextTester):
    """Test class for `TranslationEditRate` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_ter_class(
        self, ddp, preds, targets, normalize, no_punctuation, asian_support, lowercase
    ):
        """Test class implementation of metric."""
        metric_args = {
            "normalize": normalize,
            "no_punctuation": no_punctuation,
            "asian_support": asian_support,
            "lowercase": lowercase,
        }
        nltk_metric = partial(
            _reference_sacrebleu_ter,
            normalized=normalize,
            no_punct=no_punctuation,
            asian_support=asian_support,
            case_sensitive=not lowercase,
        )
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            targets=targets,
            metric_class=TranslationEditRate,
            reference_metric=nltk_metric,
            metric_args=metric_args,
        )

    def test_ter_score_functional(
        self, preds, targets, normalize, no_punctuation, asian_support, lowercase
    ):
        """Test functional implementation of metric."""
        metric_args = {
            "normalize": normalize,
            "no_punctuation": no_punctuation,
            "asian_support": asian_support,
            "lowercase": lowercase,
        }
        nltk_metric = partial(
            _reference_sacrebleu_ter,
            normalized=normalize,
            no_punct=no_punctuation,
            asian_support=asian_support,
            case_sensitive=not lowercase,
        )
        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=translation_edit_rate,
            reference_metric=nltk_metric,
            metric_args=metric_args,
        )

    def test_ter_differentiability(
        self, preds, targets, normalize, no_punctuation, asian_support, lowercase
    ):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        metric_args = {
            "normalize": normalize,
            "no_punctuation": no_punctuation,
            "asian_support": asian_support,
            "lowercase": lowercase,
        }
        self.run_differentiability_test(
            preds=preds,
            targets=targets,
            metric_module=TranslationEditRate,
            metric_functional=translation_edit_rate,
            metric_args=metric_args,
        )


def test_ter_empty_functional():
    """Test that zero is returned on empty input for functional metric."""
    preds = []
    targets = [[]]
    assert translation_edit_rate(preds, targets) == paddle.tensor(0.0)


def test_ter_empty_class():
    """Test that zero is returned on empty input for modular metric."""
    ter_metric = TranslationEditRate()
    preds = []
    targets = [[]]
    assert ter_metric(preds, targets) == paddle.tensor(0.0)


def test_ter_empty_with_non_empty_hyp_functional():
    """Test that zero is returned on empty target input for functional metric."""
    preds = ["python"]
    targets = [[]]
    assert translation_edit_rate(preds, targets) == paddle.tensor(0.0)


def test_ter_empty_with_non_empty_hyp_class():
    """Test that zero is returned on empty target input for modular metric."""
    ter_metric = TranslationEditRate()
    preds = ["python"]
    targets = [[]]
    assert ter_metric(preds, targets) == paddle.tensor(0.0)


def test_ter_return_sentence_level_score_functional():
    """Test that functional metric can return sentence level scores."""
    preds = _inputs_single_sentence_multiple_references.preds
    targets = _inputs_single_sentence_multiple_references.target
    _, sentence_ter = translation_edit_rate(
        preds, targets, return_sentence_level_score=True
    )
    isinstance(sentence_ter, paddle.Tensor)


def test_ter_return_sentence_level_class():
    """Test that modular metric can return sentence level scores."""
    ter_metric = TranslationEditRate(return_sentence_level_score=True)
    preds = _inputs_single_sentence_multiple_references.preds
    targets = _inputs_single_sentence_multiple_references.target
    _, sentence_ter = ter_metric(preds, targets)
    isinstance(sentence_ter, paddle.Tensor)
