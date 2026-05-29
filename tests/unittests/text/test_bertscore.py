import os
from collections.abc import Sequence
from functools import partial

import paddle
import pytest
from typing_extensions import Literal
from unittests._helpers import (_IS_WINDOWS, _TORCH_LESS_THAN_2_1,
                                _TRANSFORMERS_GREATER_EQUAL_4_54,
                                _TRANSFORMERS_RANGE_GE_4_50_LT_4_54,
                                skip_on_connection_issues)
from unittests.text._helpers import TextTester
from unittests.text._inputs import (
    _inputs_multiple_references, _inputs_single_reference,
    _inputs_single_sentence_multiple_references)

from paddlemetrics.functional.text.bert import bert_score
from paddlemetrics.text.bert import BERTScore
from paddlemetrics.utils.imports import _TRANSFORMERS_GREATER_EQUAL_4_4

_METRIC_KEY_TO_IDX = {"precision": 0, "recall": 1, "f1": 2}
MODEL_NAME = "albert-base-v2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@skip_on_connection_issues()
@pytest.mark.skipif(
    not _TRANSFORMERS_GREATER_EQUAL_4_4, reason="test requires transformers>4.4"
)
def _reference_bert_score(
    preds: Sequence[str],
    target: Sequence[str],
    num_layers: int,
    all_layers: bool,
    idf: bool,
    rescale_with_baseline: bool,
    metric_key: Literal["f1", "precision", "recall"],
) -> paddle.Tensor:
    try:
        from bert_score import score as original_bert_score
    except ImportError:
        pytest.skip("test requires bert_score package to be installed.")
    score_tuple = original_bert_score(
        preds,
        target,
        model_type=MODEL_NAME,
        lang="en",
        num_layers=num_layers,
        all_layers=all_layers,
        idf=idf,
        batch_size=len(preds),
        rescale_with_baseline=rescale_with_baseline,
        nthreads=0,
    )
    return score_tuple[_METRIC_KEY_TO_IDX[metric_key]]


@pytest.mark.parametrize(
    ("num_layers", "all_layers", "idf", "rescale_with_baseline", "metric_key"),
    [
        (8, False, False, False, "precision"),
        (12, False, False, "recall"),
        (12, False, False, "f1"),
        (8, False, False, "precision"),
        (12, False, "recall"),
        (12, False, "f1"),
        (8, False, "precision"),
        (12, "f1"),
    ],
)
@pytest.mark.parametrize(
    ("preds", "targets"),
    [(_inputs_single_reference.preds, _inputs_single_reference.target)],
)
@pytest.mark.skipif(
    not _TRANSFORMERS_GREATER_EQUAL_4_4, reason="test requires transformers>4.4"
)
@pytest.mark.xfail(
    RuntimeError,
    condition=_TORCH_LESS_THAN_2_1 and _TRANSFORMERS_RANGE_GE_4_50_LT_4_54,
    reason="could be due to torch compatibility issues with transformers",
)
@pytest.mark.xfail(
    ImportError,
    condition=_TORCH_LESS_THAN_2_1 and _IS_WINDOWS and _TRANSFORMERS_GREATER_EQUAL_4_54,
    reason="another strange behaviour of transformers on windows",
)
@skip_on_connection_issues()
class TestBERTScore(TextTester):
    """Tests for BERTScore."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_bertscore_class(
        self,
        ddp,
        preds,
        targets,
        num_layers,
        all_layers,
        idf,
        rescale_with_baseline,
        metric_key,
    ):
        """Test the bert score class."""
        metric_args = {
            "model_name_or_path": MODEL_NAME,
            "num_layers": num_layers,
            "all_layers": all_layers,
            "idf": idf,
            "rescale_with_baseline": rescale_with_baseline,
        }
        reference_bert_score_metric = partial(
            _reference_bert_score,
            num_layers=num_layers,
            all_layers=all_layers,
            idf=idf,
            rescale_with_baseline=rescale_with_baseline,
            metric_key=metric_key,
        )
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            targets=targets,
            metric_class=BERTScore,
            reference_metric=reference_bert_score_metric,
            metric_args=metric_args,
            key=metric_key,
            check_scriptable=False,
            ignore_order=ddp,
        )

    def test_bertscore_functional(
        self,
        preds,
        targets,
        num_layers,
        all_layers,
        idf,
        rescale_with_baseline,
        metric_key,
    ):
        """Test the bertscore functional."""
        metric_args = {
            "model_name_or_path": MODEL_NAME,
            "num_layers": num_layers,
            "all_layers": all_layers,
            "idf": idf,
            "rescale_with_baseline": rescale_with_baseline,
        }
        reference_bert_score_metric = partial(
            _reference_bert_score,
            num_layers=num_layers,
            all_layers=all_layers,
            idf=idf,
            rescale_with_baseline=rescale_with_baseline,
            metric_key=metric_key,
        )
        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=bert_score,
            reference_metric=reference_bert_score_metric,
            metric_args=metric_args,
            key=metric_key,
        )

    def test_bertscore_differentiability(
        self,
        preds,
        targets,
        num_layers,
        all_layers,
        idf,
        rescale_with_baseline,
        metric_key,
    ):
        """Test the bertscore differentiability."""
        metric_args = {
            "model_name_or_path": MODEL_NAME,
            "num_layers": num_layers,
            "all_layers": all_layers,
            "idf": idf,
            "rescale_with_baseline": rescale_with_baseline,
        }
        self.run_differentiability_test(
            preds=preds,
            targets=targets,
            metric_module=BERTScore,
            metric_functional=bert_score,
            metric_args=metric_args,
            key=metric_key,
        )


@skip_on_connection_issues()
@pytest.mark.skipif(
    not _TRANSFORMERS_GREATER_EQUAL_4_4, reason="test requires transformers>4.4"
)
@pytest.mark.xfail(
    RuntimeError,
    condition=_TORCH_LESS_THAN_2_1 and _TRANSFORMERS_RANGE_GE_4_50_LT_4_54,
    reason="could be due to torch compatibility issues with transformers",
)
@pytest.mark.xfail(
    ImportError,
    condition=_TORCH_LESS_THAN_2_1 and _IS_WINDOWS and _TRANSFORMERS_GREATER_EQUAL_4_54,
    reason="another strange behaviour of transformers on windows",
)
@pytest.mark.parametrize("idf", [True, False])
def test_bertscore_sorting(idf: bool):
    """Test that BERTScore is invariant to the order of the inputs."""
    short = "Short text"
    long = "This is a longer text"
    preds = [long, long]
    targets = [long, short]
    metric = BERTScore(idf=idf)
    score = metric(preds, targets)
    assert score["f1"][0] > score["f1"][1]


@skip_on_connection_issues()
@pytest.mark.skipif(
    not _TRANSFORMERS_GREATER_EQUAL_4_4, reason="test requires transformers>4.4"
)
@pytest.mark.xfail(
    RuntimeError,
    condition=_TORCH_LESS_THAN_2_1 and _TRANSFORMERS_RANGE_GE_4_50_LT_4_54,
    reason="could be due to torch compatibility issues with transformers",
)
@pytest.mark.xfail(
    ImportError,
    condition=_TORCH_LESS_THAN_2_1 and _IS_WINDOWS and _TRANSFORMERS_GREATER_EQUAL_4_54,
    reason="another strange behaviour of transformers on windows",
)
@pytest.mark.parametrize("truncation", [True, False])
def test_bertscore_truncation(truncation: bool):
    """Test that BERTScore truncation works as expected."""
    pred = ["abc " * 2000]
    gt = ["def " * 2000]
    bert_score = BERTScore(truncation=truncation)
    if truncation:
        res = bert_score(pred, gt)
        assert res["f1"] > 0.0
    else:
        with pytest.raises(
            RuntimeError, match="The expanded size of the tensor.*must match.*"
        ):
            bert_score(pred, gt)


@skip_on_connection_issues()
@pytest.mark.skipif(
    not _TRANSFORMERS_GREATER_EQUAL_4_4, reason="test requires transformers>4.4"
)
@pytest.mark.xfail(
    RuntimeError,
    condition=_TORCH_LESS_THAN_2_1 and _TRANSFORMERS_RANGE_GE_4_50_LT_4_54,
    reason="could be due to torch compatibility issues with transformers",
)
@pytest.mark.xfail(
    ImportError,
    condition=_TORCH_LESS_THAN_2_1 and _IS_WINDOWS and _TRANSFORMERS_GREATER_EQUAL_4_54,
    reason="another strange behaviour of transformers on windows",
)
def test_bertscore_single_str_input():
    """Test if BERTScore works with single string preds and target."""
    preds = "hello there"
    target = "hello there"
    metric = BERTScore()
    score_class = metric(preds, target)
    assert score_class["f1"].item() == pytest.approx(1.0, abs=0.0001)
    assert score_class["precision"].item() == pytest.approx(1.0, abs=0.0001)
    assert score_class["recall"].item() == pytest.approx(1.0, abs=0.0001)
    score_functional = bert_score(preds, target)
    assert score_functional["f1"].item() == pytest.approx(1.0, abs=0.0001)
    assert score_functional["precision"].item() == pytest.approx(1.0, abs=0.0001)
    assert score_functional["recall"].item() == pytest.approx(1.0, abs=0.0001)


@pytest.mark.parametrize(
    ("preds", "target", "expected"),
    [
        (
            _inputs_single_sentence_multiple_references.preds,
            _inputs_single_sentence_multiple_references.target,
            {
                "precision": paddle.tensor([0.997]),
                "recall": paddle.tensor([0.997]),
                "f1": paddle.tensor([0.997]),
            },
        ),
        (
            ["hello there", "I'm in the middle", "general kenobi"],
            (
                ["hello there", "master kenobi"],
                "I'm here",
                ("hello there", "master kenobi"),
            ),
            {
                "precision": paddle.tensor([1.0, 0.981, 0.9961]),
                "recall": paddle.tensor([1.0, 0.9811, 0.9961]),
                "f1": paddle.tensor([1.0, 0.9811, 0.9961]),
            },
        ),
    ],
)
@skip_on_connection_issues()
@pytest.mark.skipif(
    not _TRANSFORMERS_GREATER_EQUAL_4_4, reason="test requires transformers>4.4"
)
@pytest.mark.xfail(
    RuntimeError,
    condition=_TORCH_LESS_THAN_2_1 and _TRANSFORMERS_RANGE_GE_4_50_LT_4_54,
    reason="could be due to torch compatibility issues with transformers",
)
@pytest.mark.xfail(
    ImportError,
    condition=_TORCH_LESS_THAN_2_1 and _IS_WINDOWS and _TRANSFORMERS_GREATER_EQUAL_4_54,
    reason="another strange behaviour of transformers on windows",
)
def test_bertscore_multiple_references(preds, target, expected):
    """Test both functional and class APIs with multiple references."""
    result_func = bert_score(preds, target)
    for k in expected:
        assert paddle.allclose(
            x=result_func[k], y=expected[k], atol=0.0001
        ).item(), f"Functional {k} mismatch: {result_func[k]} vs {expected[k]}"
    metric = BERTScore()
    result_class = metric(preds, target)
    for k in expected:
        assert paddle.allclose(
            x=result_class[k], y=expected[k], atol=0.0001
        ).item(), f"Class {k} mismatch: {result_class[k]} vs {expected[k]}"


@pytest.mark.skipif(
    not _TRANSFORMERS_GREATER_EQUAL_4_4, reason="test requires transformers>4.4"
)
def test_bertscore_invalid_references():
    """Test both functional and class APIs with invalid references."""
    preds = _inputs_multiple_references.preds
    target = _inputs_multiple_references.target
    with pytest.raises(ValueError, match="Invalid input provided."):
        bert_score(preds, target)
    metric = BERTScore()
    with pytest.raises(ValueError, match="Invalid input provided."):
        metric(preds, target)
