from collections.abc import Sequence
from typing import Any, Callable, List, Optional, Tuple, Union, cast

import paddle
import paddleformers

from paddlemetrics.functional.text.bert import (
    _postprocess_multiple_references, _preprocess_multiple_references,
    bert_score)
from paddlemetrics.functional.text.helper_embedding_metric import \
    _preprocess_text
from paddlemetrics.metric import Metric
from paddlemetrics.utils import rank_zero_warn
from paddlemetrics.utils.checks import (_SKIP_SLOW_DOCTEST,
                                           _try_proceed_with_timeout)
from paddlemetrics.utils.data import dim_zero_cat
from paddlemetrics.utils.imports import (_MATPLOTLIB_AVAILABLE,
                                            _TRANSFORMERS_GREATER_EQUAL_4_4)
from paddlemetrics.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["BERTScore.plot"]
_DEFAULT_MODEL: str = "roberta-large"
if _SKIP_SLOW_DOCTEST and _TRANSFORMERS_GREATER_EQUAL_4_4:
    pass

    def _download_model_for_bert_score() -> None:
        """Download intensive operations."""
else:
    __doctest_skip__ = ["BERTScore", "BERTScore.plot"]


def _get_input_dict(
    input_ids: List[paddle.Tensor], attention_mask: List[paddle.Tensor]
) -> dict[str, paddle.Tensor]:
    """Create an input dictionary of ``input_ids`` and ``attention_mask`` for BERTScore calculation."""
    return {
        "input_ids": paddle.concat(input_ids),
        "attention_mask": paddle.concat(attention_mask),
    }


class BERTScore(Metric):
    """`Bert_score Evaluating Text Generation`_ for measuring text similarity.

    BERT leverages the pre-trained contextual embeddings from BERT and matches words in candidate and reference
    sentences by cosine similarity. It has been shown to correlate with human judgment on sentence-level and
    system-level evaluation. Moreover, BERTScore computes precision, recall, and F1 measure, which can be useful for
    evaluating different language generation tasks. This implementation follows the original implementation from
    `BERT_score`_.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds``: Predicted sentence(s). Can be one of:

        * A single predicted sentence as a string (``str``)
        * A sequence of predicted sentences (``Sequence[str]``)

    - ``target``: Target/reference sentence(s). Can be one of:

        * A single reference sentence as a string (``str``)
        * A sequence of reference sentences (``Sequence[str]``)
        * A sequence of sequences of reference sentences for multi-reference evaluation (``Sequence[Sequence[str]]``)

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``score`` (:class:`~Dict`): A dictionary containing the keys ``precision``, ``recall`` and ``f1`` with
      corresponding values

    Args:
        preds (Union[str, Sequence[str]]): A single predicted sentence or a sequence of predicted sentences.
        target (Union[str, Sequence[str], Sequence[Sequence[str]]]): A single target sentence, a sequence of target
            sentences, or a sequence of sequences of target sentences for multiple references per prediction.
        model_type: A name or a model path used to load ``transformers`` pretrained model.
        num_layers: A layer of representation to use.
        all_layers:
            An indication of whether the representation from all model's layers should be used.
            If ``all_layers=True``, the argument ``num_layers`` is ignored.
        model:  A user's own model. Must be of `paddle.nn.Layer` instance.
        user_tokenizer:
            A user's own tokenizer used with the own model. This must be an instance with the ``__call__`` method.
            This method must take an iterable of sentences (`List[str]`) and must return a python dictionary
            containing `"input_ids"` and `"attention_mask"` represented by :class:`~paddle.Tensor`.
            It is up to the user's model of whether `"input_ids"` is a :class:`~paddle.Tensor` of input ids or embedding
            vectors. This tokenizer must prepend an equivalent of ``[CLS]`` token and append an equivalent of ``[SEP]``
            token as ``transformers`` tokenizer does.
        user_forward_fn:
            A user's own forward function used in a combination with ``user_model``. This function must take
            ``user_model`` and a python dictionary of containing ``"input_ids"`` and ``"attention_mask"`` represented
            by :class:`~paddle.Tensor` as an input and return the model's output represented by the single
            :class:`~paddle.Tensor`.
        verbose: An indication of whether a progress bar to be displayed during the embeddings' calculation.
        idf: An indication whether normalization using inverse document frequencies should be used.
        device: A device to be used for calculation.
        max_length: A maximum length of input sequences. Sequences longer than ``max_length`` are to be trimmed.
        batch_size: A batch size used for model processing.
        num_threads: A number of threads to use for a dataloader.
        return_hash: An indication of whether the correspodning ``hash_code`` should be returned.
        lang: A language of input sentences.
        rescale_with_baseline:
            An indication of whether bertscore should be rescaled with a pre-computed baseline.
            When a pretrained model from ``transformers`` model is used, the corresponding baseline is downloaded
            from the original ``bert-score`` package from `BERT_score`_ if available.
            In other cases, please specify a path to the baseline csv/tsv file, which must follow the formatting
            of the files from `BERT_score`_.
        baseline_path: A path to the user's own local csv/tsv file with the baseline scale.
        baseline_url: A url path to the user's own  csv/tsv file with the baseline scale.
        truncation: An indication of whether the input sequences should be truncated to the ``max_length``.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from pprint import pprint
        >>> from paddlemetrics.text.bert import BERTScore
        >>> preds = ["hello there", "general kenobi"]
        >>> target = ["hello there", "master kenobi"]
        >>> bertscore = BERTScore()
        >>> pprint(bertscore(preds, target))
        {'f1': tensor([1.0000, 0.9961]), 'precision': tensor([1.0000, 0.9961]), 'recall': tensor([1.0000, 0.9961])}

    Example:
        >>> from pprint import pprint
        >>> from paddlemetrics.text.bert import BERTScore
        >>> preds = ["hello there", "general kenobi"]
        >>> target = [["hello there", "master kenobi"], ["hello there", "master kenobi"]]
        >>> bertscore = BERTScore()
        >>> pprint(bertscore(preds, target))
        {'f1': tensor([1.0000, 0.9961]), 'precision': tensor([1.0000, 0.9961]), 'recall': tensor([1.0000, 0.9961])}

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    preds_input_ids: List[paddle.Tensor]
    preds_attention_mask: List[paddle.Tensor]
    target_input_ids: List[paddle.Tensor]
    target_attention_mask: List[paddle.Tensor]

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        num_layers: Optional[int] = None,
        all_layers: bool = False,
        model: Optional[paddle.nn.Layer] = None,
        user_tokenizer: Optional[Any] = None,
        user_forward_fn: Optional[
            Callable[[Module, dict[str, paddle.Tensor]], paddle.Tensor]
        ] = None,
        verbose: bool = False,
        idf: bool = False,
        device: Optional[Union[str, paddle.device]] = None,
        max_length: int = 512,
        batch_size: int = 64,
        num_threads: int = 0,
        return_hash: bool = False,
        lang: str = "en",
        rescale_with_baseline: bool = False,
        baseline_path: Optional[str] = None,
        baseline_url: Optional[str] = None,
        truncation: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_name_or_path = model_name_or_path or _DEFAULT_MODEL
        self.num_layers = num_layers
        self.all_layers = all_layers
        self.model = model
        self.user_forward_fn = user_forward_fn
        self.verbose = verbose
        self.idf = idf
        self.embedding_device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.return_hash = return_hash
        self.lang = lang
        self.rescale_with_baseline = rescale_with_baseline
        self.baseline_path = baseline_path
        self.baseline_url = baseline_url
        self.truncation = truncation
        self.ref_group_boundaries: Optional[List[Tuple[int, int]]] = None
        if user_tokenizer:
            self.tokenizer = user_tokenizer
            self.user_tokenizer = True
        else:
            if not _TRANSFORMERS_GREATER_EQUAL_4_4:
                raise ModuleNotFoundError(
                    "`BERTScore` metric with default tokenizers requires `transformers` package be installed. Either install with `pip install transformers>=4.4` or `pip install paddlemetrics[text]`."
                )
            pass
            if model_name_or_path is None:
                rank_zero_warn(
                    f"The argument `model_name_or_path` was not specified while it is required when the default `transformers` model is used. It will use the default recommended model - {_DEFAULT_MODEL!r}."
                )