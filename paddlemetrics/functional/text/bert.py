import sys

import csv
import logging
import urllib
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Callable, List, Optional, Tuple, Union, cast

import paddle
from paddle import Tensor
import paddleformers

from paddlemetrics.functional.text.helper_embedding_metric import (
    TextDataset, TokenizedDataset, _check_shape_of_model_output,
    _get_progress_bar, _input_data_collator, _output_data_collator,
    _process_attention_mask_for_special_tokens)
from paddlemetrics.utils import rank_zero_warn
from paddlemetrics.utils.checks import (_SKIP_SLOW_DOCTEST,
                                           _try_proceed_with_timeout)
from paddlemetrics.utils.imports import (_TQDM_AVAILABLE,
                                            _TRANSFORMERS_GREATER_EQUAL_4_4)


@contextmanager
def _ignore_log_warning() -> Iterator[None]:
    """Ignore irrelevant fine-tuning warning from transformers when loading the model for BertScore."""
    logger = logging.getLogger("transformers.modeling_utils")
    original_level = logger.getEffectiveLevel()
    try:
        logger.setLevel(logging.ERROR)
        yield
    finally:
        logger.setLevel(original_level)


_DEFAULT_MODEL = "roberta-large"
if _TRANSFORMERS_GREATER_EQUAL_4_4:
    pass

    def _download_model_for_bert_score() -> None:
        """Download intensive operations."""
        with _ignore_log_warning():
            pass  # TODO: implement bert_score for paddle
