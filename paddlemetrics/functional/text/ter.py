import re
from collections.abc import Iterator, Sequence
from functools import lru_cache
from typing import List, Optional, Union

import paddle

from paddlemetrics.functional.text.helper import (_flip_trace,
                                                 _LevenshteinEditDistance,
                                                 _trace_to_alignment,
                                                 _validate_inputs)

_MAX_SHIFT_SIZE = 10
_MAX_SHIFT_DIST = 50
_MAX_SHIFT_CANDIDATES = 1000


class _TercomTokenizer:
    """Re-implementation of Tercom Tokenizer in Python 3.

    See src/ter/core/Normalizer.java in https://github.com/jhclark/tercom Note that Python doesn't support named Unicode
    blocks so the mapping for relevant blocks was taken from here: https://unicode-table.com/en/blocks/

    This implementation follows the implementation from
    https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/tokenizers/tokenizer_ter.py.

    """

    _ASIAN_PUNCTUATION = (
        "([\\u3001\\u3002\\u3008-\\u3011\\u3014-\\u301f\\uff61-\\uff65\\u30fb])"
    )
    _FULL_WIDTH_PUNCTUATION = (
        "([\\uff0e\\uff0c\\uff1f\\uff1a\\uff1b\\uff01\\uff02\\uff08\\uff09])"
    )

    def __init__(
        self,
        normalize: bool = False,
        no_punctuation: bool = False,
        lowercase: bool = True,
        asian_support: bool = False,
    ) -> None:
        """Initialize the tokenizer.

        Args:
            normalize: An indication whether a general tokenization to be applied.
            no_punctuation: An indication whteher a punctuation to be removed from the sentences.
            lowercase: An indication whether to enable case-insensitivity.
            asian_support: An indication whether asian characters to be processed.

        """
        self.normalize = normalize
        self.no_punctuation = no_punctuation
        self.lowercase = lowercase
        self.asian_support = asian_support

    @lru_cache(maxsize=2**16)
    def __call__(self, sentence: str) -> str:
        """Apply a different tokenization techniques according.

        Args:
            sentence: An input sentence to pre-process and tokenize.

        Return:
            A tokenized and pre-processed sentence.

        """
        if not sentence:
            return ""
        if self.lowercase:
            sentence = sentence.lower()
        if self.normalize:
            sentence = self._normalize_general_and_western(sentence)
            if self.asian_support:
                sentence = self._normalize_asian(sentence)
        if self.no_punctuation:
            sentence = self._remove_punct(sentence)
            if self.asian_support:
                sentence = self._remove_asian_punct(sentence)
        return " ".join(sentence.split())

    @staticmethod
    def _normalize_general_and_western(sentence: str) -> str:
        """Apply a language-independent (general) tokenization."""
        sentence = f" {sentence} "
        rules = [
            ("\\n-", ""),
            ("\\n", " "),
            ("&quot;", '"'),
            ("&amp;", "&"),
            ("&lt;", "<"),
            ("&gt;", ">"),
            ("([{-~[-` -&(-+:-@/])", " \\1 "),
            ("'s ", " 's "),
            ("'s$", " 's"),
            ("([^0-9])([\\.,])", "\\1 \\2 "),
            ("([\\.,])([^0-9])", " \\1 \\2"),
            ("([0-9])(-)", "\\1 \\2 "),
        ]
        for pattern, replacement in rules:
            sentence = re.sub(pattern, replacement, sentence)
        return sentence

    @classmethod
    def _normalize_asian(cls: type["_TercomTokenizer"], sentence: str) -> str:
        """Split Chinese chars and Japanese kanji down to character level."""
        sentence = re.sub("([\\u4e00-\\u9fff\\u3400-\\u4dbf])", " \\1 ", sentence)
        sentence = re.sub("([\\u31c0-\\u31ef\\u2e80-\\u2eff])", " \\1 ", sentence)
        sentence = re.sub(
            "([\\u3300-\\u33ff\\uf900-\\ufaff\\ufe30-\\ufe4f])", " \\1 ", sentence
        )
        sentence = re.sub("([\\u3200-\\u3f22])", " \\1 ", sentence)
        sentence = re.sub(
            "(^|^[\\u3040-\\u309f])([\\u3040-\\u309f]+)(?=$|^[\\u3040-\\u309f])",
            "\\1 \\2 ",
            sentence,
        )
        sentence = re.sub(
            "(^|^[\\u30a0-\\u30ff])([\\u30a0-\\u30ff]+)(?=$|^[\\u30a0-\\u30ff])",
            "\\1 \\2 ",
            sentence,
        )
        sentence = re.sub(
            "(^|^[\\u31f0-\\u31ff])([\\u31f0-\\u31ff]+)(?=$|^[\\u31f0-\\u31ff])",
            "\\1 \\2 ",
            sentence,
        )
        sentence = re.sub(cls._ASIAN_PUNCTUATION, " \\1 ", sentence)
        return re.sub(cls._FULL_WIDTH_PUNCTUATION, " \\1 ", sentence)

    @staticmethod
    def _remove_punct(sentence: str) -> str:
        """Remove punctuation from an input sentence string."""
        return re.sub('[\\.,\\?:;!\\"\\(\\)]', "", sentence)

    @classmethod
    def _remove_asian_punct(cls: type["_TercomTokenizer"], sentence: str) -> str:
        """Remove asian punctuation from an input sentence string."""
        sentence = re.sub(cls._ASIAN_PUNCTUATION, "", sentence)
        return re.sub(cls._FULL_WIDTH_PUNCTUATION, "", sentence)


def _preprocess_sentence(sentence: str, tokenizer: _TercomTokenizer) -> str:
    """Given a sentence, apply tokenization.

    Args:
        sentence: The input sentence string.
        tokenizer: An instance of ``_TercomTokenizer`` handling a sentence tokenization.

    Return:
        The pre-processed output sentence string.

    """
    return tokenizer(sentence.rstrip())


def _find_shifted_pairs(
    pred_words: list[str], target_words: list[str]
) -> Iterator[tuple[int, int, int]]:
    """Find matching word sub-sequences in two lists of words. Ignores sub- sequences starting at the same position.

    Args:
        pred_words: A list of a tokenized hypothesis sentence.
        target_words: A list of a tokenized reference sentence.

    Return:
        Yields tuples of ``target_start, pred_start, length`` such that:
        ``target_words[target_start : target_start + length] == pred_words[pred_start : pred_start + length]``

        pred_start:
            A list of hypothesis start indices.
        target_start:
            A list of reference start indices.
        length:
            A length of a word span to be considered.

    """
    for pred_start in range(len(pred_words)):
        for target_start in range(len(target_words)):
            if abs(target_start - pred_start) > _MAX_SHIFT_DIST:
                continue
            for length in range(1, _MAX_SHIFT_SIZE):
                if (
                    pred_words[pred_start + length - 1]
                    != target_words[target_start + length - 1]
                ):
                    break
                yield pred_start, target_start, length
                _hyp = len(pred_words) == pred_start + length
                _ref = len(target_words) == target_start + length
                if _hyp or _ref:
                    break


def _handle_corner_cases_during_shifting(
    alignments: dict[int, int],
    pred_errors: list[int],
    target_errors: list[int],
    pred_start: int,
    target_start: int,
    length: int,
) -> bool:
    """Return ``True`` if any of corner cases has been met. Otherwise, ``False`` is returned.

    Args:
        alignments: A dictionary mapping aligned positions between a reference and a hypothesis.
        pred_errors: A list of error positions in a hypothesis.
        target_errors: A list of error positions in a reference.
        pred_start: A hypothesis start index.
        target_start: A reference start index.
        length: A length of a word span to be considered.

    Return:
        An indication whether any of conrner cases has been met.

    """
    if sum(pred_errors[pred_start : pred_start + length]) == 0:
        return True
    if sum(target_errors[target_start : target_start + length]) == 0:
        return True
    return pred_start <= alignments[target_start] < pred_start + length


def _perform_shift(words: list[str], start: int, length: int, target: int) -> list[str]:
    """Perform a shift in ``words`` from ``start`` to ``target``.

    Args:
        words: A words to shift.
        start: An index where to start shifting from.
        length: A number of how many words to be considered.
        target: An index where to end shifting.

    Return:
        A list of shifted words.

    """

    def _shift_word_before_previous_position(
        words: list[str], start: int, target: int, length: int
    ) -> list[str]:
        return (
            words[:target]
            + words[start : start + length]
            + words[target:start]
            + words[start + length :]
        )

    def _shift_word_after_previous_position(
        words: list[str], start: int, target: int, length: int
    ) -> list[str]:
        return (
            words[:start]
            + words[start + length : target]
            + words[start : start + length]
            + words[target:]
        )

    def _shift_word_within_shifted_string(
        words: list[str], start: int, target: int, length: int
    ) -> list[str]:
        shifted_words = words[:start]
        shifted_words += words[start + length : length + target]
        shifted_words += words[start : start + length]
        shifted_words += words[length + target :]
        return shifted_words

    if target < start:
        return _shift_word_before_previous_position(words, start, target, length)
    if target > start + length:
        return _shift_word_after_previous_position(words, start, target, length)
    return _shift_word_within_shifted_string(words, start, target, length)


def _shift_words(
    pred_words: list[str],
    target_words: list[str],
    cached_edit_distance: _LevenshteinEditDistance,
    checked_candidates: int,
) -> tuple[int, list[str], int]:
    """Attempt to shift words to match a hypothesis with a reference.

    It returns the lowest number of required edits between a hypothesis and a provided reference, a list of shifted
    words and number of checked candidates. Note that the filtering of possible shifts and shift selection are heavily
    based on somewhat arbitrary heuristics. The code here follows as closely as possible the logic in Tercom, not
    always justifying the particular design choices.
    The paragraph copied from https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/lib_ter.py.

    Args:
        pred_words: A list of tokenized hypothesis sentence.
        target_words: A list of lists of tokenized reference sentences.
        cached_edit_distance: A pre-computed edit distance between a hypothesis and a reference.
        checked_candidates: A number of checked hypothesis candidates to match a provided reference.

    Return:
        best_score:
            The best (lowest) number of required edits to match hypothesis and reference sentences.
        shifted_words:
            A list of shifted words in hypothesis sentences.
        checked_candidates:
            A number of checked hypothesis candidates to match a provided reference.

    """
    edit_distance, inverted_trace = cached_edit_distance(pred_words)
    trace = _flip_trace(inverted_trace)
    alignments, target_errors, pred_errors = _trace_to_alignment(trace)
    best: Optional[tuple[int, int, int, int, list[str]]] = None
    for pred_start, target_start, length in _find_shifted_pairs(
        pred_words, target_words
    ):
        if _handle_corner_cases_during_shifting(
            alignments, pred_errors, target_errors, pred_start, target_start, length
        ):
            continue
        prev_idx = -1
        for offset in range(-1, length):
            if target_start + offset == -1:
                idx = 0
            elif target_start + offset in alignments:
                idx = alignments[target_start + offset] + 1
            else:
                break
            if idx == prev_idx:
                continue
            prev_idx = idx
            shifted_words = _perform_shift(pred_words, pred_start, length, idx)
            candidate = (
                edit_distance - cached_edit_distance(shifted_words)[0],
                length,
                -pred_start,
                -idx,
                shifted_words,
            )
            checked_candidates += 1
            if not best or candidate > best:
                best = candidate
        if checked_candidates >= _MAX_SHIFT_CANDIDATES:
            break
    if not best:
        return 0, pred_words, checked_candidates
    best_score, _, _, _, shifted_words = best
    return best_score, shifted_words, checked_candidates


def _translation_edit_rate(
    pred_words: list[str], target_words: list[str]
) -> paddle.Tensor:
    """Compute translation edit rate between hypothesis and reference sentences.

    Args:
        pred_words: A list of a tokenized hypothesis sentence.
        target_words: A list of lists of tokenized reference sentences.

    Return:
        A number of required edits to match hypothesis and reference sentences.

    """
    if len(target_words) == 0:
        return paddle.tensor(0.0)
    cached_edit_distance = _LevenshteinEditDistance(target_words)
    num_shifts = 0
    checked_candidates = 0
    input_words = pred_words
    while True:
        delta, new_input_words, checked_candidates = _shift_words(
            input_words, target_words, cached_edit_distance, checked_candidates
        )
        if checked_candidates >= _MAX_SHIFT_CANDIDATES or delta <= 0:
            break
        num_shifts += 1
        input_words = new_input_words
    edit_distance, _ = cached_edit_distance(input_words)
    total_edits = num_shifts + edit_distance
    return paddle.tensor(total_edits)


def _compute_sentence_statistics(
    pred_words: list[str], target_words: list[list[str]]
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Compute sentence TER statistics between hypothesis and provided references.

    Args:
        pred_words: A list of tokenized hypothesis sentence.
        target_words: A list of lists of tokenized reference sentences.

    Return:
        best_num_edits:
            The best (lowest) number of required edits to match hypothesis and reference sentences.
        avg_tgt_len:
            Average length of tokenized reference sentences.

    """
    tgt_lengths = paddle.tensor(0.0)
    best_num_edits = paddle.tensor(2e16)
    for tgt_words in target_words:
        num_edits = _translation_edit_rate(tgt_words, pred_words)
        tgt_lengths += len(tgt_words)
        if num_edits < best_num_edits:
            best_num_edits = num_edits
    avg_tgt_len = tgt_lengths / len(target_words)
    return best_num_edits, avg_tgt_len


def _compute_ter_score_from_statistics(
    num_edits: paddle.Tensor, tgt_length: paddle.Tensor
) -> paddle.Tensor:
    """Compute TER score based on pre-computed a number of edits and an average reference length.

    Args:
        num_edits: A number of required edits to match hypothesis and reference sentences.
        tgt_length: An average length of reference sentences.

    Return:
        A corpus-level TER score or 1 if reference_length == 0.

    """
    if tgt_length > 0 and num_edits > 0:
        return num_edits / tgt_length
    if tgt_length == 0 and num_edits > 0:
        return paddle.tensor(1.0)
    return paddle.tensor(0.0)


def _ter_update(
    preds: Union[str, Sequence[str]],
    target: Sequence[Union[str, Sequence[str]]],
    tokenizer: _TercomTokenizer,
    total_num_edits: paddle.Tensor,
    total_tgt_length: paddle.Tensor,
    sentence_ter: Optional[List[paddle.Tensor]] = None,
) -> tuple[paddle.Tensor, paddle.Tensor, Optional[List[paddle.Tensor]]]:
    """Update TER statistics.

    Args:
        preds: An iterable of hypothesis corpus.
        target: An iterable of iterables of reference corpus.
        tokenizer: An instance of ``_TercomTokenizer`` handling a sentence tokenization.
        total_num_edits: A total number of required edits to match hypothesis and reference sentences.
        total_tgt_length: A total average length of reference sentences.
        sentence_ter: A list of sentence-level TER values

    Return:
        total_num_edits:
            A total number of required edits to match hypothesis and reference sentences.
        total_tgt_length:
            A total average length of reference sentences.
        sentence_ter:
            (Optionally) A list of sentence-level TER.

    Raises:
        ValueError:
            If length of ``preds`` and ``target`` differs.

    """
    target, preds = _validate_inputs(target, preds)
    for pred, tgt in zip(preds, target):
        tgt_words_: list[list[str]] = [
            _preprocess_sentence(_tgt, tokenizer).split() for _tgt in tgt
        ]
        pred_words_: list[str] = _preprocess_sentence(pred, tokenizer).split()
        num_edits, tgt_length = _compute_sentence_statistics(pred_words_, tgt_words_)
        total_num_edits += num_edits
        total_tgt_length += tgt_length
        if sentence_ter is not None:
            sentence_ter.append(
                _compute_ter_score_from_statistics(num_edits, tgt_length).unsqueeze(0)
            )
    return total_num_edits, total_tgt_length, sentence_ter


def _ter_compute(
    total_num_edits: paddle.Tensor, total_tgt_length: paddle.Tensor
) -> paddle.Tensor:
    """Compute TER based on pre-computed a total number of edits and a total average reference length.

    Args:
        total_num_edits: A total number of required edits to match hypothesis and reference sentences.
        total_tgt_length: A total average length of reference sentences.

    Return:
        A corpus-level TER score.

    """
    return _compute_ter_score_from_statistics(total_num_edits, total_tgt_length)


def translation_edit_rate(
    preds: Union[str, Sequence[str]],
    target: Sequence[Union[str, Sequence[str]]],
    normalize: bool = False,
    no_punctuation: bool = False,
    lowercase: bool = True,
    asian_support: bool = False,
    return_sentence_level_score: bool = False,
) -> Union[paddle.Tensor, tuple[paddle.Tensor, List[paddle.Tensor]]]:
    """Calculate Translation edit rate (`TER`_)  of machine translated text with one or more references.

    This implementation follows the implementations from
    https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/ter.py. The `sacrebleu` implementation is a
    near-exact reimplementation of the Tercom algorithm, produces identical results on all "sane" outputs.

    Args:
        preds: An iterable of hypothesis corpus.
        target: An iterable of iterables of reference corpus.
        normalize: An indication whether a general tokenization to be applied.
        no_punctuation: An indication whteher a punctuation to be removed from the sentences.
        lowercase: An indication whether to enable case-insensitivity.
        asian_support: An indication whether asian characters to be processed.
        return_sentence_level_score: An indication whether a sentence-level TER to be returned.

    Return:
        A corpus-level translation edit rate (TER).
        (Optionally) A list of sentence-level translation_edit_rate (TER) if `return_sentence_level_score=True`.

    Example:
        >>> preds = ['the cat is on the mat']
        >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
        >>> translation_edit_rate(preds, target)
        tensor(0.1538)

    References:
        [1] A Study of Translation Edit Rate with Targeted Human Annotation
        by Mathew Snover, Bonnie Dorr, Richard Schwartz, Linnea Micciulla and John Makhoul `TER`_

    """
    if not isinstance(normalize, bool):
        raise ValueError(
            f"Expected argument `normalize` to be of type boolean but got {normalize}."
        )
    if not isinstance(no_punctuation, bool):
        raise ValueError(
            f"Expected argument `no_punctuation` to be of type boolean but got {no_punctuation}."
        )
    if not isinstance(lowercase, bool):
        raise ValueError(
            f"Expected argument `lowercase` to be of type boolean but got {lowercase}."
        )
    if not isinstance(asian_support, bool):
        raise ValueError(
            f"Expected argument `asian_support` to be of type boolean but got {asian_support}."
        )
    tokenizer: _TercomTokenizer = _TercomTokenizer(
        normalize, no_punctuation, lowercase, asian_support
    )
    total_num_edits = paddle.tensor(0.0)
    total_tgt_length = paddle.tensor(0.0)
    sentence_ter: Optional[List[paddle.Tensor]] = (
        [] if return_sentence_level_score else None
    )
    total_num_edits, total_tgt_length, sentence_ter = _ter_update(
        preds, target, tokenizer, total_num_edits, total_tgt_length, sentence_ter
    )
    ter_score = _ter_compute(total_num_edits, total_tgt_length)
    if sentence_ter:
        return ter_score, sentence_ter
    return ter_score
