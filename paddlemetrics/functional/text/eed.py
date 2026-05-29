import re
import unicodedata
from collections.abc import Sequence
from math import inf
from typing import List, Optional, Union

import paddle
from paddle import Tensor
from typing_extensions import Literal

from paddlemetrics.functional.text.helper import _validate_inputs


def _distance_between_words(preds_word: str, target_word: str) -> int:
    """Distance measure used for substitutions/identity operation.

    Code adapted from https://github.com/rwth-i6/ExtendedEditDistance/blob/master/EED.py.

    Args:
        preds_word: hypothesis word string
        target_word: reference word string

    Return:
        0 for match, 1 for no match

    """
    return int(preds_word != target_word)


def _eed_function(
    hyp: str,
    ref: str,
    alpha: float = 2.0,
    rho: float = 0.3,
    deletion: float = 0.2,
    insertion: float = 1.0,
) -> float:
    """Compute extended edit distance score for two lists of strings: hyp and ref.

    Code adapted from: https://github.com/rwth-i6/ExtendedEditDistance/blob/master/EED.py.

    Args:
        hyp: A hypothesis string
        ref: A reference string
        alpha: optimal jump penalty, penalty for jumps between characters
        rho: coverage cost, penalty for repetition of characters
        deletion: penalty for deletion of character
        insertion: penalty for insertion or substitution of character

    Return:
        Extended edit distance score as float
    """
    number_of_visits = [-1] * (len(hyp) + 1)
    row = [1.0] * (len(hyp) + 1)
    row[0] = 0.0
    next_row = [inf] * (len(hyp) + 1)
    for w in range(1, len(ref) + 1):
        for i in range(len(hyp) + 1):
            if i > 0:
                next_row[i] = min(
                    next_row[i - 1] + deletion,
                    row[i - 1] + _distance_between_words(hyp[i - 1], ref[w - 1]),
                    row[i] + insertion,
                )
            else:
                next_row[i] = row[i] + 1.0
        min_index = next_row.index(min(next_row))
        number_of_visits[min_index] += 1
        if ref[w - 1] == " ":
            jump = alpha + next_row[min_index]
            next_row = [min(x, jump) for x in next_row]
        row = next_row
        next_row = [inf] * (len(hyp) + 1)
    coverage = rho * sum(x if x >= 0 else 1 for x in number_of_visits)
    return min(1, (row[-1] + coverage) / (float(len(ref)) + coverage))


def _preprocess_en(sentence: str) -> str:
    """Preprocess english sentences.

    Copied from https://github.com/rwth-i6/ExtendedEditDistance/blob/master/util.py.

    Raises:
        ValueError: If input sentence is not of a type `str`.

    """
    if not isinstance(sentence, str):
        raise ValueError(
            f"Only strings allowed during preprocessing step, found {type(sentence)} instead"
        )
    sentence = sentence.rstrip()
    rules_interpunction = [(".", " ."), ("!", " !"), ("?", " ?"), (",", " ,")]
    for pattern, replacement in rules_interpunction:
        sentence = sentence.replace(pattern, replacement)
    rules_re = [
        ("\\s+", " "),
        ("(\\d) ([.,]) (\\d)", "\\1\\2\\3"),
        ("(Dr|Jr|Prof|Rev|Gen|Mr|Mt|Mrs|Ms) .", "\\1."),
    ]
    for pattern, replacement in rules_re:
        sentence = re.sub(pattern, replacement, sentence)
    rules_interpunction = [
        ("e . g .", "e.g."),
        ("i . e .", "i.e."),
        ("U . S .", "U.S."),
    ]
    for pattern, replacement in rules_interpunction:
        sentence = sentence.replace(pattern, replacement)
    return " " + sentence + " "


def _preprocess_ja(sentence: str) -> str:
    """Preprocess japanese sentences.

    Copy from https://github.com/rwth-i6/ExtendedEditDistance/blob/master/util.py.

    Raises:
        ValueError: If input sentence is not of a type `str`.

    """
    if not isinstance(sentence, str):
        raise ValueError(
            f"Only strings allowed during preprocessing step, found {type(sentence)} instead"
        )
    sentence = sentence.rstrip()
    return unicodedata.normalize("NFKC", sentence)


def _eed_compute(sentence_level_scores: List[paddle.Tensor]) -> paddle.Tensor:
    """Reduction for extended edit distance.

    Args:
        sentence_level_scores: list of sentence-level scores as floats

    Return:
        average of scores as a tensor

    """
    if len(sentence_level_scores) == 0:
        return paddle.tensor(0.0)
    return sum(sentence_level_scores) / paddle.tensor(len(sentence_level_scores))


def _preprocess_sentences(
    preds: Union[str, Sequence[str]],
    target: Sequence[Union[str, Sequence[str]]],
    language: Literal["en", "ja"],
) -> tuple[Union[str, Sequence[str]], Sequence[Union[str, Sequence[str]]]]:
    """Preprocess strings according to language requirements.

    Args:
        preds: An iterable of hypothesis corpus.
        target: An iterable of iterables of reference corpus.
        language: Language used in sentences. Only supports English (en) and Japanese (ja) for now. Defaults to en

    Return:
        Tuple of lists that contain the cleaned strings for target and preds

    Raises:
        ValueError: If a different language than ``'en'`` or ``'ja'`` is used
        ValueError: If length of target not equal to length of preds
        ValueError: If objects in reference and hypothesis corpus are not strings

    """
    target, preds = _validate_inputs(hypothesis_corpus=preds, ref_corpus=target)
    if language == "en":
        preprocess_function = _preprocess_en
    elif language == "ja":
        preprocess_function = _preprocess_ja
    else:
        raise ValueError(
            f"Expected argument `language` to either be `en` or `ja` but got {language}"
        )
    preds = [preprocess_function(pred) for pred in preds]
    target = [[preprocess_function(ref) for ref in reference] for reference in target]
    return preds, target


def _compute_sentence_statistics(
    preds_word: str,
    target_words: Union[str, Sequence[str]],
    alpha: float = 2.0,
    rho: float = 0.3,
    deletion: float = 0.2,
    insertion: float = 1.0,
) -> paddle.Tensor:
    """Compute scores for ExtendedEditDistance.

    Args:
        target_words: An iterable of reference words
        preds_word: A hypothesis word
        alpha: An optimal jump penalty, penalty for jumps between characters
        rho: coverage cost, penalty for repetition of characters
        deletion: penalty for deletion of character
        insertion: penalty for insertion or substitution of character

    Return:
        best_score: best (lowest) sentence-level score as a Tensor

    """
    best_score = inf
    for reference in target_words:
        score = _eed_function(preds_word, reference, alpha, rho, deletion, insertion)
        if score < best_score:
            best_score = score
    return paddle.tensor(best_score)


def _eed_update(
    preds: Union[str, Sequence[str]],
    target: Sequence[Union[str, Sequence[str]]],
    language: Literal["en", "ja"] = "en",
    alpha: float = 2.0,
    rho: float = 0.3,
    deletion: float = 0.2,
    insertion: float = 1.0,
    sentence_eed: Optional[List[paddle.Tensor]] = None,
) -> List[paddle.Tensor]:
    """Compute scores for ExtendedEditDistance.

    Args:
        preds: An iterable of hypothesis corpus
        target: An iterable of iterables of reference corpus
        language: Language used in sentences. Only supports English (en) and Japanese (ja) for now. Defaults to en
        alpha: optimal jump penalty, penalty for jumps between characters
        rho: coverage cost, penalty for repetition of characters
        deletion: penalty for deletion of character
        insertion: penalty for insertion or substitution of character
        sentence_eed: list of sentence-level scores

    Return:
        individual sentence scores as a list of Tensors

    """
    preds, target = _preprocess_sentences(preds, target, language)
    if sentence_eed is None:
        sentence_eed = []
    if 0 in (len(preds), len(target[0])):
        return sentence_eed
    for hypothesis, target_words in zip(preds, target):
        score = _compute_sentence_statistics(
            hypothesis, target_words, alpha, rho, deletion, insertion
        )
        sentence_eed.append(score)
    return sentence_eed


def extended_edit_distance(
    preds: Union[str, Sequence[str]],
    target: Sequence[Union[str, Sequence[str]]],
    language: Literal["en", "ja"] = "en",
    return_sentence_level_score: bool = False,
    alpha: float = 2.0,
    rho: float = 0.3,
    deletion: float = 0.2,
    insertion: float = 1.0,
) -> Union[paddle.Tensor, tuple[paddle.Tensor, paddle.Tensor]]:
    """Compute extended edit distance score (`ExtendedEditDistance`_) [1] for strings or list of strings.

    The metric utilises the Levenshtein distance and extends it by adding a jump operation.

    Args:
        preds: An iterable of hypothesis corpus.
        target: An iterable of iterables of reference corpus.
        language: Language used in sentences. Only supports English (en) and Japanese (ja) for now. Defaults to en
        return_sentence_level_score: An indication of whether sentence-level EED score is to be returned.
        alpha: optimal jump penalty, penalty for jumps between characters
        rho: coverage cost, penalty for repetition of characters
        deletion: penalty for deletion of character
        insertion: penalty for insertion or substitution of character

    Return:
        Extended edit distance score as a tensor

    Example:
        >>> from paddlemetrics.functional.text import extended_edit_distance
        >>> preds = ["this is the prediction", "here is an other sample"]
        >>> target = ["this is the reference", "here is another one"]
        >>> extended_edit_distance(preds=preds, target=target)
        tensor(0.3078)

    References:
        [1] P. Stanchev, W. Wang, and H. Ney, “EED: Extended Edit Distance Measure for Machine Translation”,
        submitted to WMT 2019. `ExtendedEditDistance`_

    """
    for param_name, param in zip(
        ["alpha", "rho", "deletion", "insertion"], [alpha, rho, deletion, insertion]
    ):
        if not isinstance(param, float) or isinstance(param, float) and param < 0:
            raise ValueError(
                f"Parameter `{param_name}` is expected to be a non-negative float."
            )
    sentence_level_scores = _eed_update(
        preds, target, language, alpha, rho, deletion, insertion
    )
    average = _eed_compute(sentence_level_scores)
    if return_sentence_level_score:
        return average, paddle.stack(sentence_level_scores)
    return average
