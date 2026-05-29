from typing import Union

import paddle

from paddlemetrics.functional.text.helper import _edit_distance


def _cer_update(
    preds: Union[str, list[str]], target: Union[str, list[str]]
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Update the cer score with the current set of references and predictions.

    Args:
        preds: Transcription(s) to score as a string or list of strings
        target: Reference(s) for each speech input as a string or list of strings

    Returns:
        Number of edit operations to get from the reference to the prediction, summed over all samples
        Number of character overall references

    """
    if isinstance(preds, str):
        preds = [preds]
    if isinstance(target, str):
        target = [target]
    errors = paddle.tensor(0, dtype=paddle.float32)
    total = paddle.tensor(0, dtype=paddle.float32)
    for pred, tgt in zip(preds, target):
        pred_tokens = pred
        tgt_tokens = tgt
        errors += _edit_distance(list(pred_tokens), list(tgt_tokens))
        total += len(tgt_tokens)
    return errors, total


def _cer_compute(errors: paddle.Tensor, total: paddle.Tensor) -> paddle.Tensor:
    """Compute the Character error rate.

    Args:
        errors: Number of edit operations to get from the reference to the prediction, summed over all samples
        total: Number of characters over all references

    Returns:
        Character error rate score

    """
    return errors / total


def char_error_rate(
    preds: Union[str, list[str]], target: Union[str, list[str]]
) -> paddle.Tensor:
    """Compute Character Error Rate used for performance of an automatic speech recognition system.

    This value indicates the percentage of characters that were incorrectly predicted. The lower the value, the better
    the performance of the ASR system with a CER of 0 being a perfect score.

    Args:
        preds: Transcription(s) to score as a string or list of strings
        target: Reference(s) for each speech input as a string or list of strings

    Returns:
        Character error rate score

    Examples:
        >>> preds = ["this is the prediction", "there is an other sample"]
        >>> target = ["this is the reference", "there is another one"]
        >>> char_error_rate(preds=preds, target=target)
        tensor(0.3415)

    """
    errors, total = _cer_update(preds, target)
    return _cer_compute(errors, total)
