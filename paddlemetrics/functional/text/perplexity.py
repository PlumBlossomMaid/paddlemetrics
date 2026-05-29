from typing import Optional

import paddle


def _check_shape_and_type_consistency(
    preds: paddle.Tensor, target: paddle.Tensor
) -> None:
    """Check shape and type consistency of input vectors.

    Args:
        preds:
            Logits or a unnormalized score assigned to each token in a sequence with shape [batch_size, seq_len,
            vocab_size]. Scores will be normalized internally using softmax.
        target:
            Ground truth values with a shape [batch_size, seq_len].

    Raises:
        ValueError:
            If ``preds`` tensor has no 3 dimensions.
        ValueError:
            If ``target`` tensor has no 2 dimensions.
        ValueError:
            If the first two dimensions of ``preds`` and ``target`` do not equal.
        TypeError:
            If ``preds`` dtype is not one of ``(paddle.float16, paddle.float32, paddle.float64)``
        TypeError:
            If ``target`` is not of a type LongTensor (paddle.int64)

    """
    if len(preds.shape) != 3:
        raise ValueError(
            f"Input tensor `preds` is expected to have 3 dimensions, [batch_size, seq_len, vocab_size], but got {len(preds.shape)}."
        )
    if len(target.shape) != 2:
        raise ValueError(
            f"Input tensor `target` is expected to have 2 dimensions, [batch_size, seq_len], but got {len(target.shape)}."
        )
    if preds.shape[:2] != target.shape:
        raise ValueError(
            f"Input tensors `preds` and `target` are expected to have equaling first two dimensions, [batch_size, seq_len], but got {preds.shape[:2]} and {target.shape}."
        )
    if not preds.is_floating_point():
        raise TypeError(
            f"Input tensor `preds` is expected to be of floating point type but got {preds.dtype}."
        )
    if target.dtype != paddle.int64:
        raise TypeError(
            f"Input tensor `target` is expected to be of a type {paddle.int64} but got {target.dtype}."
        )


def _perplexity_update(
    preds: paddle.Tensor, target: paddle.Tensor, ignore_index: Optional[int] = None
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Compute intermediate statistics for Perplexity.

    Args:
        preds:
            Logits or a unnormalized score assigned to each token in a sequence with shape [batch_size, seq_len,
            vocab_size]. Scores will be normalized internally using softmax.
        target:
            Ground truth values with a shape [batch_size, seq_len].
        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score.

    Returns:
        Log probabilities, summed over all samples
        Number of samples

    """
    _check_shape_and_type_consistency(preds, target)
    probs = paddle.nn.functional.softmax(
        preds.reshape(-1, preds.shape[-1]), axis=1
    )
    target = target.reshape(-1)
    if ignore_index is not None:
        mask = target.ne(ignore_index)
        target = paddle.where(
            target != ignore_index, target, paddle.tensor(0, device=target.place)
        )
    else:
        mask = paddle.ones_like(target, dtype=paddle.bool)
    probs = probs[paddle.arange(target.size), target][mask]
    total_log_probs = -probs.log().sum()
    count = mask.sum()
    return total_log_probs, count


def _perplexity_compute(total: paddle.Tensor, count: paddle.Tensor) -> paddle.Tensor:
    """Compute the Perplexity.

    Args:
        total: Log probabilities, summed over all samples
        count: Number of samples
    Returns:
        Perplexity

    """
    return paddle.exp(total / count)


def perplexity(
    preds: paddle.Tensor, target: paddle.Tensor, ignore_index: Optional[int] = None
) -> paddle.Tensor:
    """Perplexity measures how well a language model predicts a text sample.

    This metric is calculated as the average number of bits per word a model needs to represent the sample.

    Args:
        preds:
            Logits or a unnormalized score assigned to each token in a sequence with shape [batch_size, seq_len,
            vocab_size], which is the output of a language model. Scores will be normalized internally using softmax.
        target:
            Ground truth values with a shape [batch_size, seq_len].
        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score.

    Returns:
        Perplexity value

    Examples:
        >>> from paddle import rand, randint
        >>> preds = rand(2, 8, 5)
        >>> target = randint(5, (2, 8))
        >>> target[0, 6:] = -100
        >>> perplexity(preds, target, ignore_index=-100)
        tensor(5.8540)

    """
    total, count = _perplexity_update(preds, target, ignore_index)
    return _perplexity_compute(total, count)
