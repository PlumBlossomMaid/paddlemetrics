import sys

from typing import Any

import paddle
import pytest
from unittests import (BATCH_SIZE, EXTRA_DIM, NUM_BATCHES, NUM_CLASSES,
                       _GroupInput, _Input)
from unittests._helpers import seed_all

seed_all(1)


def _inv_sigmoid(x: paddle.Tensor) -> paddle.Tensor:
    return (x / (1 - x)).log()


def _logsoftmax(x: paddle.Tensor, dim: int = -1) -> paddle.Tensor:
    return paddle.nn.functional.log_softmax(x=x, axis=dim)


_input_binary_prob = _Input(
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE),
    target=paddle.randint(low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE)),
)
_input_binary = _Input(
    preds=paddle.randint(low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE)),
    target=paddle.randint(low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE)),
)
_input_binary_logits = _Input(
    preds=paddle.randn(NUM_BATCHES, BATCH_SIZE),
    target=paddle.randint(low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE)),
)
_input_multilabel_prob = _Input(
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES),
    target=paddle.randint(low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)),
)
_input_multilabel_multidim_prob = _Input(
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM),
    target=paddle.randint(
        low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM)
    ),
)
_input_multilabel_logits = _Input(
    preds=paddle.randn(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES),
    target=paddle.randint(low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)),
)
_input_multilabel = _Input(
    preds=paddle.randint(low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)),
    target=paddle.randint(low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)),
)
_input_multilabel_multidim = _Input(
    preds=paddle.randint(
        low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM)
    ),
    target=paddle.randint(
        low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM)
    ),
)
_binary_cases = (
    pytest.param(
        _Input(
            preds=paddle.randint(low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE)),
            target=paddle.randint(low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE)),
        ),
        id="input[single_dim-labels]",
    ),
    pytest.param(
        _Input(
            preds=paddle.rand(NUM_BATCHES, BATCH_SIZE),
            target=paddle.randint(low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE)),
        ),
        id="input[single_dim-probs]",
    ),
    pytest.param(
        _Input(
            preds=_inv_sigmoid(paddle.rand(NUM_BATCHES, BATCH_SIZE)),
            target=paddle.randint(low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE)),
        ),
        id="input[single_dim-logits]",
    ),
    pytest.param(
        _Input(
            preds=paddle.randint(
                low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)
            ),
            target=paddle.randint(
                low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)
            ),
        ),
        id="input[multi_dim-labels]",
    ),
    pytest.param(
        _Input(
            preds=paddle.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM),
            target=paddle.randint(
                low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)
            ),
        ),
        id="input[multi_dim-probs]",
    ),
    pytest.param(
        _Input(
            preds=_inv_sigmoid(paddle.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)),
            target=paddle.randint(
                low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)
            ),
        ),
        id="input[multi_dim-logits]",
    ),
)


def _multiclass_with_missing_class(*shape: Any, num_classes=NUM_CLASSES):
    """Generate multiclass input where a class is missing.

    Args:
        shape: shape of the tensor
        num_classes: number of classes

    Returns:
        tensor with missing classes

    """
    x = paddle.randint(low=0, high=num_classes, shape=shape)
    x[x == 0] = 2
    return x


_multiclass_cases = (
    pytest.param(
        _Input(
            preds=paddle.randint(
                low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE)
            ),
            target=paddle.randint(
                low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE)
            ),
        ),
        id="input[single_dim-labels]",
    ),
    pytest.param(
        _Input(
            preds=paddle.randn(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES).softmax(-1),
            target=paddle.randint(
                low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE)
            ),
        ),
        id="input[single_dim-probs]",
    ),
    pytest.param(
        _Input(
            preds=_logsoftmax(paddle.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES), -1),
            target=paddle.randint(
                low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE)
            ),
        ),
        id="input[single_dim-logits]",
    ),
    pytest.param(
        _Input(
            preds=paddle.randint(
                low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)
            ),
            target=paddle.randint(
                low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)
            ),
        ),
        id="input[multi_dim-labels]",
    ),
    pytest.param(
        _Input(
            preds=paddle.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM).softmax(
                -2
            ),
            target=paddle.randint(
                low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)
            ),
        ),
        id="input[multi_dim-probs]",
    ),
    pytest.param(
        _Input(
            preds=_logsoftmax(
                paddle.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM), -2
            ),
            target=paddle.randint(
                low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)
            ),
        ),
        id="input[multi_dim-logits]",
    ),
    pytest.param(
        _Input(
            preds=_multiclass_with_missing_class(
                NUM_BATCHES, BATCH_SIZE, num_classes=NUM_CLASSES
            ),
            target=_multiclass_with_missing_class(
                NUM_BATCHES, BATCH_SIZE, num_classes=NUM_CLASSES
            ),
        ),
        id="input[single_dim-labels-missing_class]",
    ),
)
_multilabel_cases = (
    pytest.param(
        _Input(
            preds=paddle.randint(
                low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)
            ),
            target=paddle.randint(
                low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)
            ),
        ),
        id="input[single_dim-labels]",
    ),
    pytest.param(
        _Input(
            preds=paddle.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES),
            target=paddle.randint(
                low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)
            ),
        ),
        id="input[single_dim-probs]",
    ),
    pytest.param(
        _Input(
            preds=_inv_sigmoid(paddle.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)),
            target=paddle.randint(
                low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)
            ),
        ),
        id="input[single_dim-logits]",
    ),
    pytest.param(
        _Input(
            preds=paddle.randint(
                low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM)
            ),
            target=paddle.randint(
                low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM)
            ),
        ),
        id="input[multi_dim-labels]",
    ),
    pytest.param(
        _Input(
            preds=paddle.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM),
            target=paddle.randint(
                low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM)
            ),
        ),
        id="input[multi_dim-probs]",
    ),
    pytest.param(
        _Input(
            preds=_inv_sigmoid(
                paddle.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM)
            ),
            target=paddle.randint(
                low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM)
            ),
        ),
        id="input[multi_dim-logits]",
    ),
)
_group_cases = (
    pytest.param(
        _GroupInput(
            preds=paddle.randint(low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE)),
            target=paddle.randint(low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE)),
            groups=paddle.randint(low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE)),
        ),
        id="input[single_dim-labels]",
    ),
    pytest.param(
        _GroupInput(
            preds=paddle.rand(NUM_BATCHES, BATCH_SIZE),
            target=paddle.randint(low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE)),
            groups=paddle.randint(low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE)),
        ),
        id="input[single_dim-probs]",
    ),
    pytest.param(
        _GroupInput(
            preds=_inv_sigmoid(paddle.rand(NUM_BATCHES, BATCH_SIZE)),
            target=paddle.randint(low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE)),
            groups=paddle.randint(low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE)),
        ),
        id="input[single_dim-logits]",
    ),
)
__temp_preds = paddle.randint(
    low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)
)
__temp_target = abs(__temp_preds - 1)
_input_multilabel_no_match = _Input(preds=__temp_preds, target=__temp_target)
__mc_prob_logits = 10 * paddle.randn(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)
__mc_prob_preds = __mc_prob_logits.abs() / __mc_prob_logits.abs().sum(
    dim=2, keepdim=True
)
_input_multiclass_prob = _Input(
    preds=__mc_prob_preds,
    target=paddle.randint(low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE)),
)
_input_multiclass_logits = _Input(
    preds=__mc_prob_logits,
    target=paddle.randint(low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE)),
)
_input_multiclass = _Input(
    preds=paddle.randint(low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE)),
    target=paddle.randint(low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE)),
)
__mdmc_prob_preds = paddle.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM)
__mdmc_prob_preds = __mdmc_prob_preds / __mdmc_prob_preds.sum(dim=2, keepdim=True)
_input_multidim_multiclass_prob = _Input(
    preds=__mdmc_prob_preds,
    target=paddle.randint(
        low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)
    ),
)
_input_multidim_multiclass = _Input(
    preds=paddle.randint(
        low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)
    ),
    target=paddle.randint(
        low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)
    ),
)


def _generate_plausible_inputs_multilabel(
    num_classes=NUM_CLASSES, num_batches=NUM_BATCHES, batch_size=BATCH_SIZE
):
    correct_targets = paddle.randint(
        low=0, high=num_classes, shape=(num_batches, batch_size)
    )
    preds = paddle.rand(num_batches, batch_size, num_classes)
    targets = paddle.zeros_like(preds, dtype=paddle.long)
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            targets[i, j, correct_targets[i, j]] = 1
    preds += paddle.rand(num_batches, batch_size, num_classes) * targets / 3
    preds = preds / preds.sum(dim=2, keepdim=True)
    return _Input(preds=preds, target=targets)


def _generate_plausible_inputs_binary(num_batches=NUM_BATCHES, batch_size=BATCH_SIZE):
    targets = paddle.randint(low=0, high=2, shape=(num_batches, batch_size))
    preds = (
        paddle.rand(num_batches, batch_size)
        + paddle.rand(num_batches, batch_size) * targets / 3
    )
    return _Input(preds=preds / (preds._max() + 0.01), target=targets)


_input_multilabel_prob_plausible = _generate_plausible_inputs_multilabel()
_input_binary_prob_plausible = _generate_plausible_inputs_binary()
_temp = paddle.randint(low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE))
_class_remove, _class_replace = paddle.multinomial(
    paddle.ones(NUM_CLASSES), num_samples=2, replacement=False
)
_temp[_temp == _class_remove] = _class_replace
_input_multiclass_with_missing_class = _Input(_temp.clone(), _temp.clone())
_negmetric_noneavg = {
    "pred1": paddle.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    "target1": paddle.tensor([0, 1]),
    "res1": paddle.tensor([0.0, 0.0, float("nan")]),
    "pred2": paddle.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    "target2": paddle.tensor([0, 2]),
    "res2": paddle.tensor([0.0, 0.0, 0.0]),
}
