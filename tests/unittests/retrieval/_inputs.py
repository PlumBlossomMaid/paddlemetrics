from typing import NamedTuple

import paddle
from paddle import Tensor
from unittests import BATCH_SIZE, EXTRA_DIM, NUM_BATCHES


class _Input(NamedTuple):
    indexes: Tensor
    preds: Tensor
    target: Tensor


_input_retrieval_scores = _Input(
    indexes=paddle.randint(low=0, high=10, shape=(NUM_BATCHES, BATCH_SIZE)),
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE),
    target=paddle.randint(low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE)),
)
_input_retrieval_scores_for_adaptive_k = _Input(
    indexes=paddle.randint(
        low=0, high=NUM_BATCHES * BATCH_SIZE // 2, shape=(NUM_BATCHES, BATCH_SIZE)
    ),
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE),
    target=paddle.randint(low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE)),
)
_input_retrieval_scores_extra = _Input(
    indexes=paddle.randint(low=0, high=10, shape=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)),
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM),
    target=paddle.randint(low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)),
)
_input_retrieval_scores_int_target = _Input(
    indexes=paddle.randint(low=0, high=10, shape=(NUM_BATCHES, 2 * BATCH_SIZE)),
    preds=paddle.rand(NUM_BATCHES, 2 * BATCH_SIZE),
    target=paddle.randint(low=-1, high=4, shape=(NUM_BATCHES, 2 * BATCH_SIZE)),
)
_input_retrieval_scores_float_target = _Input(
    indexes=paddle.randint(low=0, high=10, shape=(NUM_BATCHES, 2 * BATCH_SIZE)),
    preds=paddle.rand(NUM_BATCHES, 2 * BATCH_SIZE),
    target=paddle.rand(NUM_BATCHES, 2 * BATCH_SIZE),
)
_input_retrieval_scores_with_ignore_index = _Input(
    indexes=paddle.randint(low=0, high=10, shape=(NUM_BATCHES, BATCH_SIZE)),
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE),
    target=paddle.randint(low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE)).masked_fill(
        mask=paddle.randn(NUM_BATCHES, BATCH_SIZE) > 0.5, value=-100
    ),
)
_input_retrieval_scores_no_target = _Input(
    indexes=paddle.randint(low=0, high=10, shape=(NUM_BATCHES, BATCH_SIZE)),
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE),
    target=paddle.randint(low=0, high=1, shape=(NUM_BATCHES, BATCH_SIZE)),
)
_input_retrieval_scores_all_target = _Input(
    indexes=paddle.randint(low=0, high=10, shape=(NUM_BATCHES, BATCH_SIZE)),
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE),
    target=paddle.randint(low=1, high=2, shape=(NUM_BATCHES, BATCH_SIZE)),
)
_input_retrieval_scores_empty = _Input(
    indexes=paddle.randint(low=0, high=10, shape=[0]),
    preds=paddle.rand(0),
    target=paddle.randint(low=0, high=2, shape=[0]),
)
_input_retrieval_scores_mismatching_sizes = _Input(
    indexes=paddle.randint(low=0, high=10, shape=(NUM_BATCHES, BATCH_SIZE - 2)),
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE),
    target=paddle.randint(low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE)),
)
_input_retrieval_scores_mismatching_sizes_func = _Input(
    indexes=paddle.randint(low=0, high=10, shape=(NUM_BATCHES, BATCH_SIZE)),
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE - 2),
    target=paddle.randint(low=0, high=2, shape=(NUM_BATCHES, BATCH_SIZE)),
)
_input_retrieval_scores_wrong_targets = _Input(
    indexes=paddle.randint(low=0, high=10, shape=(NUM_BATCHES, BATCH_SIZE)),
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE),
    target=paddle.randint(
        low=-(2**31), high=2**31, shape=(NUM_BATCHES, BATCH_SIZE)
    ),
)
