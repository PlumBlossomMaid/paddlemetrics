import paddle

__all__ = ["_Input"]
from unittests import BATCH_SIZE, NUM_BATCHES, NUM_CLASSES, _Input
from unittests._helpers import seed_all

seed_all(42)
to_one_hot = lambda x: paddle.nn.functional.one_hot(x, NUM_CLASSES).permute(
    0, 1, 4, 2, 3
)
_one_hot_input_1 = _Input(
    preds=to_one_hot(
        paddle.randint(low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE, 16, 16))
    ),
    target=to_one_hot(
        paddle.randint(low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE, 16, 16))
    ),
)
_one_hot_input_2 = _Input(
    preds=to_one_hot(
        paddle.randint(low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE, 32, 32))
    ),
    target=to_one_hot(
        paddle.randint(low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE, 32, 32))
    ),
)
_index_input_1 = _Input(
    preds=paddle.randint(
        low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE, 32, 32)
    ),
    target=paddle.randint(
        low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE, 32, 32)
    ),
)
_index_input_2 = _Input(
    preds=paddle.randint(low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE, 32)),
    target=paddle.randint(low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE, 32)),
)
_mixed_input_1 = _Input(
    preds=to_one_hot(
        paddle.randint(low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE, 32, 32))
    ),
    target=paddle.randint(
        low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE, 32, 32)
    ),
)
_mixed_input_2 = _Input(
    preds=paddle.randint(
        low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE, 32, 32)
    ),
    target=to_one_hot(
        paddle.randint(low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE, 32, 32))
    ),
)
_mixed_logits_input = _Input(
    preds=paddle.rand((NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, 32, 32)) * 12 - 6,
    target=paddle.randint(
        low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE, 32, 32)
    ),
)
