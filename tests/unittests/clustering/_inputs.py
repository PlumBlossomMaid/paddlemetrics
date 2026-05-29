from typing import NamedTuple

import paddle
from paddle import Tensor
from sklearn.datasets import make_blobs
from unittests import BATCH_SIZE, EXTRA_DIM, NUM_BATCHES, NUM_CLASSES, _Input
from unittests._helpers import seed_all

seed_all(42)


class _IntrinsicInput(NamedTuple):
    data: Tensor
    labels: Tensor


def _batch_blobs(num_batches, num_samples, num_features, num_classes):
    data, labels = [], []
    for _ in range(num_batches):
        _data, _labels = make_blobs(num_samples, num_features, centers=num_classes)
        data.append(paddle.tensor(_data))
        labels.append(paddle.tensor(_labels))
    return _IntrinsicInput(data=paddle.stack(data), labels=paddle.stack(labels))


_single_target_extrinsic1 = _Input(
    preds=paddle.randint(low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE)),
    target=paddle.randint(low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE)),
)
_single_target_extrinsic2 = _Input(
    preds=paddle.randint(low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE)),
    target=paddle.randint(low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE)),
)
_float_inputs_extrinsic = _Input(
    preds=paddle.rand((NUM_BATCHES, BATCH_SIZE)),
    target=paddle.rand((NUM_BATCHES, BATCH_SIZE)),
)
_single_target_intrinsic1 = _batch_blobs(
    NUM_BATCHES, BATCH_SIZE, EXTRA_DIM, NUM_CLASSES
)
_single_target_intrinsic2 = _batch_blobs(
    NUM_BATCHES, BATCH_SIZE, EXTRA_DIM, NUM_CLASSES
)
