import itertools
import random
import paddle
from compat import pack_padded_sequence, pad_packed_sequence


def dim2perm(ndim, dim0, dim1):
    perm = list(range(ndim))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return perm


def test_pack_padded_sequence():
    def generate_test_case(sorted_lengths, should_shuffle):
        def pad(tensor, length):
            return paddle.concat(
                [
                    tensor,
                    paddle.zeros([length - tensor.size(0), *tensor.size()[1:]]),
                ]
            )

        max_length = sorted_lengths[0]
        batch_sizes = [
            sum(map(bool, filter(lambda x: x >= i, sorted_lengths)))
            for i in range(1, max_length + 1)
        ]
        padded = paddle.concat(
            [
                pad(i * 100 + paddle.arange(1.0, 5 * l + 1).view(l, 1, 5), max_length)
                for i, l in enumerate(sorted_lengths, 1)
            ],
            1,
        )
        expected_data = [
            [(paddle.arange(1.0, 6) + (i + 1) * 100 + 5 * n) for i in range(batch_size)]
            for n, batch_size in enumerate(batch_sizes)
        ]
        expected_data = list(itertools.chain.from_iterable(expected_data))
        expected_data = paddle.stack(expected_data, axis=0)
        if should_shuffle:
            permutation = list(range(len(sorted_lengths)))
            random.shuffle(permutation)
            unsorted_indices = paddle.tensor(permutation)
            padded = padded.index_select(1, unsorted_indices)
            lengths = paddle.tensor(sorted_lengths).index_select(0, unsorted_indices)
        else:
            unsorted_indices = None
            lengths = sorted_lengths
        return (
            padded.requires_grad_(),
            lengths,
            expected_data,
            batch_sizes,
            unsorted_indices,
        )

    test_cases = [
        [[10, 8, 4, 2, 2, 2, 1], False],
        [[11, 10, 8, 6, 4, 3, 1], False],
        [[11, 10, 8, 6, 4, 3, 1]],
    ]
    for test_case, batch_first in itertools.product(test_cases, (True, False)):
        sorted_lengths, should_shuffle = test_case
        (
            padded,
            lengths,
            expected_data,
            batch_sizes,
            unsorted_indices,
        ) = generate_test_case(sorted_lengths, should_shuffle)
        src = padded
        if batch_first:
            src = src.transpose(0, 1)
        packed = pack_padded_sequence(
            src, lengths, batch_first=batch_first, enforce_sorted=not should_shuffle
        )
        packed_data, batch_sizes, sorted_indices, unsorted_indices = packed
        assert paddle.allclose(packed_data.data, expected_data), packed.data.data - expected_data
        assert paddle.allclose(batch_sizes, batch_sizes)
        assert paddle.allclose(unsorted_indices, unsorted_indices)
        unpacked, unpacked_len = pad_packed_sequence(packed, batch_first=batch_first)
        assert paddle.allclose(unpacked, src)
        #lengths = paddle.to_tensor(lengths)
        #assert paddle.allclose(unpacked_len, lengths)
        if padded.grad is not None:
            padded.grad.data.zero_()
        grad_output = unpacked.data.clone().normal_()
        unpacked.backward(grad_tensor=grad_output)
        if batch_first:
            grad_output.transpose_(perm=dim2perm(grad_output.ndim, 0, 1))
        for i, l in enumerate(lengths):
            assert paddle.allclose(padded.grad.data[:l, i], grad_output[:l, i])
            if l < 10:
                assert paddle.allclose(padded.grad.data[l:, i].abs().sum(), paddle.to_tensor(0).astype(padded.grad.data.dtype))

    
    try:
        packed = pack_padded_sequence(paddle.randn([3, 3]), [1, 3, 2])
    except ValueError:
        pass
    try:
        packed = pack_padded_sequence(paddle.randn([0, 0]), [])
    except ValueError:
        pass
    try:
        packed = pack_padded_sequence(
            paddle.randn([0, 1, 10]), paddle.randn([11, 14, 14, 2])
        )
    except ValueError:
        pass


if __name__ == "__main__":
    test_pack_padded_sequence()
    print("done")