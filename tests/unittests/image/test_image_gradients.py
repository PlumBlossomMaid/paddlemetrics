import paddle
import pytest

from paddlemetrics.functional import image_gradients


def test_invalid_input_img_type():
    """Test Whether the module successfully handles invalid input data type."""
    invalid_dummy_input = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    with pytest.raises(TypeError):
        image_gradients(invalid_dummy_input)


def test_invalid_input_ndims(batch_size=1, height=5, width=5, channels=1):
    """Test whether the module successfully handles invalid number of dimensions of input tensor."""
    image = paddle.arange(
        0, batch_size * height * width * channels, dtype=paddle.float32
    )
    image = paddle.reshape(image, (height, width))
    with pytest.raises(RuntimeError):
        image_gradients(image)


def test_multi_batch_image_gradients(batch_size=5, height=5, width=5, channels=1):
    """Test whether the module correctly calculates gradients for known input with non-unity batch size."""
    single_channel_img = paddle.arange(
        0, 1 * height * width * channels, dtype=paddle.float32
    )
    single_channel_img = paddle.reshape(single_channel_img, (channels, height, width))
    image = paddle.stack([single_channel_img for _ in range(batch_size)], axis=0)
    true_dy = [
        [5.0, 5.0, 5.0, 5.0, 5.0],
        [5.0, 5.0, 5.0, 5.0, 5.0],
        [5.0, 5.0, 5.0, 5.0, 5.0],
        [5.0, 5.0, 5.0, 5.0, 5.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    true_dy = paddle.Tensor(true_dy)
    dy, dx = image_gradients(image)
    for batch_id in range(batch_size):
        assert paddle.allclose(x=dy[batch_id, 0, :, :], y=true_dy).item()
    assert dy.shape == (batch_size, 1, height, width)
    assert dx.shape == (batch_size, 1, height, width)


def test_image_gradients(batch_size=1, height=5, width=5, channels=1):
    """Test whether the module correctly calculates gradients for known input.

    Example input-output pair taken from TF's implementation of image- gradients

    """
    image = paddle.arange(
        0, batch_size * height * width * channels, dtype=paddle.float32
    )
    image = paddle.reshape(image, (batch_size, channels, height, width))
    true_dy = [
        [5.0, 5.0, 5.0, 5.0, 5.0],
        [5.0, 5.0, 5.0, 5.0, 5.0],
        [5.0, 5.0, 5.0, 5.0, 5.0],
        [5.0, 5.0, 5.0, 5.0, 5.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    true_dx = [
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0],
    ]
    true_dy = paddle.Tensor(true_dy)
    true_dx = paddle.Tensor(true_dx)
    dy, dx = image_gradients(image)
    assert paddle.allclose(x=dy, y=true_dy).item(), "dy fails test"
    assert paddle.allclose(x=dx, y=true_dx).item(), "dx fails tests"
