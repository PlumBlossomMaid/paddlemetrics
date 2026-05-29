import paddle
from paddle import Tensor


def _image_gradients_validate(img: paddle.Tensor) -> None:
    """Validate whether img is a 4D torch Tensor."""
    if not isinstance(img, paddle.Tensor):
        raise TypeError(
            f"The `img` expects a value of <Tensor> type but got {type(img)}"
        )
    if img.ndim != 4:
        raise RuntimeError(f"The `img` expects a 4D tensor but got {img.ndim}D tensor")


def _compute_image_gradients(img: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Compute image gradients (dy/dx) for a given image."""
    batch_size, channels, height, width = img.shape
    dy = img[..., 1:, :] - img[..., :-1, :]
    dx = img[..., :, 1:] - img[..., :, :-1]
    shapey = [batch_size, channels, 1, width]
    dy = paddle.concat(
        [dy, paddle.zeros(shapey, device=img.device, dtype=img.dtype)], axis=2
    )
    dy = dy.view(img.shape)
    shapex = [batch_size, channels, height, 1]
    dx = paddle.concat(
        [dx, paddle.zeros(shapex, device=img.device, dtype=img.dtype)], axis=3
    )
    dx = dx.view(img.shape)
    return dy, dx


def image_gradients(img: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Compute `Gradient Computation of Image`_ of a given image using finite difference.

    Args:
        img: An ``(N, C, H, W)`` input tensor where ``C`` is the number of image channels

    Return:
        Tuple of ``(dy, dx)`` with each gradient of shape ``[N, C, H, W]``

    Raises:
        TypeError:
            If ``img`` is not of the type :class:`~paddle.Tensor`.
        RuntimeError:
            If ``img`` is not a 4D tensor.

    Example:
        >>> from paddlemetrics.functional.image import image_gradients
        >>> image = paddle.arange(0, 1*1*5*5, dtype=paddle.float32)
        >>> image = paddle.reshape(image, (1, 1, 5, 5))
        >>> dy, dx = image_gradients(image)
        >>> dy[0, 0, :, :]
        tensor([[5., 5., 5., 5., 5.],
                [5., 5., 5., 5., 5.],
                [5., 5., 5., 5., 5.],
                [5., 5., 5., 5., 5.],
                [0., 0., 0., 0., 0.]])

    .. note::
           The implementation follows the 1-step finite difference method as followed
           by the TF implementation. The values are organized such that the gradient of
           [I(x+1, y)-[I(x, y)]] are at the (x, y) location

    """
    _image_gradients_validate(img)
    return _compute_image_gradients(img)
