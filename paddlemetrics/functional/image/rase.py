import paddle

from paddlemetrics.functional.image.rmse_sw import (_rmse_sw_compute,
                                                   _rmse_sw_update)
from paddlemetrics.functional.image.utils import _uniform_filter


def _rase_update(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    window_size: int,
    rmse_map: paddle.Tensor,
    target_sum: paddle.Tensor,
    total_images: paddle.Tensor,
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Calculate the sum of RMSE map values for the batch of examples and update intermediate states.

    Args:
        preds: Deformed image
        target: Ground truth image
        window_size: Sliding window used for RMSE calculation
        rmse_map: Sum of RMSE map values over all examples
        target_sum: target...
        total_images: Total number of images

    Return:
        Intermediate state of RMSE map
        Updated total number of already processed images

    """
    _, rmse_map, total_images = _rmse_sw_update(
        preds,
        target,
        window_size,
        rmse_val_sum=None,
        rmse_map=rmse_map,
        total_images=total_images,
    )
    target_sum += paddle.sum(
        _uniform_filter(target, window_size) / window_size**2, axis=0
    )
    return rmse_map, target_sum, total_images


def _rase_compute(
    rmse_map: paddle.Tensor,
    target_sum: paddle.Tensor,
    total_images: paddle.Tensor,
    window_size: int,
) -> paddle.Tensor:
    """Compute RASE.

    Args:
        rmse_map: Sum of RMSE map values over all examples
        target_sum: target...
        total_images: Total number of images.
        window_size: Sliding window used for rmse calculation

    Return:
        Relative Average Spectral Error (RASE)

    """
    _, rmse_map = _rmse_sw_compute(
        rmse_val_sum=None, rmse_map=rmse_map, total_images=total_images
    )
    target_mean = target_sum / total_images
    target_mean = target_mean.mean(0)
    rase_map = 100 / target_mean * paddle.sqrt(paddle.mean(rmse_map**2, 0))
    crop_slide = round(window_size / 2)
    return paddle.mean(rase_map[crop_slide:-crop_slide, crop_slide:-crop_slide])


def relative_average_spectral_error(
    preds: paddle.Tensor, target: paddle.Tensor, window_size: int = 8
) -> paddle.Tensor:
    """Compute Relative Average Spectral Error (RASE) (RelativeAverageSpectralError_).

    Args:
        preds: Deformed image
        target: Ground truth image
        window_size: Sliding window used for rmse calculation

    Return:
        Relative Average Spectral Error (RASE)

    Example:
        >>> from paddle import rand
        >>> from paddlemetrics.functional.image import relative_average_spectral_error
        >>> preds = rand(4, 3, 16, 16)
        >>> target = rand(4, 3, 16, 16)
        >>> relative_average_spectral_error(preds, target)
        tensor(5326.40...)

    Raises:
        ValueError: If ``window_size`` is not a positive integer.

    """
    if (
        not isinstance(window_size, int)
        or isinstance(window_size, int)
        and window_size < 1
    ):
        raise ValueError("Argument `window_size` is expected to be a positive integer.")
    img_shape = target.shape[1:]
    rmse_map = paddle.zeros(img_shape, dtype=target.dtype, device=target.place)
    target_sum = paddle.zeros(img_shape, dtype=target.dtype, device=target.place)
    total_images = paddle.tensor(0.0, device=target.place)
    rmse_map, target_sum, total_images = _rase_update(
        preds, target, window_size, rmse_map, target_sum, total_images
    )
    return _rase_compute(rmse_map, target_sum, total_images, window_size)
