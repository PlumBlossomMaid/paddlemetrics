from collections.abc import Sequence
from typing import Optional, Union

import paddle
from typing_extensions import Literal

from paddlemetrics.functional.image.d_lambda import spectral_distortion_index
from paddlemetrics.functional.image.ergas import \
    error_relative_global_dimensionless_synthesis
from paddlemetrics.functional.image.gradients import image_gradients
from paddlemetrics.functional.image.psnr import peak_signal_noise_ratio
from paddlemetrics.functional.image.rase import relative_average_spectral_error
from paddlemetrics.functional.image.rmse_sw import \
    root_mean_squared_error_using_sliding_window
from paddlemetrics.functional.image.sam import spectral_angle_mapper
from paddlemetrics.functional.image.ssim import (
    multiscale_structural_similarity_index_measure,
    structural_similarity_index_measure)
from paddlemetrics.functional.image.tv import total_variation
from paddlemetrics.functional.image.uqi import universal_image_quality_index
from paddlemetrics.utils.prints import _deprecated_root_import_func


def _spectral_distortion_index(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    p: int = 1,
    reduction: Literal["elementwise_mean", "sum", "none"] = "elementwise_mean",
) -> paddle.Tensor:
    """Wrapper for deprecated import.

    >>> from paddle import rand
    >>> preds = rand([16, 3, 16, 16])
    >>> target = rand([16, 3, 16, 16])
    >>> _spectral_distortion_index(preds, target)
    tensor(0.0234)

    """
    _deprecated_root_import_func("spectral_distortion_index", "image")
    return spectral_distortion_index(
        preds=preds, target=target, p=p, reduction=reduction
    )


def _error_relative_global_dimensionless_synthesis(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    ratio: float = 4,
    reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
) -> paddle.Tensor:
    """Wrapper for deprecated import.

    >>> from paddle import rand
    >>> preds = rand([16, 1, 16, 16])
    >>> target = preds * 0.75
    >>> _error_relative_global_dimensionless_synthesis(preds, target).round()
    tensor(10.)

    """
    _deprecated_root_import_func(
        "error_relative_global_dimensionless_synthesis", "image"
    )
    return error_relative_global_dimensionless_synthesis(
        preds=preds, target=target, ratio=ratio, reduction=reduction
    )


def _image_gradients(img: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Wrapper for deprecated import.

    >>> import paddle
    >>> image = paddle.arange(0, 1*1*5*5, dtype=paddle.float32)
    >>> image = paddle.reshape(image, (1, 1, 5, 5))
    >>> dy, dx = _image_gradients(image)
    >>> dy[0, 0, :, :]
    tensor([[5., 5., 5., 5., 5.],
            [5., 5., 5., 5., 5.],
            [5., 5., 5., 5., 5.],
            [5., 5., 5., 5., 5.],
            [0., 0., 0., 0., 0.]])

    """
    _deprecated_root_import_func("image_gradients", "image")
    return image_gradients(img=img)


def _peak_signal_noise_ratio(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    data_range: Union[float, tuple[float, float]] = 3.0,
    base: float = 10.0,
    reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
    dim: Optional[Union[int, tuple[int, ...]]] = None,
) -> paddle.Tensor:
    """Wrapper for deprecated import.

    >>> from paddle import tensor
    >>> pred = tensor([[0.0, 1.0], [2.0, 3.0]])
    >>> target = tensor([[3.0, 2.0], [1.0, 0.0]])
    >>> _peak_signal_noise_ratio(pred, target)
    tensor(2.5527)

    """
    _deprecated_root_import_func("peak_signal_noise_ratio", "image")
    return peak_signal_noise_ratio(
        preds=preds,
        target=target,
        data_range=data_range,
        base=base,
        reduction=reduction, axis=dim,
    )


def _relative_average_spectral_error(
    preds: paddle.Tensor, target: paddle.Tensor, window_size: int = 8
) -> paddle.Tensor:
    """Wrapper for deprecated import.

    >>> from paddle import rand
    >>> preds = rand(4, 3, 16, 16)
    >>> target = rand(4, 3, 16, 16)
    >>> _relative_average_spectral_error(preds, target)
    tensor(5326.40...)

    """
    _deprecated_root_import_func("relative_average_spectral_error", "image")
    return relative_average_spectral_error(
        preds=preds, target=target, window_size=window_size
    )


def _root_mean_squared_error_using_sliding_window(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    window_size: int = 8,
    return_rmse_map: bool = False,
) -> Union[Optional[paddle.Tensor], tuple[Optional[paddle.Tensor], paddle.Tensor]]:
    """Wrapper for deprecated import.

    >>> from paddle import rand
    >>> preds = rand(4, 3, 16, 16)
    >>> target = rand(4, 3, 16, 16)
    >>> _root_mean_squared_error_using_sliding_window(preds, target)
    tensor(0.4158)

    """
    _deprecated_root_import_func(
        "root_mean_squared_error_using_sliding_window", "image"
    )
    return root_mean_squared_error_using_sliding_window(
        preds=preds,
        target=target,
        window_size=window_size,
        return_rmse_map=return_rmse_map,
    )


def _spectral_angle_mapper(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
) -> paddle.Tensor:
    """Wrapper for deprecated import.

    >>> from paddle import rand
    >>> preds = rand([16, 3, 16, 16])
    >>> target = rand([16, 3, 16, 16])
    >>> _spectral_angle_mapper(preds, target)
    tensor(0.5914)

    """
    _deprecated_root_import_func("spectral_angle_mapper", "image")
    return spectral_angle_mapper(preds=preds, target=target, reduction=reduction)


def _multiscale_structural_similarity_index_measure(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    gaussian_kernel: bool = True,
    sigma: Union[float, Sequence[float]] = 1.5,
    kernel_size: Union[int, Sequence[int]] = 11,
    reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
    data_range: Optional[Union[float, tuple[float, float]]] = None,
    k1: float = 0.01,
    k2: float = 0.03,
    betas: tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
    normalize: Optional[Literal["relu", "simple"]] = "relu",
) -> paddle.Tensor:
    """Wrapper for deprecated import.

    >>> from paddle import rand
    >>> preds = rand([3, 3, 256, 256])
    >>> target = preds * 0.75
    >>> _multiscale_structural_similarity_index_measure(preds, target, data_range=1.0)
    tensor(0.9628)

    """
    _deprecated_root_import_func(
        "multiscale_structural_similarity_index_measure", "image"
    )
    return multiscale_structural_similarity_index_measure(
        preds=preds,
        target=target,
        gaussian_kernel=gaussian_kernel,
        sigma=sigma,
        kernel_size=kernel_size,
        reduction=reduction,
        data_range=data_range,
        k1=k1,
        k2=k2,
        betas=betas,
        normalize=normalize,
    )


def _structural_similarity_index_measure(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    gaussian_kernel: bool = True,
    sigma: Union[float, Sequence[float]] = 1.5,
    kernel_size: Union[int, Sequence[int]] = 11,
    reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
    data_range: Optional[Union[float, tuple[float, float]]] = None,
    k1: float = 0.01,
    k2: float = 0.03,
    return_full_image: bool = False,
    return_contrast_sensitivity: bool = False,
) -> Union[paddle.Tensor, tuple[paddle.Tensor, paddle.Tensor]]:
    """Wrapper for deprecated import.

    >>> import paddle
    >>> preds = paddle.rand([3, 3, 256, 256])
    >>> target = preds * 0.75
    >>> _structural_similarity_index_measure(preds, target)
    tensor(0.9219)

    """
    _deprecated_root_import_func("spectral_angle_mapper", "image")
    return structural_similarity_index_measure(
        preds=preds,
        target=target,
        gaussian_kernel=gaussian_kernel,
        sigma=sigma,
        kernel_size=kernel_size,
        reduction=reduction,
        data_range=data_range,
        k1=k1,
        k2=k2,
        return_full_image=return_full_image,
        return_contrast_sensitivity=return_contrast_sensitivity,
    )


def _total_variation(
    img: paddle.Tensor, reduction: Literal["mean", "sum", "none", None] = "sum"
) -> paddle.Tensor:
    """Wrapper for deprecated import.

    >>> from paddle import rand
    >>> img = rand(5, 3, 28, 28)
    >>> _total_variation(img)
    tensor(7546.8018)

    """
    _deprecated_root_import_func("total_variation", "image")
    return total_variation(img=img, reduction=reduction)


def _universal_image_quality_index(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    kernel_size: Sequence[int] = (11, 11),
    sigma: Sequence[float] = (1.5, 1.5),
    reduction: Optional[
        Literal["elementwise_mean", "sum", "none"]
    ] = "elementwise_mean",
) -> paddle.Tensor:
    """Wrapper for deprecated import.

    >>> import paddle
    >>> preds = paddle.rand([16, 1, 16, 16])
    >>> target = preds * 0.75
    >>> _universal_image_quality_index(preds, target)
    tensor(0.9216)

    """
    _deprecated_root_import_func("universal_image_quality_index", "image")
    return universal_image_quality_index(
        preds=preds,
        target=target,
        kernel_size=kernel_size,
        sigma=sigma,
        reduction=reduction,
    )
