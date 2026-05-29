from typing import Optional

import paddle
from typing_extensions import Literal

from paddlemetrics.utils.prints import rank_zero_warn


def _nominal_input_validation(
    nan_strategy: str, nan_replace_value: Optional[float]
) -> None:
    if nan_strategy not in ["replace", "drop"]:
        raise ValueError(
            f"Argument `nan_strategy` is expected to be one of `['replace', 'drop']`, but got {nan_strategy}"
        )
    if nan_strategy == "replace" and not isinstance(nan_replace_value, (float, int)):
        raise ValueError(
            f"Argument `nan_replace` is expected to be of a type `int` or `float` when `nan_strategy = 'replace`, but got {nan_replace_value}"
        )


def _compute_expected_freqs(confmat: paddle.Tensor) -> paddle.Tensor:
    """Compute the expected frequenceis from the provided confusion matrix."""
    margin_sum_rows, margin_sum_cols = confmat.sum(1), confmat.sum(0)
    return paddle.einsum("r, c -> rc", margin_sum_rows, margin_sum_cols) / confmat.sum()


def _compute_chi_squared(
    confmat: paddle.Tensor, bias_correction: bool
) -> paddle.Tensor:
    """Chi-square test of independenc of variables in a confusion matrix table.

    Adapted from: https://github.com/scipy/scipy/blob/v1.9.2/scipy/stats/contingency.py.

    """
    expected_freqs = _compute_expected_freqs(confmat)
    df = expected_freqs.size - sum(expected_freqs.shape) + expected_freqs.ndim - 1
    if df == 0:
        return paddle.tensor(0.0, device=confmat.place)
    if df == 1 and bias_correction:
        diff = expected_freqs - confmat
        direction = diff.sign()
        confmat += direction * paddle.minimum(
            0.5 * paddle.ones_like(direction), direction.abs()
        )
    return paddle.sum((confmat - expected_freqs) ** 2 / expected_freqs)


def _drop_empty_rows_and_cols(confmat: paddle.Tensor) -> paddle.Tensor:
    """Drop all rows and columns containing only zeros.

    Example:
        >>> from paddle import randint
        >>> from paddlemetrics.functional.nominal.utils import _drop_empty_rows_and_cols
        >>> matrix = randint(10, size=(4, 3))
        >>> matrix[1, :] = matrix[:, 1] = 0
        >>> matrix
        tensor([[2, 0, 6],
                [0, 0, 0],
                [0, 0, 0],
                [3, 0, 4]])
        >>> _drop_empty_rows_and_cols(matrix)
        tensor([[2, 6],
                [3, 4]])

    """
    confmat = confmat[confmat.sum(1) != 0]
    return confmat[:, confmat.sum(0) != 0]


def _compute_phi_squared_corrected(
    phi_squared: paddle.Tensor, num_rows: int, num_cols: int, confmat_sum: paddle.Tensor
) -> paddle.Tensor:
    """Compute bias-corrected Phi Squared."""
    return paddle.max(
        paddle.tensor(0.0, device=phi_squared.place),
        phi_squared - (num_rows - 1) * (num_cols - 1) / (confmat_sum - 1),
    )


def _compute_rows_and_cols_corrected(
    num_rows: int, num_cols: int, confmat_sum: paddle.Tensor
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Compute bias-corrected number of rows and columns."""
    rows_corrected = num_rows - (num_rows - 1) ** 2 / (confmat_sum - 1)
    cols_corrected = num_cols - (num_cols - 1) ** 2 / (confmat_sum - 1)
    return rows_corrected, cols_corrected


def _compute_bias_corrected_values(
    phi_squared: paddle.Tensor, num_rows: int, num_cols: int, confmat_sum: paddle.Tensor
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Compute bias-corrected Phi Squared and number of rows and columns."""
    phi_squared_corrected = _compute_phi_squared_corrected(
        phi_squared, num_rows, num_cols, confmat_sum
    )
    rows_corrected, cols_corrected = _compute_rows_and_cols_corrected(
        num_rows, num_cols, confmat_sum
    )
    return phi_squared_corrected, rows_corrected, cols_corrected


def _handle_nan_in_data(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    nan_strategy: Literal["replace", "drop"] = "replace",
    nan_replace_value: Optional[float] = 0.0,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Handle ``NaN`` values in input data.

    If ``nan_strategy = 'replace'``, all ``NaN`` values are replaced with ``nan_replace_value``.
    If ``nan_strategy = 'drop'``, all rows containing ``NaN`` in any of two vectors are dropped.

    Args:
        preds: 1D tensor of categorical (nominal) data
        target: 1D tensor of categorical (nominal) data
        nan_strategy: Indication of whether to replace or drop ``NaN`` values
        nan_replace_value: Value to replace ``NaN`s when ``nan_strategy = 'replace```

    Returns:
        Updated ``preds`` and ``target`` tensors which contain no ``Nan``

    Raises:
        ValueError: If ``nan_strategy`` is not from ``['replace', 'drop']``.
        ValueError: If ``nan_strategy = replace`` and ``nan_replace_value`` is not of a type ``int`` or ``float``.

    """
    if nan_strategy == "replace":
        return preds.nan_to_num(nan_replace_value), target.nan_to_num(nan_replace_value)
    rows_contain_nan = paddle.logical_or(preds.isnan(), target.isnan())
    return preds[~rows_contain_nan], target[~rows_contain_nan]


def _unable_to_use_bias_correction_warning(metric_name: str) -> None:
    rank_zero_warn(
        f"Unable to compute {metric_name} using bias correction. Please consider to set `bias_correction=False`."
    )
