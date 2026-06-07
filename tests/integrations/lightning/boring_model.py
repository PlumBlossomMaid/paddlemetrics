from __future__ import annotations

from importlib import util as importlib_util

import paddle

if importlib_util.find_spec("lightning"):
    pass


class RandomDictStringDataset(paddle.io.Dataset):
    """Class for creating a dictionary of random strings."""

    def __init__(self, size, length) -> None:
        self.len = length
        self.data = paddle.randn(length, size)

    def __getitem__(self, index) -> dict:
        """Get datapoint."""
        return {"id": str(index), "x": self.data[index]}

    def __len__(self) -> int:
        """Return length of dataset."""
        return self.len


class RandomDataset(paddle.io.Dataset):
    """Random dataset for testing PL Module."""

    def __init__(self, size, length) -> None:
        self.len = length
        self.data = paddle.randn(length, size)

    def __getitem__(self, index) -> paddle.Tensor:
        """Get datapoint."""
        return self.data[index]

    def __len__(self) -> int:
        """Get length of dataset."""
        return self.len
