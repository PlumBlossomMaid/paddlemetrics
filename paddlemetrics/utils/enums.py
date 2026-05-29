"""Enum utilities for paddlemetrics."""
import sys
from enum import Enum
from typing import Any, Optional

from typing_extensions import Literal

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    class StrEnum(str, Enum):
        """Backport of StrEnum for Python < 3.11."""

        def __new__(cls, *args: Any, **kwargs: Any) -> "StrEnum":
            value = str(args[0])
            obj = str.__new__(cls, value)
            obj._value_ = value
            return obj

        @classmethod
        def _missing_(cls, value: object) -> Optional["StrEnum"]:
            for member in cls:
                if member.value == value:
                    return member
            return None


class EnumStr(StrEnum):
    """Base Enum."""

    @staticmethod
    def _name() -> str:
        return "Task"

    @classmethod
    def from_str(
        cls: type["EnumStr"], value: str, source: Literal["key", "value", "any"] = "key"
    ) -> "EnumStr":
        """Load from string."""
        try:
            # Try to find by name (key)
            if source in ("key", "any"):
                value_clean = value.replace("-", "_").upper()
                if hasattr(cls, value_clean):
                    return cls[value_clean]
            # Try to find by value
            if source in ("value", "any"):
                for member in cls:
                    if member.value == value:
                        return member
            raise ValueError(f"Invalid value: {value}")
        except (KeyError, ValueError):
            allowed = [m.name.lower() for m in cls]
            raise ValueError(
                f"Invalid {cls._name()}: expected one of {allowed}, but got {value}."
            )


class DataType(EnumStr):
    """Enum to represent data type."""

    @staticmethod
    def _name() -> str:
        return "Data type"

    BINARY = "binary"
    MULTILABEL = "multi-label"
    MULTICLASS = "multi-class"
    MULTIDIM_MULTICLASS = "multi-dim multi-class"


class AverageMethod(EnumStr):
    """Enum to represent average method."""

    @staticmethod
    def _name() -> str:
        return "Average method"

    MICRO = "micro"
    MACRO = "macro"
    WEIGHTED = "weighted"
    NONE = "none"
    SAMPLES = "samples"


class MDMCAverageMethod(EnumStr):
    """Enum to represent multi-dim multi-class average method."""

    @staticmethod
    def _name() -> str:
        return "MDMC Average method"

    GLOBAL = "global"
    SAMPLEWISE = "samplewise"


class ClassificationTask(EnumStr):
    """Enum to represent the different tasks in classification metrics."""

    @staticmethod
    def _name() -> str:
        return "Classification"

    BINARY = "binary"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"


class ClassificationTaskNoBinary(EnumStr):
    """Enum for classification tasks excluding binary."""

    @staticmethod
    def _name() -> str:
        return "Classification"

    MULTILABEL = "multilabel"
    MULTICLASS = "multiclass"


class ClassificationTaskNoMultilabel(EnumStr):
    """Enum for classification tasks excluding multilabel."""

    @staticmethod
    def _name() -> str:
        return "Classification"

    BINARY = "binary"
    MULTICLASS = "multiclass"
