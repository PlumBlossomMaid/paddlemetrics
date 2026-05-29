import logging
import os
import warnings
from functools import partial, wraps
from typing import Any, Callable

# Set up logger for paddlemetrics
log = logging.getLogger("paddlemetrics")


def rank_zero_only(fn: Callable) -> Callable:
    """Call a function only on rank 0 in distributed settings."""

    @wraps(fn)
    def wrapped_fn(*args: Any, **kwargs: Any) -> Any:
        if rank_zero_only.rank == 0:
            return fn(*args, **kwargs)
        return None

    return wrapped_fn


rank_zero_only.rank = getattr(
    rank_zero_only, "rank", int(os.environ.get("LOCAL_RANK", 0))
)


def _warn(*args: Any, **kwargs: Any) -> None:
    warnings.warn(*args, **kwargs)


def _info(*args: Any, **kwargs: Any) -> None:
    log.info(*args, **kwargs)


def _debug(*args: Any, **kwargs: Any) -> None:
    log.debug(*args, **kwargs)


rank_zero_debug = rank_zero_only(_debug)
rank_zero_info = rank_zero_only(_info)
rank_zero_warn = rank_zero_only(_warn)
_future_warning = partial(warnings.warn, category=FutureWarning)


def _deprecated_root_import_class(name: str, domain: str) -> None:
    """Warn user that importing class from deprecated location."""
    _future_warning(
        f"Importing `{name}` from `paddlemetrics` was deprecated. Import `{name}` from `paddlemetrics.{domain}` instead."
    )


def _deprecated_root_import_func(name: str, domain: str) -> None:
    """Warn user that importing function from deprecated location."""
    _future_warning(
        f"Importing `{name}` from `paddlemetrics.functional` was deprecated. Import `{name}` from `paddlemetrics.{domain}` instead."
    )
