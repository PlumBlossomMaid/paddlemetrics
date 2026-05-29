from pathlib import Path

import pytest
from integrations import _PATH_DATASETS


@pytest.fixture(scope="session")
def datadir():
    """Global data dir for location of datasets."""
    return Path(_PATH_DATASETS)


def pytest_configure(config):
    """Local configuration of pytest."""
    config.addinivalue_line(
        "markers",
        "spawn: spawn test in a separate process using paddle.multiprocessing.spawn",
    )
