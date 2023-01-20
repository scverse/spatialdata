import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any


# suppress logger debug from ome_zarr with context manager
@contextmanager
def ome_zarr_logger(level: Any) -> Generator[None, None, None]:
    logger = logging.getLogger("ome_zarr")
    current_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(current_level)
