import numpy as np

__all__ = ["ArrayLike", "DTypeLike"]

try:
    from numpy.typing import DTypeLike, NDArray

    ArrayLike = NDArray[np.float_]
except (ImportError, TypeError):
    ArrayLike = np.ndarray  # type: ignore[misc]
    DTypeLike = np.dtype  # type: ignore[misc]
