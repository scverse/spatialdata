from __future__ import annotations

from spatialdata.models import C, X, Y, Z


def _validate_dims(dims: tuple[str, ...]) -> None:
    for c in dims:
        if c not in (X, Y, Z, C):
            raise ValueError(f"Invalid dimension: {c}")
    if dims not in [(X,), (Y,), (Z,), (C,), (X, Y), (X, Y, Z), (Y, X), (Z, Y, X), (C, Y, X), (C, Z, Y, X)]:
        raise ValueError(f"Invalid dimensions: {dims}")
