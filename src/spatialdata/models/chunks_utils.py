from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

Chunks_t: TypeAlias = int | tuple[int, ...] | tuple[tuple[int, ...], ...] | Mapping[Any, None | int | tuple[int, ...]]


def normalize_chunks(
    chunks: Chunks_t,
    axes: Sequence[str],
) -> dict[str, None | int | tuple[int, ...]]:
    """Normalize chunk specification to dict format.

    This function converts various chunk formats to a dict mapping dimension names
    to chunk sizes. The dict format is preferred because it's explicit about which
    dimension gets which chunk size.

    Parameters
    ----------
    chunks
        Chunk specification. Can be:
        - int: Applied to all axes
        - tuple[int, ...]: Chunk sizes in order corresponding to axes
        - tuple[tuple[int, ...], ...]: Explicit per-block chunk sizes per axis
        - dict: Mapping of axis names to chunk sizes. Values can be:
            - int: uniform chunk size for that axis
            - tuple[int, ...]: explicit per-block chunk sizes
            - None: keep existing chunks (or use full dimension when no chunks were available)
    axes
        Tuple of axis names that defines the expected dimensions (e.g., ('c', 'y', 'x')).

    Returns
    -------
    dict[str, None | int | tuple[int, ...]]
        Dict mapping axis names to chunk sizes. ``None`` values are preserved
        with dask semantics (keep existing chunks, or use full dimension size if chunks
        where not available and are being created).

    Raises
    ------
    ValueError
        If chunks format is not supported or incompatible with axes.
    """
    if isinstance(chunks, int):
        return dict.fromkeys(axes, chunks)

    if isinstance(chunks, Mapping):
        chunks_dict = dict(chunks)
        missing = set(axes) - set(chunks_dict.keys())
        if missing:
            raise ValueError(f"chunks dict missing keys for axes {missing}, got: {list(chunks_dict.keys())}")
        return {ax: chunks_dict[ax] for ax in axes}

    if isinstance(chunks, tuple):
        if len(chunks) != len(axes):
            raise ValueError(f"chunks tuple length {len(chunks)} doesn't match axes {axes} (length {len(axes)})")
        if not all(isinstance(c, (int, tuple)) for c in chunks):
            raise ValueError(f"All elements in chunks tuple must be int or tuple[int, ...], got: {chunks}")
        return dict(zip(axes, chunks, strict=True))  # type: ignore[arg-type]

    raise ValueError(f"Unsupported chunks type: {type(chunks)}. Expected int, tuple, dict, or None.")
