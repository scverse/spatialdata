import pytest

from spatialdata.models.chunks_utils import Chunks_t, normalize_chunks


@pytest.mark.parametrize(
    "chunks, axes, expected",
    [
        # 2D (y, x)
        (256, ("y", "x"), {"y": 256, "x": 256}),
        ((200, 100), ("x", "y"), {"y": 100, "x": 200}),
        ({"y": 300, "x": 400}, ("x", "y"), {"y": 300, "x": 400}),
        # 2D with channel (c, y, x)
        (256, ("c", "y", "x"), {"c": 256, "y": 256, "x": 256}),
        ((1, 100, 200), ("c", "y", "x"), {"c": 1, "y": 100, "x": 200}),
        ({"c": 1, "y": 300, "x": 400}, ("c", "y", "x"), {"c": 1, "y": 300, "x": 400}),
        # 3D (z, y, x)
        ((10, 100, 200), ("z", "y", "x"), {"z": 10, "y": 100, "x": 200}),
        ({"z": 10, "y": 300, "x": 400}, ("z", "y", "x"), {"z": 10, "y": 300, "x": 400}),
        # Mapping with None values (passed through)
        ({"y": None, "x": 400}, ("y", "x"), {"y": None, "x": 400}),
        ({"c": None, "y": None, "x": None}, ("c", "y", "x"), {"c": None, "y": None, "x": None}),
        # Mapping with tuple[int, ...] values (explicit per-block chunk sizes)
        ({"y": (256, 256, 128), "x": 512}, ("y", "x"), {"y": (256, 256, 128), "x": 512}),
        ({"c": 1, "y": (100, 100), "x": (200, 50)}, ("c", "y", "x"), {"c": 1, "y": (100, 100), "x": (200, 50)}),
        # Tuple of tuples (explicit per-block chunk sizes per axis)
        (((256, 256, 128), (512, 512)), ("y", "x"), {"y": (256, 256, 128), "x": (512, 512)}),
    ],
)
def test_normalize_chunks_valid(chunks: Chunks_t, axes: tuple[str, ...], expected: dict[str, int]) -> None:
    assert normalize_chunks(chunks, axes=axes) == expected


@pytest.mark.parametrize(
    "chunks, axes, match",
    [
        ({"y": 100}, ("y", "x"), "missing keys for axes"),
        ((1, 2, 3), ("y", "x"), "doesn't match axes"),
        ((1.5, 2), ("y", "x"), "must be int or tuple"),
        ("invalid", ("y", "x"), "Unsupported chunks type"),
    ],
)
def test_normalize_chunks_errors(chunks: Chunks_t, axes: tuple[str, ...], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        normalize_chunks(chunks, axes=axes)
