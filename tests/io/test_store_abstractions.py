"""Abstraction stress tests for ``SpatialData`` io against a memory-backed ``UPath``.

These tests exercise the same read/write code paths that would be hit by a real remote
backend (S3/Azure/GCS/HTTPS), using only ``fsspec.filesystem("memory")`` and a thin
no-listing wrapper to approximate HTTP-like semantics. No emulators, no network.

The file is deliberately scoped to the **public interface** (``SpatialData.read`` /
``SpatialData.write``) plus tamper-evident inspection of the underlying fsspec backend;
the lower-level ``ZarrStore`` / ``_resolve_zarr_store`` plumbing is unit-tested separately
in ``tests/io/test_store.py``.

Coverage goals (generic, not provider-specific):
- ``SpatialData.read`` does not mutate backend bytes (tamper-evident snapshot equality).
- Full write / write-read-write round-trip through a remote-backed ``UPath`` for images,
  labels, shapes, points, and a full sdata. The write-read-write cycle specifically pins
  the categorical-schema invariant that the arrow-filesystem migration (this PR) had to
  re-establish in ``_read_points``.
- Writing to a ``UPath`` lands the root metadata artifact in the backend. The read-time
  consumption of consolidated metadata is an xfail placeholder for the cloud-native
  follow-up.
- A ``MemoryFileSystem`` subclass that refuses listing proves that ``SpatialData.read``
  does not depend on directory listing for basic elements (the precondition for serving
  public HTTPS zarrs).

These tests are strictly stronger than moto/s3 emulator coverage: they need no external
process, no subprocess, no network, and they pin the exact abstraction boundary that the
cloud-native follow-up must not regress.
"""

from __future__ import annotations

import pytest
from fsspec.implementations.memory import MemoryFileSystem
from upath import UPath

from spatialdata import SpatialData
from spatialdata.testing import assert_spatial_data_objects_are_identical


def _fresh_memory_upath(key: str) -> UPath:
    """Build a UPath bound to a fresh (per-test) in-memory fsspec filesystem.

    ``skip_instance_cache=True`` ensures every test gets an isolated memory backend so
    tests cannot leak state across each other.
    """
    fs = MemoryFileSystem(skip_instance_cache=True)
    return UPath(f"memory://{key}.zarr", fs=fs)


# ---------------------------------------------------------------------------
# SpatialData.read is side-effect-free against the backend.
# ---------------------------------------------------------------------------


class TestReadIsSideEffectFree:
    """``SpatialData.read`` must not mutate a single byte of the backend store.

    Using a memory filesystem as a tamper-evident substrate, we snapshot every key+bytes
    before and after the read and assert full equality. This is strictly a public-interface
    invariant: if ``read_zarr`` (or any element reader) ever silently wrote to a remote
    backend, this test is the first to catch it. The lower-level guarantee that
    ``_resolve_zarr_store`` forwards ``read_only=True`` to the backend store is unit-tested
    separately in ``tests/io/test_store.py``.
    """

    def test_spatialdata_read_does_not_mutate_backend(self, images: SpatialData) -> None:
        upath = _fresh_memory_upath("read-only-invariant")
        images.write(upath, overwrite=True)

        fs = upath.fs

        def snapshot() -> dict[str, bytes]:
            return {key: fs.cat_file(key) for key in fs.find(upath.path)}

        before = snapshot()
        SpatialData.read(upath)
        after = snapshot()

        assert before.keys() == after.keys(), (
            f"read added/removed backend keys; added={after.keys() - before.keys()}, "
            f"removed={before.keys() - after.keys()}"
        )
        # Equality on bytes (not just on keys) is what makes this tamper-evident: even a
        # same-size rewrite of the same key would be caught.
        assert before == after, "read mutated bytes in the backend store"


# ---------------------------------------------------------------------------
# Full SpatialData round-trip through a memory-backed UPath: the generic
# remote-backend stress test.
# ---------------------------------------------------------------------------


class TestMemoryUPathRoundtrip:
    """Round-trip ``SpatialData`` objects through a memory-backed ``UPath``.

    Every code path from ``make_zarr_store`` -> ``_resolve_zarr_store`` ->
    ``open_write_store`` / ``open_read_store`` -> ``zarr.open_group(FsspecStore)`` ->
    ``io_raster`` / ``io_shapes`` / ``io_points`` / ``io_table`` is exercised identically
    to how it would be against S3/Azure/GCS. If any of these regresses for remote backends,
    one of these tests must break.

    Note that ``overwrite=True`` is required on every ``write()`` call that targets a
    ``UPath`` (per the guard in ``_validate_can_safely_write_to_path``): remote existence
    checks are unreliable across fsspec backends, so the caller must explicitly opt in.
    """

    def test_roundtrip_images_only(self, images: SpatialData) -> None:
        upath = _fresh_memory_upath("images")
        images.write(upath, overwrite=True)
        read = SpatialData.read(upath)
        assert_spatial_data_objects_are_identical(images, read)

    def test_roundtrip_labels_only(self, labels: SpatialData) -> None:
        upath = _fresh_memory_upath("labels")
        labels.write(upath, overwrite=True)
        read = SpatialData.read(upath)
        assert_spatial_data_objects_are_identical(labels, read)

    def test_roundtrip_shapes_only(self, shapes: SpatialData) -> None:
        upath = _fresh_memory_upath("shapes")
        shapes.write(upath, overwrite=True)
        read = SpatialData.read(upath)
        assert_spatial_data_objects_are_identical(shapes, read)

    def test_roundtrip_points_only(self, points: SpatialData) -> None:
        upath = _fresh_memory_upath("points")
        points.write(upath, overwrite=True)
        read = SpatialData.read(upath)
        assert_spatial_data_objects_are_identical(points, read)

    def test_write_read_write_points_preserves_categorical_schema(
        self, points: SpatialData
    ) -> None:
        """Regression guard for the arrow-filesystem categorical round-trip.

        This PR migrated points io to ``to_parquet`` / ``read_parquet`` with
        ``filesystem=arrow_fs``. ``read_parquet(filesystem=arrow_fs)`` eagerly pandas-ifies
        pyarrow dictionaries into ``CategoricalDtype`` marked ``known=True`` with an empty
        category list -- that would defeat ``write_points``'s ``as_known()`` normalization
        and a subsequent ``to_parquet(filesystem=arrow_fs)`` would fail with a per-partition
        schema mismatch (``dictionary<values=null>`` vs ``dictionary<values=string>``). The
        fix lives in ``_read_points`` (demote such categoricals to unknown so that
        ``write_points`` recomputes categories across partitions); this test pins it.
        """
        upath1 = _fresh_memory_upath("points-rt1")
        upath2 = _fresh_memory_upath("points-rt2")
        points.write(upath1, overwrite=True)
        read = SpatialData.read(upath1)
        read.write(upath2, overwrite=True)
        round_tripped = SpatialData.read(upath2)
        assert_spatial_data_objects_are_identical(points, round_tripped)

    def test_write_read_write_full_sdata(self, full_sdata: SpatialData) -> None:
        """End-to-end guard: a full sdata round-trips write -> read -> write cleanly.

        Pinned for the same reason as the points-only variant above: the arrow-filesystem
        migration in this PR had to re-establish the categorical-schema invariant on the
        read side so that write does not fail on the second pass.
        """
        upath1 = _fresh_memory_upath("full-rt1")
        upath2 = _fresh_memory_upath("full-rt2")
        full_sdata.write(upath1, overwrite=True)
        read = SpatialData.read(upath1)
        read.write(upath2, overwrite=True)
        round_tripped = SpatialData.read(upath2)
        assert_spatial_data_objects_are_identical(full_sdata, round_tripped)

    def test_roundtrip_full_sdata(self, full_sdata: SpatialData) -> None:
        upath = _fresh_memory_upath("full")
        full_sdata.write(upath, overwrite=True)
        read = SpatialData.read(upath)
        assert_spatial_data_objects_are_identical(full_sdata, read)


# ---------------------------------------------------------------------------
# Consolidated metadata on read.
# ---------------------------------------------------------------------------


class TestConsolidatedMetadataOnRead:
    """Writing produces a consolidated-metadata artifact; the read path does not consume it yet.

    The follow-up cloud-native PR will thread ``use_consolidated=True`` through
    ``open_read_store`` / ``read_zarr``. When that lands, the xfail here flips to a pass
    and the assertion becomes strict.
    """

    def test_write_produces_root_metadata_on_memory_upath(self, images: SpatialData) -> None:
        upath = _fresh_memory_upath("consolidated")
        images.write(upath, overwrite=True)
        fs = upath.fs
        # The root metadata artifact differs by zarr version: zarr v3 writes ``zarr.json``
        # at every group, zarr v2 writes ``.zmetadata`` at the consolidated root. Accepting
        # either keeps the test valid across versions and asserts that the write path
        # actually reaches the memory backend.
        root_keys = [p.rsplit("/", 1)[-1] for p in fs.find(upath.path)]
        assert "zarr.json" in root_keys or ".zmetadata" in root_keys, root_keys

    @pytest.mark.xfail(
        reason=(
            "read_zarr opens the root group with zarr.open_group(store, mode='r') without "
            "use_consolidated=True, so a consolidated metadata artifact is ignored on remote "
            "reads. The cloud-native follow-up will thread use_consolidated through open_read_store."
        ),
        strict=True,
    )
    def test_read_zarr_opens_via_consolidated_metadata(self, images: SpatialData) -> None:
        upath = _fresh_memory_upath("consolidated-read")
        images.write(upath, overwrite=True)

        # Count store GETs on the memory fs to detect that consolidated metadata is used:
        # without consolidation, reading one image requires many small zarr.json / .zgroup GETs.
        fs = upath.fs
        original_cat_file = fs._cat_file
        call_count = {"n": 0}

        def counting_cat_file(path, *args, **kwargs):
            call_count["n"] += 1
            return original_cat_file(path, *args, **kwargs)

        fs._cat_file = counting_cat_file
        try:
            SpatialData.read(upath)
        finally:
            fs._cat_file = original_cat_file

        # With consolidated metadata, we expect very few small-metadata GETs for a
        # trivial 1-image sdata. Without it, typical count is >> 10. The exact bound is
        # a documented, loose sanity check, not a micro-benchmark.
        assert call_count["n"] < 10, (
            f"expected consolidated metadata to reduce GETs, saw {call_count['n']}"
        )


# ---------------------------------------------------------------------------
# HTTP-like read-only filesystem: simulates a remote that does not support listing.
# ---------------------------------------------------------------------------


class _NoListMemoryFileSystem(MemoryFileSystem):
    """MemoryFileSystem that refuses directory listing, approximating HTTPS zarr semantics.

    Public HTTPS zarr reads cannot do ``ls`` / ``find`` on an arbitrary prefix; they can
    only GET known keys. This wrapper fails any listing operation so we can prove that
    our read path does not rely on listing -- the precondition for public HTTPS datasets
    to be readable.
    """

    def _ls(self, path, detail=True, **kwargs):  # type: ignore[override]
        raise NotImplementedError("listing disabled to simulate HTTP-like semantics")

    def ls(self, path, detail=True, **kwargs):  # type: ignore[override]
        raise NotImplementedError("listing disabled to simulate HTTP-like semantics")

    def find(self, path, **kwargs):  # type: ignore[override]
        raise NotImplementedError("listing disabled to simulate HTTP-like semantics")


class TestHttpLikeReadOnlyStore:
    """Approximate HTTPS zarr semantics: a read-only filesystem that refuses listing.

    The point is not to re-test zarr's FsspecStore but to catch the case where our own
    ``read_zarr`` implementation (or an element reader) assumes it can list a directory.
    That is exactly the pattern that breaks when pointed at a real public HTTPS zarr.
    """

    def test_read_sdata_from_no_list_fs(self, images: SpatialData, tmp_path) -> None:
        # Write locally, then copy bytes into a no-list memory fs so that the backend
        # resembles a public HTTPS zarr: every known key is readable but listing is disabled.
        local_path = tmp_path / "local.zarr"
        images.write(local_path)

        no_list_fs = _NoListMemoryFileSystem(skip_instance_cache=True)
        remote_root = "no-list.zarr"
        for p in local_path.rglob("*"):
            if p.is_file():
                rel = p.relative_to(local_path).as_posix()
                no_list_fs.pipe_file(f"{remote_root}/{rel}", p.read_bytes())

        upath = UPath(f"memory://{remote_root}", fs=no_list_fs)
        read = SpatialData.read(upath)
        assert_spatial_data_objects_are_identical(images, read)
