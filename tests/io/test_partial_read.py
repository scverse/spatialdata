from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import TYPE_CHECKING

import anndata
import py
import pytest
from pyarrow import ArrowInvalid
from zarr.errors import ArrayNotFoundError, ZarrUserWarning

from spatialdata import SpatialData, read_zarr
from spatialdata._io.format import SpatialDataContainerFormatV01
from spatialdata.datasets import blobs

if TYPE_CHECKING:
    import _pytest.fixtures


@contextmanager
def pytest_warns_multiple(
    expected_warning: type[Warning] | tuple[type[Warning], ...] = Warning, matches: Iterable[str] = ()
) -> Generator[None, None, None]:
    """
    Assert that code raises warnings matching particular patterns.

    Like `pytest.warns`, but with multiple patterns which each must match a warning.

    Parameters
    ----------
    expected_warning
        A warning class or a tuple of warning classes for which at least one matching warning must be found
    matches
        Regular expression patterns that of which each must be found in at least one warning message.
    """
    if not matches:
        yield
    else:
        with (
            pytest.warns(expected_warning, match=matches[0]),
            pytest_warns_multiple(expected_warning, matches=matches[1:]),
        ):
            yield


@pytest.fixture(scope="module")
def test_case(request: _pytest.fixtures.SubRequest):
    """
    Fixture that helps to use fixtures as arguments in parametrize.

    The fixture `test_case` takes up values from other fixture functions used as parameters.
    """
    fixture_function = request.param
    fixture_name = fixture_function.__name__
    return request.getfixturevalue(fixture_name)


@dataclass
class PartialReadTestCase:
    path: Path
    expected_elements: list[str]
    expected_exceptions: type[Exception] | tuple[type[Exception] | IOError, ...]
    warnings_patterns: list[str]


@pytest.fixture(scope="session")
def session_tmp_path(request: _pytest.fixtures.SubRequest) -> Path:
    """
    Create a temporary directory as a fixture with session scope and deletes it afterward.

    The default tmp_path fixture has function scope and cannot be used as input to session-scoped
    fixtures.
    """
    directory = py.path.local(tempfile.mkdtemp())
    request.addfinalizer(lambda: directory.remove(rec=1))
    return Path(directory)


@pytest.fixture(scope="module")
def sdata_with_corrupted_elem_types_zgroup(session_tmp_path: Path) -> PartialReadTestCase:
    # Zarr v2
    sdata = blobs()
    sdata_path = session_tmp_path / "sdata_with_corrupted_top_level_zgroup.zarr"
    # Errors only when no consolidation metadata store is used as this takes precedence over group metadata when reading
    sdata.write(sdata_path, sdata_formats=SpatialDataContainerFormatV01(), consolidate_metadata=False)

    (sdata_path / "images" / ".zgroup").unlink()  # missing, not detected by reader. So it doesn't raise an exception,
    # but it will not be found in the read SpatialData object
    (sdata_path / "labels" / ".zgroup").write_text("")  # corrupted
    (sdata_path / "points" / ".zgroup").write_text("{}")  # invalid
    not_corrupted = [name for t, name, _ in sdata.gen_elements() if t not in ("images", "labels", "points")]

    return PartialReadTestCase(
        path=sdata_path,
        expected_elements=not_corrupted,
        expected_exceptions=(JSONDecodeError, ZarrUserWarning),
        warnings_patterns=["labels: JSONDecodeError", "Object at"],
    )


@pytest.fixture(scope="module")
def sdata_with_corrupted_elem_types_zarr_json(session_tmp_path: Path) -> PartialReadTestCase:
    # Zarr v3
    sdata = blobs()
    sdata_path = session_tmp_path / "sdata_with_corrupted_top_level_zarr_json.zarr"
    # Errors only when no consolidation metadata store is used as this takes precedence over group metadata when reading
    sdata.write(sdata_path, consolidate_metadata=False)

    (sdata_path / "images" / "zarr.json").unlink()  # missing, not detected by reader. So it doesn't raise an exception,
    # but it will not be found in the read SpatialData object
    (sdata_path / "labels" / "zarr.json").write_text("")  # corrupted
    (sdata_path / "points" / "zarr.json").write_text('"not_valid": "not_valid"}')  # invalid
    not_corrupted = [name for t, name, _ in sdata.gen_elements() if t not in ("images", "labels", "points")]

    return PartialReadTestCase(
        path=sdata_path,
        expected_elements=not_corrupted,
        expected_exceptions=(JSONDecodeError),
        warnings_patterns=["labels: JSONDecodeError", "Extra data"],
    )


@pytest.fixture(scope="module")
def sdata_with_corrupted_zarr_json_elements(session_tmp_path: Path) -> PartialReadTestCase:
    # Zarr v3
    # zarr.json is a zero-byte file, aborted during write, or contains invalid JSON syntax
    sdata = blobs()
    sdata_path = session_tmp_path / "sdata_with_corrupted_zarr_json_elements.zarr"
    sdata.write(sdata_path)

    corrupted_elements = ["blobs_image", "blobs_labels", "blobs_points", "blobs_polygons", "table"]
    warnings_patterns = []
    for corrupted_element in corrupted_elements:
        elem_path = sdata.locate_element(sdata[corrupted_element])[0]
        (sdata_path / elem_path / "zarr.json").write_bytes(b"")
        warnings_patterns.append(rf"{elem_path}: (?:OSError|JSONDecodeError):")
    not_corrupted = [name for _, name, _ in sdata.gen_elements() if name not in corrupted_elements]

    return PartialReadTestCase(
        path=sdata_path,
        expected_elements=not_corrupted,
        expected_exceptions=(JSONDecodeError, OSError),
        warnings_patterns=warnings_patterns,
    )


@pytest.fixture(scope="module")
def sdata_with_corrupted_zattrs_elements(session_tmp_path: Path) -> PartialReadTestCase:
    # Zarr v2
    # .zattrs is a zero-byte file, aborted during write, or contains invalid JSON syntax
    sdata = blobs()
    sdata_path = session_tmp_path / "sdata_with_corrupted_zattrs_elements.zarr"
    sdata.write(sdata_path, sdata_formats=SpatialDataContainerFormatV01())

    corrupted_elements = ["blobs_image", "blobs_labels", "blobs_points", "blobs_polygons", "table"]
    warnings_patterns = []
    for corrupted_element in corrupted_elements:
        elem_path = sdata.locate_element(sdata[corrupted_element])[0]
        (sdata_path / elem_path / ".zattrs").write_bytes(b"")
        warnings_patterns.append(rf"{elem_path}: (?:OSError|JSONDecodeError):")
    not_corrupted = [name for _, name, _ in sdata.gen_elements() if name not in corrupted_elements]

    return PartialReadTestCase(
        path=sdata_path,
        expected_elements=not_corrupted,
        expected_exceptions=(OSError, JSONDecodeError),
        warnings_patterns=warnings_patterns,
    )


@pytest.fixture(scope="module")
def sdata_with_corrupted_image_chunks_zarrv3(session_tmp_path: Path) -> PartialReadTestCase:
    # images/blobs_image/0 is a zero-byte file or aborted during write
    sdata = blobs()
    sdata_path = session_tmp_path / "sdata_with_corrupted_image_chunks_zarrv3.zarr"
    sdata.write(sdata_path)

    corrupted = "blobs_image"
    os.unlink(sdata_path / "images" / corrupted / "0" / "zarr.json")  # it will hide the "0" array from the Zarr reader
    os.rename(sdata_path / "images" / corrupted / "0", sdata_path / "images" / corrupted / "0_corrupted")
    (sdata_path / "images" / corrupted / "0").touch()

    not_corrupted = [name for _, name, _ in sdata.gen_elements() if name != corrupted]

    return PartialReadTestCase(
        path=sdata_path,
        expected_elements=not_corrupted,
        expected_exceptions=(ArrayNotFoundError,),
        warnings_patterns=[rf"images/{corrupted}: ArrayNotFoundError"],
    )


@pytest.fixture(scope="module")
def sdata_with_corrupted_image_chunks_zarrv2(session_tmp_path: Path) -> PartialReadTestCase:
    # images/blobs_image/0 is a zero-byte file or aborted during write
    sdata = blobs()
    sdata_path = session_tmp_path / "sdata_with_corrupted_image_chunks_zarrv2.zarr"
    sdata.write(sdata_path, sdata_formats=SpatialDataContainerFormatV01())

    corrupted = "blobs_image"
    os.unlink(sdata_path / "images" / corrupted / "0" / ".zarray")  # it will hide the "0" array from the Zarr reader
    os.rename(sdata_path / "images" / corrupted / "0", sdata_path / "images" / corrupted / "0_corrupted")
    (sdata_path / "images" / corrupted / "0").touch()
    not_corrupted = [name for _, name, _ in sdata.gen_elements() if name != corrupted]

    return PartialReadTestCase(
        path=sdata_path,
        expected_elements=not_corrupted,
        expected_exceptions=(ArrayNotFoundError,),
        warnings_patterns=[rf"images/{corrupted}: ArrayNotFoundError"],
    )


@pytest.fixture(scope="module")
def sdata_with_corrupted_parquet_zarrv3(session_tmp_path: Path) -> PartialReadTestCase:
    # points/blobs_points/0 is a zero-byte file or aborted during write
    sdata = blobs()
    sdata_path = session_tmp_path / "sdata_with_corrupted_parquet_zarrv3.zarr"
    sdata.write(sdata_path)

    corrupted = "blobs_points"
    os.rename(
        sdata_path / "points" / corrupted / "points.parquet",
        sdata_path / "points" / corrupted / "points_corrupted.parquet",
    )
    (sdata_path / "points" / corrupted / "points.parquet").touch()

    not_corrupted = [name for _, name, _ in sdata.gen_elements() if name != corrupted]

    return PartialReadTestCase(
        path=sdata_path,
        expected_elements=not_corrupted,
        expected_exceptions=ArrowInvalid,
        warnings_patterns=[rf"points/{corrupted}: ArrowInvalid"],
    )


@pytest.fixture(scope="module")
def sdata_with_corrupted_parquet_zarrv2(session_tmp_path: Path) -> PartialReadTestCase:
    # points/blobs_points/0 is a zero-byte file or aborted during write
    sdata = blobs()
    sdata_path = session_tmp_path / "sdata_with_corrupted_parquet_zarrv2.zarr"
    sdata.write(sdata_path, sdata_formats=SpatialDataContainerFormatV01())

    corrupted = "blobs_points"
    os.rename(
        sdata_path / "points" / corrupted / "points.parquet",
        sdata_path / "points" / corrupted / "points_corrupted.parquet",
    )
    (sdata_path / "points" / corrupted / "points.parquet").touch()

    not_corrupted = [name for _, name, _ in sdata.gen_elements() if name != corrupted]

    return PartialReadTestCase(
        path=sdata_path,
        expected_elements=not_corrupted,
        expected_exceptions=ArrowInvalid,
        warnings_patterns=[rf"points/{corrupted}: ArrowInvalid"],
    )


@pytest.fixture(scope="module")
def sdata_with_missing_zarr_json_element(session_tmp_path: Path) -> PartialReadTestCase:
    # zarr.json is missing
    sdata = blobs()
    sdata_path = session_tmp_path / "sdata_with_missing_zarr_json_element.zarr"
    sdata.write(sdata_path)

    corrupted = "blobs_image"
    (sdata_path / "images" / corrupted / "zarr.json").unlink()
    not_corrupted = [name for _, name, _ in sdata.gen_elements() if name != corrupted]

    return PartialReadTestCase(
        path=sdata_path,
        expected_elements=not_corrupted,
        expected_exceptions=OSError,
        warnings_patterns=[r"images/blobs_image: OSError:"],
    )


@pytest.fixture(scope="module")
def sdata_with_missing_zattrs_element(session_tmp_path: Path) -> PartialReadTestCase:
    # Zarrv2
    # .zattrs is missing
    sdata = blobs()
    sdata_path = session_tmp_path / "sdata_with_missing_zattrs_element.zarr"
    sdata.write(sdata_path, sdata_formats=SpatialDataContainerFormatV01())

    corrupted = "blobs_image"
    (sdata_path / "images" / corrupted / ".zattrs").unlink()
    not_corrupted = [name for _, name, _ in sdata.gen_elements() if name != corrupted]

    return PartialReadTestCase(
        path=sdata_path,
        expected_elements=not_corrupted,
        expected_exceptions=OSError,
        warnings_patterns=["OSError: Image location"],
    )


@pytest.fixture(scope="module")
def sdata_with_missing_image_chunks_zarrv3(
    session_tmp_path: Path,
) -> PartialReadTestCase:
    sdata = blobs()
    sdata_path = session_tmp_path / "sdata_with_missing_image_chunks_zarrv3.zarr"
    sdata.write(sdata_path)

    corrupted = "blobs_image"
    os.unlink(sdata_path / "images" / corrupted / "0" / "zarr.json")
    os.rename(sdata_path / "images" / corrupted / "0", sdata_path / "images" / corrupted / "0_corrupted")

    not_corrupted = [name for _, name, _ in sdata.gen_elements() if name != corrupted]

    return PartialReadTestCase(
        path=sdata_path,
        expected_elements=not_corrupted,
        expected_exceptions=(ArrayNotFoundError,),
        warnings_patterns=[rf"images/{corrupted}: ArrayNotFoundError"],
    )


@pytest.fixture(scope="module")
def sdata_with_missing_image_chunks_zarrv2(
    session_tmp_path: Path,
) -> PartialReadTestCase:
    # Zarrv2
    # .zattrs exists, but refers to binary array chunks that do not exist
    sdata = blobs()
    sdata_path = session_tmp_path / "sdata_with_missing_image_chunks_zarrv2.zarr"
    sdata.write(sdata_path, sdata_formats=SpatialDataContainerFormatV01())

    corrupted = "blobs_image"
    os.unlink(sdata_path / "images" / corrupted / "0" / ".zarray")
    os.rename(sdata_path / "images" / corrupted / "0", sdata_path / "images" / corrupted / "0_corrupted")

    not_corrupted = [name for _, name, _ in sdata.gen_elements() if name != corrupted]

    return PartialReadTestCase(
        path=sdata_path,
        expected_elements=not_corrupted,
        expected_exceptions=(ArrayNotFoundError,),
        warnings_patterns=[rf"images/{corrupted}: (ArrayNotFoundError|TypeError)"],
    )


@pytest.fixture(scope="module")
def sdata_with_invalid_zattrs_element_violating_spec(session_tmp_path: Path) -> PartialReadTestCase:
    # Zarr v2
    # .zattrs contains readable JSON which is not valid for SpatialData/NGFF specs
    # for example due to a missing/misspelled/renamed key
    sdata = blobs()
    sdata_path = session_tmp_path / "sdata_with_invalid_zattrs_violating_spec.zarr"
    sdata.write(sdata_path, sdata_formats=SpatialDataContainerFormatV01())

    corrupted = "blobs_image"
    json_dict = json.loads((sdata_path / "images" / corrupted / ".zattrs").read_text())
    del json_dict["multiscales"][0]["coordinateTransformations"]
    (sdata_path / "images" / corrupted / ".zattrs").write_text(json.dumps(json_dict, indent=4))
    not_corrupted = [name for _, name, _ in sdata.gen_elements() if name != corrupted]

    return PartialReadTestCase(
        path=sdata_path,
        expected_elements=not_corrupted,
        expected_exceptions=KeyError,
        warnings_patterns=[rf"images/{corrupted}: KeyError: coordinateTransformations"],
    )


@pytest.fixture(scope="module")
def sdata_with_invalid_zarr_json_element_violating_spec(session_tmp_path: Path) -> PartialReadTestCase:
    # zarr.json contains readable JSON which is not valid for SpatialData/NGFF specs
    # for example due to a missing/misspelled/renamed key
    sdata = blobs()
    sdata_path = session_tmp_path / "sdata_with_invalid_zarr_json_violating_spec.zarr"
    sdata.write(sdata_path)

    corrupted = "blobs_image"
    json_dict = json.loads((sdata_path / "images" / corrupted / "zarr.json").read_text())
    del json_dict["attributes"]["ome"]["multiscales"][0]["coordinateTransformations"]
    (sdata_path / "images" / corrupted / "zarr.json").write_text(json.dumps(json_dict, indent=4))
    not_corrupted = [name for _, name, _ in sdata.gen_elements() if name != corrupted]

    return PartialReadTestCase(
        path=sdata_path,
        expected_elements=not_corrupted,
        expected_exceptions=KeyError,
        warnings_patterns=[rf"images/{corrupted}: KeyError: coordinateTransformations"],
    )


def _create_sdata_with_table_region_not_found(session_tmp_path: Path, zarr_version: int) -> PartialReadTestCase:
    """Helper for table region not found test cases (zarr v2 and v3)."""
    # table/table/.zarr referring to a region that is not found
    # This has been emitting just a warning, but does not fail reading the table element.
    sdata = blobs()
    sdata_path = session_tmp_path / f"sdata_with_table_region_not_found_zarrv{zarr_version}.zarr"
    if zarr_version == 2:
        sdata.write(sdata_path, sdata_formats=SpatialDataContainerFormatV01())
    else:
        sdata.write(sdata_path)

    corrupted = "blobs_labels"
    # The element data is missing
    sdata.delete_element_from_disk(corrupted)
    # But the labels element is referenced as a region in a table
    adata = anndata.read_zarr(sdata_path / "tables" / "table")
    assert corrupted in adata.obs["region"].values

    not_corrupted = [name for _, name, _ in sdata.gen_elements() if name != corrupted]

    return PartialReadTestCase(
        path=sdata_path,
        expected_elements=not_corrupted,
        expected_exceptions=(),
        warnings_patterns=[
            rf"The table is annotating '{re.escape(corrupted)}', which is not present in the SpatialData object"
        ],
    )


@pytest.fixture(scope="module")
def sdata_with_table_region_not_found_zarrv3(session_tmp_path: Path) -> PartialReadTestCase:
    return _create_sdata_with_table_region_not_found(session_tmp_path, zarr_version=3)


@pytest.fixture(scope="module")
def sdata_with_table_region_not_found_zarrv2(session_tmp_path: Path) -> PartialReadTestCase:
    return _create_sdata_with_table_region_not_found(session_tmp_path, zarr_version=2)


@pytest.mark.parametrize(
    "test_case",
    [
        sdata_with_corrupted_elem_types_zgroup,  # JSONDecodeError
        sdata_with_corrupted_elem_types_zarr_json,  # JSONDecodeError
        sdata_with_corrupted_zarr_json_elements,  # OSError
        sdata_with_corrupted_zattrs_elements,  # OSError
        sdata_with_corrupted_image_chunks_zarrv3,  # zarr.errors.ArrayNotFoundError
        sdata_with_corrupted_image_chunks_zarrv2,  # zarr.errors.ArrayNotFoundError
        sdata_with_corrupted_parquet_zarrv3,  # ArrowInvalid
        sdata_with_corrupted_parquet_zarrv2,  # ArrowInvalid
        sdata_with_missing_zarr_json_element,  # OSError
        sdata_with_missing_zattrs_element,  # OSError
        sdata_with_missing_image_chunks_zarrv3,  # zarr.errors.ArrayNotFoundError
        sdata_with_missing_image_chunks_zarrv2,  # zarr.errors.ArrayNotFoundError
        sdata_with_invalid_zattrs_element_violating_spec,  # KeyError
        sdata_with_invalid_zarr_json_element_violating_spec,  # KeyError
        sdata_with_table_region_not_found_zarrv3,
        sdata_with_table_region_not_found_zarrv2,
    ],
    indirect=True,
)
def test_read_zarr_with_error(test_case: PartialReadTestCase):
    if test_case.expected_exceptions:
        with pytest.raises(test_case.expected_exceptions):
            read_zarr(test_case.path, on_bad_files="error")
    else:
        read_zarr(test_case.path, on_bad_files="error")


@pytest.mark.parametrize(
    "test_case",
    [
        sdata_with_corrupted_elem_types_zgroup,  # JSONDecodeError
        sdata_with_corrupted_elem_types_zarr_json,  # JSONDecodeError
        sdata_with_corrupted_zarr_json_elements,  # JSONDecodeError for non raster, else OSError
        sdata_with_corrupted_zattrs_elements,  # JSONDecodeError for non raster, else OSError
        sdata_with_corrupted_image_chunks_zarrv3,  # zarr.errors.ArrayNotFoundError
        sdata_with_corrupted_image_chunks_zarrv2,  # zarr.errors.ArrayNotFoundError
        sdata_with_corrupted_parquet_zarrv3,  # ArrowInvalid
        sdata_with_corrupted_parquet_zarrv2,  # ArrowInvalid
        sdata_with_missing_zarr_json_element,  # OSError
        sdata_with_missing_zattrs_element,  # OSError
        sdata_with_missing_image_chunks_zarrv3,  # zarr.errors.ArrayNotFoundError
        sdata_with_missing_image_chunks_zarrv2,  # zarr.errors.ArrayNotFoundError
        sdata_with_invalid_zattrs_element_violating_spec,  # KeyError
        sdata_with_invalid_zarr_json_element_violating_spec,  # KeyError
        sdata_with_table_region_not_found_zarrv3,
        sdata_with_table_region_not_found_zarrv2,
    ],
    indirect=True,
)
def test_read_zarr_with_warnings(test_case: PartialReadTestCase):
    with pytest_warns_multiple(UserWarning, matches=test_case.warnings_patterns):
        actual: SpatialData = read_zarr(test_case.path, on_bad_files="warn")

    actual_elements = {name for _, name, _ in actual.gen_elements()}
    assert set(test_case.expected_elements) == actual_elements
