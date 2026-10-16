"""Tests for MuData tables, used to store multimodal measurements (e.g. RNA and protein) of the same instances."""

from __future__ import annotations

from pathlib import Path

import annsel as an
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import zarr
from anndata import AnnData
from mudata import MuData
from shapely import Point

from spatialdata import (
    SpatialData,
    concatenate,
    deepcopy,
    filter_by_table_query,
    get_values,
    join_spatialelement_table,
    match_table_to_element,
    sanitize_table,
)
from spatialdata._core.validation import ValidationError
from spatialdata.models import ShapesModel, TableModel
from spatialdata.testing import assert_elements_are_identical, assert_spatial_data_objects_are_identical

N_CELLS = 6
# the protein panel is measured only on the first N_PROT_CELLS cells, to test modalities covering a subset of the obs
N_PROT_CELLS = 4
RNG = np.random.default_rng(0)


def _get_circles(n: int = N_CELLS) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(geometry=[Point(i, i) for i in range(n)], index=np.arange(n))
    gdf["radius"] = 0.5
    return ShapesModel.parse(gdf)


def _get_mudata(region: str = "cells", n: int = N_CELLS, n_prot: int = N_PROT_CELLS, prefix: str = "") -> MuData:
    obs_names = [f"{prefix}cell_{i}" for i in range(n)]
    rna = AnnData(RNG.random((n, 3)), obs=pd.DataFrame(index=obs_names))
    rna.var_names = ["Gene1", "Gene2", "CD45"]
    rna.obs["leiden"] = pd.Categorical(["a", "b"] * (n // 2))
    prot = AnnData(RNG.random((n_prot, 2)), obs=pd.DataFrame(index=obs_names[:n_prot]))
    prot.var_names = ["CD45", "CD68"]
    mdata = MuData({"rna": rna, "prot": prot})
    mdata.obs["region"] = pd.Categorical([region] * n)
    mdata.obs["instance_id"] = np.arange(n)
    return TableModel.parse(mdata, region=region, region_key="region", instance_key="instance_id")


@pytest.fixture
def sdata_mudata() -> SpatialData:
    return SpatialData(shapes={"cells": _get_circles()}, tables={"table": _get_mudata()})


def test_parse_and_validate():
    mdata = _get_mudata()
    assert isinstance(mdata, MuData)
    assert mdata.uns[TableModel.ATTRS_KEY] == {"region": "cells", "region_key": "region", "instance_key": "instance_id"}
    assert isinstance(mdata.obs["region"].dtype, pd.CategoricalDtype)
    TableModel.validate(mdata)


def test_parse_requires_global_obs_columns():
    mdata = _get_mudata()
    del mdata.uns[TableModel.ATTRS_KEY]
    mdata.mod["rna"].obs["cell_index"] = np.arange(N_CELLS)
    with pytest.raises(ValueError, match="mdata.pull_obs"):
        TableModel.parse(mdata, region="cells", region_key="region", instance_key="cell_index")


def test_validate_invalid_names_in_modality():
    mdata = _get_mudata()
    mdata.mod["prot"].obs["bad name"] = 0
    with pytest.raises(ValidationError, match="Table contains invalid names") as e:
        TableModel.validate(mdata)
    assert e.value.errors[0].location == ("mod", "prot", "obs", "bad name")
    sanitize_table(mdata)
    assert "bad_name" in mdata.mod["prot"].obs
    TableModel.validate(mdata)


def test_sdata_with_mudata_table(sdata_mudata: SpatialData):
    assert isinstance(sdata_mudata["table"], MuData)
    assert SpatialData.get_annotated_regions(sdata_mudata["table"]) == ["cells"]
    assert "table" in repr(sdata_mudata)
    sdata_mudata.validate_table_in_spatialdata(sdata_mudata["table"])


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_io_roundtrip(sdata_mudata: SpatialData, tmp_path: Path, zarr_format: int):
    from spatialdata._io.format import SpatialDataContainerFormatV01, TablesFormatV01

    path = tmp_path / "data.zarr"
    formats = None if zarr_format == 3 else [SpatialDataContainerFormatV01(), TablesFormatV01()]
    if zarr_format == 2:
        with pytest.warns(UserWarning):
            sdata_mudata.write(path, sdata_formats=formats)
    else:
        sdata_mudata.write(path, sdata_formats=formats)

    table_group = zarr.open_group(path / "tables" / "table", mode="r")
    assert table_group.attrs["encoding-type"] == "MuData"
    assert table_group.attrs["spatialdata-encoding-type"] == "ngff:regions_table"

    sdata_read = SpatialData.read(path)
    assert isinstance(sdata_read["table"], MuData)
    assert sdata_read["table"].mod["prot"].n_obs == N_PROT_CELLS
    assert_spatial_data_objects_are_identical(sdata_mudata, sdata_read)


def test_write_element(sdata_mudata: SpatialData, tmp_path: Path):
    path = tmp_path / "data.zarr"
    sdata = SpatialData(shapes=sdata_mudata.shapes)
    sdata.write(path)
    sdata["table"] = sdata_mudata["table"]
    sdata.write_element("table")
    assert isinstance(SpatialData.read(path)["table"], MuData)


def test_deepcopy(sdata_mudata: SpatialData):
    copied = deepcopy(sdata_mudata)
    assert copied["table"] is not sdata_mudata["table"]
    assert_spatial_data_objects_are_identical(sdata_mudata, copied)


def test_match_table_to_element(sdata_mudata: SpatialData):
    # drop and reorder some cells of the shapes element; the table must follow the order of the element
    sdata_mudata["cells"] = sdata_mudata["cells"].iloc[[3, 1, 5]]
    matched = match_table_to_element(sdata_mudata, element_name="cells", table_name="table")
    assert isinstance(matched, MuData)
    assert matched.obs["instance_id"].tolist() == [3, 1, 5]
    assert matched.mod["rna"].n_obs == 3
    # cell 5 has no protein measurement
    assert matched.mod["prot"].obs_names.tolist() == ["cell_3", "cell_1"]


@pytest.mark.parametrize(
    ("how", "expected_table_ids", "expected_n_prot"),
    [("left", [0, 2, 4, 5], 2), ("inner", [0, 2, 4, 5], 2), ("right", list(range(N_CELLS)), N_PROT_CELLS)],
)
def test_join(sdata_mudata: SpatialData, how: str, expected_table_ids: list[int], expected_n_prot: int):
    sdata_mudata["cells"] = sdata_mudata["cells"].iloc[[0, 2, 4, 5]]
    elements, table = join_spatialelement_table(
        sdata=sdata_mudata, spatial_element_names="cells", table_name="table", how=how
    )
    assert isinstance(table, MuData)
    assert table.obs["instance_id"].tolist() == expected_table_ids
    assert table.mod["prot"].n_obs == expected_n_prot
    assert elements["cells"].index.tolist() == [0, 2, 4, 5]


def test_filter_by_table_query(sdata_mudata: SpatialData):
    # the global var_names are not unique (CD45 is both a gene and a protein), which annsel does not support
    assert not sdata_mudata["table"].var_names.is_unique
    filtered = filter_by_table_query(sdata_mudata, "table", obs_expr=an.col("instance_id") > 2)
    assert isinstance(filtered["table"], MuData)
    assert filtered["table"].obs["instance_id"].tolist() == [3, 4, 5]
    assert filtered["table"].mod["prot"].n_obs == 1
    assert filtered["cells"].index.tolist() == [3, 4, 5]
    with pytest.raises(NotImplementedError, match="only `obs_expr` and `obs_names_expr` are supported"):
        filter_by_table_query(sdata_mudata, "table", var_names_expr=an.var_names == "CD68")


def test_bounding_box_query_filters_table(sdata_mudata: SpatialData):
    queried = sdata_mudata.query.bounding_box(
        axes=("x", "y"), min_coordinate=[-0.1, -0.1], max_coordinate=[2.1, 2.1], target_coordinate_system="global"
    )
    assert isinstance(queried["table"], MuData)
    assert queried["table"].obs["instance_id"].tolist() == [0, 1, 2]
    assert queried["table"].mod["prot"].n_obs == 3


def test_get_values(sdata_mudata: SpatialData):
    mdata = sdata_mudata["table"]

    # global obs column
    df = get_values("instance_id", sdata=sdata_mudata, element_name="cells", table_name="table")
    assert df["instance_id"].tolist() == list(range(N_CELLS))

    # variable found in a single modality, no prefix needed; NaN for the cells without protein measurement
    df = get_values("CD68", sdata=sdata_mudata, element_name="cells", table_name="table")
    expected = mdata.mod["prot"][:, "CD68"].X.ravel()
    np.testing.assert_allclose(df["CD68"].to_numpy()[:N_PROT_CELLS], expected)
    assert df["CD68"].isna().sum() == N_CELLS - N_PROT_CELLS
    assert df.index.tolist() == list(range(N_CELLS))

    # variable found in multiple modalities requires the modality prefix
    with pytest.raises(ValueError, match="prefix it with the name of the modality"):
        get_values("CD45", sdata=sdata_mudata, element_name="cells", table_name="table")
    df = get_values(["rna:CD45", "prot:CD45"], element=mdata)
    np.testing.assert_allclose(df["rna:CD45"], mdata.mod["rna"][:, "CD45"].X.ravel())
    np.testing.assert_allclose(df["prot:CD45"].to_numpy()[:N_PROT_CELLS], mdata.mod["prot"][:, "CD45"].X.ravel())

    # categorical obs column of a modality
    df = get_values("rna:leiden", element=mdata)
    assert isinstance(df["rna:leiden"].dtype, pd.CategoricalDtype)
    assert df["rna:leiden"].tolist() == mdata.mod["rna"].obs["leiden"].tolist()

    with pytest.raises(ValueError, match="has not been found"):
        get_values("not_a_key", element=mdata)


def test_concatenate_tables():
    sdata0 = SpatialData(shapes={"cells0": _get_circles()}, tables={"table": _get_mudata(region="cells0")})
    sdata1 = SpatialData(
        shapes={"cells1": _get_circles()}, tables={"table": _get_mudata(region="cells1", prefix="s1_")}
    )
    merged = concatenate([sdata0, sdata1], concatenate_tables=True)
    table = merged["table"]
    assert isinstance(table, MuData)
    assert table.n_obs == 2 * N_CELLS
    assert table.mod["prot"].n_obs == 2 * N_PROT_CELLS
    assert set(SpatialData.get_annotated_regions(table)) == {"cells0", "cells1"}
    assert table.obs_names.is_unique

    # elements with the same names: the obs names of the global obs and of the modalities get the same suffix
    merged = concatenate({"a": sdata0, "b": sdata0}, concatenate_tables=True)
    table = merged["table"]
    assert set(SpatialData.get_annotated_regions(table)) == {"cells0-a", "cells0-b"}
    assert table.mod["prot"].obs_names.tolist() == [f"cell_{i}-a" for i in range(N_PROT_CELLS)] + [
        f"cell_{i}-b" for i in range(N_PROT_CELLS)
    ]
    assert set(table.mod["prot"].obs_names).issubset(table.obs_names)


def test_concatenate_tables_duplicated_obs_names():
    sdata0 = SpatialData(shapes={"cells0": _get_circles()}, tables={"table": _get_mudata(region="cells0")})
    sdata1 = SpatialData(shapes={"cells1": _get_circles()}, tables={"table": _get_mudata(region="cells1")})
    with pytest.raises(ValueError, match="`obs_names` of the MuData tables to concatenate are not unique"):
        concatenate([sdata0, sdata1], concatenate_tables=True)


def test_concatenate_mixed_table_types():
    adata = AnnData(
        obs=pd.DataFrame({"region": pd.Categorical(["cells1"] * N_CELLS), "instance_id": np.arange(N_CELLS)})
    )
    adata = TableModel.parse(adata, region="cells1", region_key="region", instance_key="instance_id")
    sdata0 = SpatialData(shapes={"cells0": _get_circles()}, tables={"table": _get_mudata(region="cells0")})
    sdata1 = SpatialData(shapes={"cells1": _get_circles()}, tables={"table": adata})
    with pytest.raises(TypeError, match="Cannot concatenate AnnData tables with MuData tables"):
        concatenate([sdata0, sdata1], concatenate_tables=True)


def test_assert_elements_are_identical_detects_modality_differences():
    mdata0 = _get_mudata()
    mdata1 = mdata0.copy()
    mdata1.mod["prot"].X[0, 0] += 1
    with pytest.raises(AssertionError):
        assert_elements_are_identical(mdata0, mdata1)
