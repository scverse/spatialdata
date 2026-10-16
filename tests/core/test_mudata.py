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
from spatialdata._io.exceptions import WritingToZarrV2DeprecationWarning
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
    rna.var["highly_variable"] = [True, False, True]
    rna.layers["counts"] = RNG.integers(0, 10, (n, 3)).astype(float)
    rna.obs["leiden"] = pd.Categorical(["a", "b"] * (n // 2))
    rna.obs["sample"] = ["s0", "s1"] * (n // 2)
    prot = AnnData(RNG.random((n_prot, 2)), obs=pd.DataFrame(index=obs_names[:n_prot]))
    prot.var_names = ["CD45", "CD68"]
    prot.obs["pbool"] = [True, False] * (n_prot // 2)
    mdata = MuData({"rna": rna, "prot": prot})
    mdata.obs["region"] = pd.Categorical([region] * n)
    mdata.obs["instance_id"] = np.arange(n)
    # global annotations of the dtypes that MuData converts when it is rebuilt
    mdata.obs["score"] = np.arange(n, dtype=float)
    mdata.obs["score_nan"] = [np.nan] + list(np.arange(1, n, dtype=float))
    mdata.obs["label"] = ["x", "y", "y"] * (n // 3)
    mdata.obs["flag"] = [True, False] * (n // 2)
    mdata.obs.index.name = "cell"
    mdata.obsm["spatial"] = np.stack([np.arange(n), np.arange(n)], axis=1).astype(float)
    mdata.obsp["distances"] = np.abs(np.subtract.outer(np.arange(n), np.arange(n))).astype(float)
    return TableModel.parse(mdata, region=region, region_key="region", instance_key="instance_id")


def _obs_dtypes(table: MuData) -> dict[str, dict[str, object]]:
    dtypes = {"global": table.obs.dtypes.to_dict()} | {name: m.obs.dtypes.to_dict() for name, m in table.mod.items()}
    index_names = {"global": table.obs.index.name} | {name: m.obs.index.name for name, m in table.mod.items()}
    return {"dtypes": dtypes, "index_names": index_names}


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
    with pytest.raises(ValueError, match="prefix_unique=False"):
        TableModel.parse(mdata, region="cells", region_key="region", instance_key="cell_index")
    # the failed parse left no metadata behind, and the suggested fix works
    assert TableModel.ATTRS_KEY not in mdata.uns
    mdata.pull_obs(columns=["cell_index"], mods=["rna"], prefix_unique=False)
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


def test_sanitize_table_renames_modalities():
    mdata = _get_mudata()
    mdata = MuData({"Gene Expression": mdata.mod["rna"], "prot": mdata.mod["prot"]}, obs=mdata.obs, uns=mdata.uns)
    with pytest.raises(ValidationError, match="sanitize_table"):
        TableModel.validate(mdata)
    sanitize_table(mdata)
    assert list(mdata.mod) == ["Gene_Expression", "prot"]
    for mapping in (mdata.obsm, mdata.varm, mdata.obsmap, mdata.varmap):
        assert list(mapping) == ["Gene_Expression", "prot"]
    TableModel.validate(mdata)
    TableModel.validate(deepcopy(mdata))


def test_sanitize_table_keeps_modality_masks(tmp_path: Path):
    mdata = _get_mudata()
    # collides case-insensitively with the mask of the modality "rna", and comes before it
    mdata = MuData(dict(mdata.mod), obs=mdata.obs, uns=mdata.uns, obsm={"RNA": np.ones((N_CELLS, 1))})
    assert list(mdata.obsm)[0] == "RNA"
    with pytest.raises(ValidationError):
        TableModel.validate(mdata)
    sanitize_table(mdata)
    assert "rna" in mdata.obsm
    assert "RNA_1" in mdata.obsm
    sdata = SpatialData(shapes={"cells": _get_circles()}, tables={"table": mdata})
    sdata.write(tmp_path / "data.zarr")
    SpatialData.read(tmp_path / "data.zarr")
    deepcopy(sdata)


def test_sanitize_table_with_varp():
    mdata = _get_mudata()
    mdata.varp["corr"] = np.eye(mdata.n_vars)
    mdata.obsm["bad key"] = np.ones((N_CELLS, 1))
    sanitize_table(mdata)
    assert list(mdata.varp) == ["corr"]
    assert "bad_key" in mdata.obsm
    mdata.varp["corr2"] = np.eye(mdata.n_vars)


def test_sanitize_table_not_inplace():
    mdata = _get_mudata()
    mdata.mod["prot"].obs["bad name"] = 0
    sanitized = sanitize_table(mdata, inplace=False)
    assert isinstance(sanitized, MuData)
    assert "bad_name" in sanitized.mod["prot"].obs
    assert "bad name" in mdata.mod["prot"].obs


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
        with pytest.warns(WritingToZarrV2DeprecationWarning):
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


def test_write_does_not_modify_table(sdata_mudata: SpatialData, tmp_path: Path):
    before = _obs_dtypes(sdata_mudata["table"])
    sdata_mudata.write(tmp_path / "data.zarr")
    assert _obs_dtypes(sdata_mudata["table"]) == before
    # the str column of the modality was not converted to categorical in place
    sdata_mudata["table"].mod["rna"].obs.loc["cell_0", "sample"] = "new_sample"
    assert _obs_dtypes(SpatialData.read(tmp_path / "data.zarr")["table"]) == before


def test_io_roundtrip_repeated_str_instance_ids(tmp_path: Path):
    # two elements indexed by the same str ids: the instance ids repeat across regions, and must not be stored as
    # categorical, otherwise they do not match the dtype of the index of the elements when reading
    shapes = {}
    for region in ("a", "b"):
        gdf = _get_circles(3)
        gdf.index = pd.Index(["c0", "c1", "c2"], dtype=object)
        shapes[region] = gdf
    rna = AnnData(RNG.random((6, 2)), obs=pd.DataFrame(index=[f"{r}_c{i}" for r in "ab" for i in range(3)]))
    mdata = MuData({"rna": rna})
    mdata.obs["region"] = pd.Categorical(["a"] * 3 + ["b"] * 3)
    mdata.obs["cell_id"] = pd.Series(["c0", "c1", "c2"] * 2, index=mdata.obs_names, dtype=object)
    mdata = TableModel.parse(mdata, region=["a", "b"], region_key="region", instance_key="cell_id")
    sdata = SpatialData(shapes=shapes, tables={"table": mdata})
    sdata.write(tmp_path / "data.zarr")
    sdata_read = SpatialData.read(tmp_path / "data.zarr")
    assert not isinstance(sdata_read["table"].obs["cell_id"].dtype, pd.CategoricalDtype)
    assert_elements_are_identical(sdata["table"], sdata_read["table"])


def test_write_view(sdata_mudata: SpatialData, tmp_path: Path):
    view = sdata_mudata["table"][[0, 1, 2, 4]]
    assert view.is_view
    sdata = SpatialData(shapes=sdata_mudata.shapes, tables={"table": view})
    sdata.write(tmp_path / "data.zarr")
    table_read = SpatialData.read(tmp_path / "data.zarr")["table"]
    assert table_read.obs["instance_id"].tolist() == [0, 1, 2, 4]
    assert table_read.mod["prot"].n_obs == 3
    assert view.var_names.nlevels == 1


def test_read_incomplete_table_group_with_warn(sdata_mudata: SpatialData, tmp_path: Path):
    from spatialdata import read_zarr

    path = tmp_path / "data.zarr"
    # a write that failed while writing the table leaves an empty table group, and no consolidated metadata
    SpatialData(shapes=sdata_mudata.shapes).write(path, consolidate_metadata=False)
    zarr.open_group(path / "tables", mode="a").create_group("table")
    with pytest.warns(UserWarning, match="tables/table"):
        sdata = read_zarr(path, on_bad_files="warn")
    assert "cells" in sdata.shapes
    assert len(sdata.tables) == 0


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
    # MuData.copy() alone would turn the float columns into int64/Int64 and the str column into "string"
    assert _obs_dtypes(copied["table"]) == _obs_dtypes(sdata_mudata["table"])


def test_match_table_to_element(sdata_mudata: SpatialData):
    # drop and reorder some cells of the shapes element; the table must follow the order of the element
    sdata_mudata["cells"] = sdata_mudata["cells"].iloc[[3, 1, 5]]
    matched = match_table_to_element(sdata_mudata, element_name="cells", table_name="table")
    assert isinstance(matched, MuData)
    assert matched.obs["instance_id"].tolist() == [3, 1, 5]
    assert matched.mod["rna"].n_obs == 3
    # cell 5 has no protein measurement
    assert matched.mod["prot"].obs_names.tolist() == ["cell_3", "cell_1"]
    assert matched.obs.dtypes.to_dict() == sdata_mudata["table"].obs.dtypes.to_dict()


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


def test_subset_keeps_modality_categories(sdata_mudata: SpatialData):
    # cells 0, 2 and 4 are all in leiden cluster "a"
    sdata_mudata["cells"] = sdata_mudata["cells"].iloc[[0, 2, 4]]
    matched = match_table_to_element(sdata_mudata, element_name="cells", table_name="table")
    assert matched.mod["rna"].obs["leiden"].cat.categories.tolist() == ["a", "b"]
    values = get_values("rna:leiden", sdata=sdata_mudata, element_name="cells", table_name="table")
    assert values["rna:leiden"].cat.categories.tolist() == ["a", "b"]


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


def _get_two_fov_mudata(prot_cells: list[int]) -> MuData:
    """Two fields of view whose cells have the same obs names ("0", "1", "2") in both."""
    obs_names = ["0", "1", "2"] * 2
    rna = AnnData(np.arange(6, dtype=float)[:, None], obs=pd.DataFrame(index=obs_names))
    rna.var_names = ["Gene1"]
    prot = AnnData(
        100 + np.arange(6, dtype=float)[prot_cells, None], obs=pd.DataFrame(index=[obs_names[i] for i in prot_cells])
    )
    prot.var_names = ["CD68"]
    obsmap = {"rna": np.arange(1, 7, dtype=np.uint32), "prot": np.zeros(6, dtype=np.uint32)}
    obsmap["prot"][prot_cells] = np.arange(1, len(prot_cells) + 1)
    obs = pd.DataFrame(
        {"region": pd.Categorical(["fov_a"] * 3 + ["fov_b"] * 3), "instance_id": [0, 1, 2] * 2}, index=obs_names
    )
    mdata = MuData({"rna": rna, "prot": prot}, obs=obs, obsmap=obsmap)
    return TableModel.parse(mdata, region=["fov_a", "fov_b"], region_key="region", instance_key="instance_id")


@pytest.mark.parametrize(
    ("prot_cells", "expected"),
    [([0, 1, 2], [100, 101, 102, np.nan, np.nan, np.nan]), ([0, 1, 3, 4], [100, 101, np.nan, 103, 104, np.nan])],
)
def test_get_values_repeated_obs_names(prot_cells: list[int], expected: list[float]):
    mdata = _get_two_fov_mudata(prot_cells)
    np.testing.assert_array_equal(get_values("CD68", element=mdata)["CD68"].to_numpy(), expected)
    np.testing.assert_array_equal(get_values("CD68", element=mdata, element_name="fov_b")["CD68"], expected[3:])


def test_get_values_bool_with_missing_cells_is_float(sdata_mudata: SpatialData):
    values = get_values("prot:pbool", sdata=sdata_mudata, element_name="cells", table_name="table")["prot:pbool"]
    assert values.dtype == np.float64
    np.testing.assert_array_equal(values, [1.0, 0.0, 1.0, 0.0, np.nan, np.nan])
    # measured in all the cells: the dtype is kept
    assert get_values("flag", element=sdata_mudata["table"])["flag"].dtype == bool


def test_get_values_obsm_key_also_a_variable(sdata_mudata: SpatialData):
    mdata = sdata_mudata["table"]
    mdata.obsm["CD68"] = np.ones((N_CELLS, 2))
    df = get_values("CD68", element=mdata)
    assert df.columns.tolist() == ["CD68_0", "CD68_1"]


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


def test_concatenate_does_not_modify_inputs():
    sdata0 = SpatialData(shapes={"cells0": _get_circles()}, tables={"table0": _get_mudata(region="cells0")})
    sdata1 = SpatialData(
        shapes={"cells1": _get_circles()}, tables={"table1": _get_mudata(region="cells1", prefix="s1_")}
    )
    before = _obs_dtypes(sdata0["table0"])
    merged = concatenate([sdata0, sdata1])
    assert _obs_dtypes(sdata0["table0"]) == before
    assert _obs_dtypes(merged["table0"]) == before


def test_concatenate_keeps_obs_index_name_of_anndata_tables():
    adata = AnnData(obs=pd.DataFrame({"region": pd.Categorical(["cells"] * 3), "instance_id": np.arange(3)}))
    adata.obs.index.name = "cell_id"
    adata = TableModel.parse(adata, region="cells", region_key="region", instance_key="instance_id")
    sdata = SpatialData(shapes={"cells": _get_circles(3)}, tables={"table": adata})
    merged = concatenate({"x": sdata, "y": sdata}, concatenate_tables=True)
    assert merged["table"].obs.index.name == "cell_id"


def _assert_modalities_match_global_obs(table: MuData) -> None:
    """Each modality row must map to the global observation with the same name and the same instance id."""
    for name, modality in table.mod.items():
        obsmap = np.asarray(table.obsmap[name]).ravel()
        measured = obsmap > 0
        assert (modality.obs_names[obsmap[measured] - 1] == table.obs_names[measured]).all()


@pytest.mark.parametrize("index_unique", [None, "-"])
def test_concatenate_tables_duplicated_obs_names(index_unique: str | None):
    sdata0 = SpatialData(shapes={"cells0": _get_circles()}, tables={"table": _get_mudata(region="cells0")})
    sdata1 = SpatialData(shapes={"cells1": _get_circles()}, tables={"table": _get_mudata(region="cells1")})
    kwargs = {} if index_unique is None else {"index_unique": index_unique}
    table = concatenate([sdata0, sdata1], concatenate_tables=True, **kwargs)["table"]
    assert table.n_obs == 2 * N_CELLS
    assert table.obs_names.is_unique
    assert table.obs["region"].tolist() == ["cells0"] * N_CELLS + ["cells1"] * N_CELLS
    assert table.mod["prot"].n_obs == 2 * N_PROT_CELLS
    _assert_modalities_match_global_obs(table)


def test_concatenate_tables_result_is_not_a_view():
    sdata0 = SpatialData(shapes={"cells0": _get_circles()}, tables={"table": _get_mudata(region="cells0")})
    sdata1 = SpatialData(
        shapes={"cells1": _get_circles()}, tables={"table": _get_mudata(region="cells1", prefix="s1_")}
    )
    for obs_names_make_unique in (True, False):
        table = concatenate([sdata0, sdata1], concatenate_tables=True, obs_names_make_unique=obs_names_make_unique)[
            "table"
        ]
        table.obsm["spatial"][0, 0] = 42
        assert table.obsm["spatial"][0, 0] == 42
        table.obs["new_column"] = 1
        assert "new_column" in table.obs


def test_concatenate_tables_with_different_modalities():
    # the first table has no protein measurement: the positional arguments must follow the global observations
    mdata0 = _get_mudata(region="cells0")
    mdata0 = TableModel.parse(MuData({"rna": mdata0.mod["rna"]}, obs=mdata0.obs.copy(), uns=mdata0.uns))
    sdata0 = SpatialData(shapes={"cells0": _get_circles()}, tables={"table": mdata0})
    sdata1 = SpatialData(
        shapes={"cells1": _get_circles()}, tables={"table": _get_mudata(region="cells1", prefix="s1_")}
    )
    table = concatenate([sdata0, sdata1], concatenate_tables=True, label="batch", keys=["A", "B"])["table"]
    assert table.obs["batch"].tolist() == ["A"] * N_CELLS + ["B"] * N_CELLS
    assert table.mod["prot"].n_obs == N_PROT_CELLS
    assert (table.obs.loc[table.mod["prot"].obs_names, "batch"] == "B").all()
    _assert_modalities_match_global_obs(table)
    table = concatenate([sdata0, sdata1], concatenate_tables=True, index_unique="-")["table"]
    _assert_modalities_match_global_obs(table)


def test_concatenate_tables_keeps_global_attributes():
    sdata0 = SpatialData(shapes={"cells0": _get_circles()}, tables={"table": _get_mudata(region="cells0")})
    sdata1 = SpatialData(
        shapes={"cells1": _get_circles()}, tables={"table": _get_mudata(region="cells1", prefix="s1_")}
    )
    sdata0["table"].varm["loadings"] = np.ones((sdata0["table"].n_vars, 2))
    sdata1["table"].varm["loadings"] = np.ones((sdata1["table"].n_vars, 2))
    table = concatenate([sdata0, sdata1], concatenate_tables=True, pairwise=True, merge="same")["table"]
    assert table.obsp["distances"].shape == (2 * N_CELLS, 2 * N_CELLS)
    assert "loadings" in table.varm
    dtypes = table.obs.dtypes.drop("region").to_dict()
    assert dtypes == sdata0["table"].obs.dtypes.drop("region").to_dict()


def test_concatenate_tables_keeps_axis_and_row_order():
    # with a modality covering only some cells first, the rows must still follow the order of the tables
    def _get_table(region: str, prefix: str, axis: int) -> MuData:
        names = [f"{prefix}{i}" for i in range(4)]
        if axis == 0:
            prot = AnnData(np.ones((2, 1)), obs=pd.DataFrame(index=names[:2]))
            rna = AnnData(np.ones((4, 2)), obs=pd.DataFrame(index=names))
            mdata = MuData({"prot": prot, "rna": rna})
        else:
            b1 = AnnData(np.ones((2, 2)), obs=pd.DataFrame(index=names[:2]))
            b2 = AnnData(np.ones((2, 2)), obs=pd.DataFrame(index=names[2:]))
            mdata = MuData({"b1": b1, "b2": b2}, axis=1)
        mdata.obs["region"] = pd.Categorical([region] * 4)
        mdata.obs["instance_id"] = np.arange(4)
        return TableModel.parse(mdata, region=region, region_key="region", instance_key="instance_id")

    for axis in (0, 1):
        sdatas = [
            SpatialData(shapes={f"cells{k}": _get_circles(4)}, tables={"table": _get_table(f"cells{k}", p, axis)})
            for k, p in enumerate(["A", "B"])
        ]
        table = concatenate(sdatas, concatenate_tables=True)["table"]
        assert table.axis == axis
        assert table.obs_names.tolist() == [f"{p}{i}" for p in "AB" for i in range(4)]
        _assert_modalities_match_global_obs(table)


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
