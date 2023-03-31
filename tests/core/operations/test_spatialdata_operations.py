import tempfile
from pathlib import Path

import numpy as np
import pytest
from anndata import AnnData
from dask.dataframe.core import DataFrame as DaskDataFrame
from dask.delayed import Delayed
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
from spatialdata import SpatialData
from spatialdata._core.concatenate import _concatenate_tables, concatenate
from spatialdata.models import (
    Image2DModel,
    Labels2DModel,
    PointsModel,
    ShapesModel,
    TableModel,
)
from spatialdata.transformations.operations import set_transformation
from spatialdata.transformations.transformations import Identity, Scale

from tests.conftest import _get_table


def test_element_names_unique():
    shapes = ShapesModel.parse(np.array([[0, 0]]), geometry=0, radius=1)
    points = PointsModel.parse(np.array([[0, 0]]))
    labels = Labels2DModel.parse(np.array([[0, 0], [0, 0]]), dims=["y", "x"])
    image = Image2DModel.parse(np.array([[[0, 0], [0, 0]]]), dims=["c", "y", "x"])

    with pytest.raises(ValueError):
        SpatialData(images={"image": image}, points={"image": points})
    with pytest.raises(ValueError):
        SpatialData(images={"image": image}, shapes={"image": shapes})
    with pytest.raises(ValueError):
        SpatialData(images={"image": image}, labels={"image": labels})

    sdata = SpatialData(
        images={"image": image}, points={"points": points}, shapes={"shapes": shapes}, labels={"labels": labels}
    )

    with pytest.raises(ValueError):
        sdata.add_image(name="points", image=image)
    with pytest.raises(ValueError):
        sdata.add_points(name="image", points=points)
    with pytest.raises(ValueError):
        sdata.add_shapes(name="image", shapes=shapes)
    with pytest.raises(ValueError):
        sdata.add_labels(name="image", labels=labels)


def _assert_elements_left_to_right_seem_identical(sdata0: SpatialData, sdata1: SpatialData):
    for element_type, element_name, element in sdata0._gen_elements():
        elements = sdata1.__getattribute__(element_type)
        assert element_name in elements
        element1 = elements[element_name]
        if isinstance(element, (AnnData, SpatialImage, GeoDataFrame)):
            assert element.shape == element1.shape
        elif isinstance(element, DaskDataFrame):
            for s0, s1 in zip(element.shape, element1.shape):
                if isinstance(s0, Delayed):
                    s0 = s0.compute()
                if isinstance(s1, Delayed):
                    s1 = s1.compute()
                assert s0 == s1
        elif isinstance(element, MultiscaleSpatialImage):
            assert len(element) == len(element1)
        else:
            raise TypeError(f"Unsupported type {type(element)}")


def _assert_tables_seem_identical(table0: AnnData, table1: AnnData):
    assert table0.shape == table1.shape


def _assert_spatialdata_objects_seem_identical(sdata0: SpatialData, sdata1: SpatialData):
    # this is not a full comparison, but it's fine anyway
    assert len(list(sdata0._gen_elements())) == len(list(sdata1._gen_elements()))
    assert set(sdata0.coordinate_systems) == set(sdata1.coordinate_systems)
    _assert_elements_left_to_right_seem_identical(sdata0, sdata1)
    _assert_elements_left_to_right_seem_identical(sdata1, sdata0)
    _assert_tables_seem_identical(sdata0.table, sdata1.table)


def test_filter_by_coordinate_system(full_sdata):
    sdata = full_sdata.filter_by_coordinate_system(coordinate_system="global", filter_table=False)
    _assert_spatialdata_objects_seem_identical(sdata, full_sdata)

    scale = Scale([2.0], axes=("x",))
    set_transformation(full_sdata.images["image2d"], scale, "my_space0")
    set_transformation(full_sdata.shapes["circles"], Identity(), "my_space0")
    set_transformation(full_sdata.shapes["poly"], Identity(), "my_space1")

    sdata_my_space = full_sdata.filter_by_coordinate_system(coordinate_system="my_space0", filter_table=False)
    assert len(list(sdata_my_space._gen_elements())) == 2
    _assert_tables_seem_identical(sdata_my_space.table, full_sdata.table)

    sdata_my_space1 = full_sdata.filter_by_coordinate_system(
        coordinate_system=["my_space0", "my_space1", "my_space2"], filter_table=False
    )
    assert len(list(sdata_my_space1._gen_elements())) == 3


def test_filter_by_coordinate_system_also_table(full_sdata):
    from spatialdata.models import TableModel

    rng = np.random.default_rng(seed=0)
    full_sdata.table.obs["annotated_shapes"] = rng.choice(["circles", "poly"], size=full_sdata.table.shape[0])
    adata = full_sdata.table
    del adata.uns[TableModel.ATTRS_KEY]
    del full_sdata.table
    full_sdata.table = TableModel.parse(
        adata, region=["circles", "poly"], region_key="annotated_shapes", instance_key="instance_id"
    )

    scale = Scale([2.0], axes=("x",))
    set_transformation(full_sdata.shapes["circles"], scale, "my_space0")
    set_transformation(full_sdata.shapes["poly"], scale, "my_space1")

    filtered_sdata0 = full_sdata.filter_by_coordinate_system(coordinate_system="my_space0")
    filtered_sdata1 = full_sdata.filter_by_coordinate_system(coordinate_system="my_space1")
    filtered_sdata2 = full_sdata.filter_by_coordinate_system(coordinate_system="my_space0", filter_table=False)

    assert len(filtered_sdata0.table) + len(filtered_sdata1.table) == len(full_sdata.table)
    assert len(filtered_sdata2.table) == len(full_sdata.table)


def test_concatenate_tables():
    """
    The concatenation uses AnnData.concatenate(), here we test the
    concatenation result on region, region_key, instance_key
    """
    table0 = _get_table(region="shapes/circles", instance_key="instance_id")
    table1 = _get_table(region="shapes/poly", instance_key="instance_id")
    table2 = _get_table(region="shapes/poly2", instance_key="instance_id")
    with pytest.raises(ValueError):
        _concatenate_tables([])
    assert len(_concatenate_tables([table0])) == len(table0)
    assert len(_concatenate_tables([table0, table1, table2])) == len(table0) + len(table1) + len(table2)

    table0.obs["annotated_element_merged"] = np.arange(len(table0))
    c0 = _concatenate_tables([table0, table1])
    assert len(c0) == len(table0) + len(table1)

    d = c0.uns[TableModel.ATTRS_KEY]
    d["region"] = sorted(d["region"])
    assert d == {
        "region": ["shapes/circles", "shapes/poly"],
        "region_key": "region",
        "instance_key": "instance_id",
    }

    table3 = _get_table(region="shapes/circles", region_key="annotated_shapes_other", instance_key="instance_id")
    with pytest.raises(ValueError):
        _concatenate_tables([table0, table3], region_key="region")

    table4 = _get_table(
        region=["shapes/circles1", "shapes/poly1"], region_key="annotated_shape0", instance_key="instance_id"
    )
    table5 = _get_table(
        region=["shapes/circles2", "shapes/poly2"], region_key="annotated_shape0", instance_key="instance_id"
    )
    table6 = _get_table(
        region=["shapes/circles3", "shapes/poly3"], region_key="annotated_shape1", instance_key="instance_id"
    )
    with pytest.raises(ValueError, match="`region_key` must be specified if tables have different region keys"):
        _concatenate_tables([table4, table5, table6])
    assert len(_concatenate_tables([table4, table5, table6], region_key="region")) == len(table4) + len(table5) + len(
        table6
    )


def test_concatenate_sdatas(full_sdata):
    with pytest.raises(KeyError):
        concatenate([full_sdata, SpatialData(images={"image2d": full_sdata.images["image2d"]})])
    with pytest.raises(KeyError):
        concatenate([full_sdata, SpatialData(labels={"labels2d": full_sdata.labels["labels2d"]})])
    with pytest.raises(KeyError):
        concatenate([full_sdata, SpatialData(points={"points_0": full_sdata.points["points_0"]})])
    with pytest.raises(KeyError):
        concatenate([full_sdata, SpatialData(shapes={"circles": full_sdata.shapes["circles"]})])

    assert concatenate([full_sdata, SpatialData()]).table is not None

    set_transformation(full_sdata.shapes["circles"], Identity(), "my_space0")
    set_transformation(full_sdata.shapes["poly"], Identity(), "my_space1")
    filtered = full_sdata.filter_by_coordinate_system(coordinate_system=["my_space0", "my_space1"], filter_table=False)
    assert len(list(filtered._gen_elements())) == 2
    filtered0 = filtered.filter_by_coordinate_system(coordinate_system="my_space0", filter_table=False)
    filtered1 = filtered.filter_by_coordinate_system(coordinate_system="my_space1", filter_table=False)
    # this is needed cause we can't handle regions with same name.
    # TODO: fix this
    new_region = "sample2"
    table_new = filtered1.table.copy()
    del filtered1.table
    filtered1.table = table_new
    filtered1.table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] = new_region
    filtered1.table.obs[filtered1.table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]] = new_region
    concatenated = concatenate([filtered0, filtered1])
    assert len(list(concatenated._gen_elements())) == 2


def test_locate_spatial_element(full_sdata):
    assert full_sdata._locate_spatial_element(full_sdata.images["image2d"]) == ("image2d", "images")
    im = full_sdata.images["image2d"]
    del full_sdata.images["image2d"]
    with pytest.raises(ValueError, match="Element not found in the SpatialData object."):
        full_sdata._locate_spatial_element(im)
    full_sdata.images["image2d"] = im
    full_sdata.images["image2d_again"] = im
    with pytest.raises(ValueError):
        full_sdata._locate_spatial_element(im)


def test_get_item(points):
    assert id(points["points_0"]) == id(points.points["points_0"])

    # removed this test after this change: https://github.com/scverse/spatialdata/pull/145#discussion_r1133122720
    # to be uncommented/removed/modified after this is closed: https://github.com/scverse/spatialdata/issues/186
    # # this should be illegal: https://github.com/scverse/spatialdata/issues/176
    # points.images["points_0"] = Image2DModel.parse(np.array([[[1]]]), dims=("c", "y", "x"))
    # with pytest.raises(AssertionError):
    #     _ = points["points_0"]

    with pytest.raises(KeyError):
        _ = points["not_present"]


def test_set_item(full_sdata):
    for name in ["image2d", "labels2d", "points_0", "circles", "poly"]:
        full_sdata[name + "_again"] = full_sdata[name]
        with pytest.raises(KeyError):
            full_sdata[name] = full_sdata[name]
    with tempfile.TemporaryDirectory() as tmpdir:
        full_sdata.write(Path(tmpdir) / "test.zarr")
        for name in ["image2d", "labels2d", "points_0"]:
            # trying to overwrite the file used for backing (only for images, labels and points)
            with pytest.raises(ValueError):
                full_sdata[name] = full_sdata[name]
