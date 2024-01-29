from __future__ import annotations

import math

import numpy as np
import pytest
from anndata import AnnData
from dask.dataframe.core import DataFrame as DaskDataFrame
from dask.delayed import Delayed
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
from spatialdata._core.concatenate import _concatenate_tables, concatenate
from spatialdata._core.operations._utils import transform_to_data_extent
from spatialdata._core.spatialdata import SpatialData
from spatialdata.datasets import blobs
from spatialdata.models import (
    Image2DModel,
    Labels2DModel,
    PointsModel,
    ShapesModel,
    TableModel,
)
from spatialdata.transformations.operations import get_transformation, set_transformation
from spatialdata.transformations.transformations import Affine, Identity, Scale, Sequence, Translation

from tests.conftest import _get_table


def test_element_names_unique() -> None:
    shapes = ShapesModel.parse(np.array([[0, 0]]), geometry=0, radius=1)
    points = PointsModel.parse(np.array([[0, 0]]))
    labels = Labels2DModel.parse(np.array([[0, 0], [0, 0]]), dims=["y", "x"])
    image = Image2DModel.parse(np.array([[[0, 0], [0, 0]]]), dims=["c", "y", "x"])

    with pytest.raises(KeyError):
        SpatialData(images={"image": image}, points={"image": points})
    with pytest.raises(KeyError):
        SpatialData(images={"image": image}, shapes={"image": shapes})
    with pytest.raises(KeyError):
        SpatialData(images={"image": image}, labels={"image": labels})

    sdata = SpatialData(
        images={"image": image}, points={"points": points}, shapes={"shapes": shapes}, labels={"labels": labels}
    )

    # add elements with the same name
    # of element of same type
    with pytest.warns(UserWarning):
        sdata.images["image"] = image
    with pytest.warns(UserWarning):
        sdata.points["points"] = points
    with pytest.warns(UserWarning):
        sdata.shapes["shapes"] = shapes
    with pytest.warns(UserWarning):
        sdata.labels["labels"] = labels

    # add elements with the same name
    # of element of different type
    with pytest.raises(KeyError):
        sdata.images["points"] = image
    with pytest.raises(KeyError):
        sdata.images["shapes"] = image
    with pytest.raises(KeyError):
        sdata.labels["points"] = labels
    with pytest.raises(KeyError):
        sdata.points["shapes"] = points
    with pytest.raises(KeyError):
        sdata.shapes["labels"] = shapes

    assert sdata["image"].shape == image.shape
    assert sdata["labels"].shape == labels.shape
    assert len(sdata["points"]) == len(points)
    assert sdata["shapes"].shape == shapes.shape

    # add elements with the same name, test only couples of elements
    with pytest.raises(KeyError):
        sdata["labels"] = image
    with pytest.warns(UserWarning):
        sdata["points"] = points

    # this should not raise warnings because it's a different (new) name
    sdata["image2"] = image

    # test replacing complete attribute
    sdata = SpatialData(
        images={"image": image}, points={"points": points}, shapes={"shapes": shapes}, labels={"labels": labels}
    )
    # test for images
    sdata.images = {"image2": image}
    assert set(sdata.images.keys()) == {"image2"}
    assert "image2" in sdata._shared_keys
    assert "image" not in sdata._shared_keys
    # test for labels
    sdata.labels = {"labels2": labels}
    assert set(sdata.labels.keys()) == {"labels2"}
    assert "labels2" in sdata._shared_keys
    assert "labels" not in sdata._shared_keys
    # test for points
    sdata.points = {"points2": points}
    assert set(sdata.points.keys()) == {"points2"}
    assert "points2" in sdata._shared_keys
    assert "points" not in sdata._shared_keys
    # test for points
    sdata.shapes = {"shapes2": shapes}
    assert set(sdata.shapes.keys()) == {"shapes2"}
    assert "shapes2" in sdata._shared_keys
    assert "shapes" not in sdata._shared_keys


def _assert_elements_left_to_right_seem_identical(sdata0: SpatialData, sdata1: SpatialData) -> None:
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


def _assert_tables_seem_identical(table0: AnnData | None, table1: AnnData | None) -> None:
    assert table0 is None and table1 is None or table0.shape == table1.shape


def _assert_spatialdata_objects_seem_identical(sdata0: SpatialData, sdata1: SpatialData) -> None:
    # this is not a full comparison, but it's fine anyway
    assert len(list(sdata0._gen_elements())) == len(list(sdata1._gen_elements()))
    assert set(sdata0.coordinate_systems) == set(sdata1.coordinate_systems)
    _assert_elements_left_to_right_seem_identical(sdata0, sdata1)
    _assert_elements_left_to_right_seem_identical(sdata1, sdata0)
    _assert_tables_seem_identical(sdata0.table, sdata1.table)


def test_filter_by_coordinate_system(full_sdata: SpatialData) -> None:
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


def test_filter_by_coordinate_system_also_table(full_sdata: SpatialData) -> None:
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


def test_rename_coordinate_systems(full_sdata: SpatialData) -> None:
    # all the elements point to global, add new coordinate systems
    set_transformation(
        element=full_sdata.shapes["circles"], transformation=Identity(), to_coordinate_system="my_space0"
    )
    set_transformation(element=full_sdata.shapes["poly"], transformation=Identity(), to_coordinate_system="my_space1")
    set_transformation(
        element=full_sdata.shapes["multipoly"], transformation=Identity(), to_coordinate_system="my_space2"
    )

    elements_in_global_before = {
        name for _, name, _ in full_sdata.filter_by_coordinate_system("global")._gen_elements()
    }

    # test a renaming without collisions
    full_sdata.rename_coordinate_systems({"my_space0": "my_space00", "my_space1": "my_space11"})
    assert {"my_space00", "my_space11", "global", "my_space2"}.issubset(full_sdata.coordinate_systems)
    assert "my_space0" not in full_sdata.coordinate_systems
    assert "my_space1" not in full_sdata.coordinate_systems

    # renaming with collisions (my_space2 already exists)
    with pytest.raises(ValueError):
        full_sdata.rename_coordinate_systems({"my_space00": "my_space2"})

    # renaming with collisions (my_space3 doesn't exist but it's target of two renamings)
    with pytest.raises(ValueError):
        full_sdata.rename_coordinate_systems({"my_space00": "my_space3", "my_space11": "my_space3"})

    # invalid renaming: my_space3 is not a valid coordinate system
    with pytest.raises(ValueError):
        full_sdata.rename_coordinate_systems({"my_space3": "my_space4"})

    # invalid renaming: my_space3 is not a valid coordinate system (it doesn't matter if my_space3 is target of one
    # renaming, as it doesn't exist at the time of the function call)
    with pytest.raises(ValueError):
        full_sdata.rename_coordinate_systems(
            {"my_space00": "my_space3", "my_space11": "my_space3", "my_space3": "my_space4"}
        )

    # valid renaming with collisions
    full_sdata.rename_coordinate_systems({"my_space00": "my_space2", "my_space2": "my_space3"})
    assert get_transformation(full_sdata.shapes["circles"], get_all=True)["my_space2"] == Identity()
    assert get_transformation(full_sdata.shapes["multipoly"], get_all=True)["my_space3"] == Identity()

    # renaming without effect
    full_sdata.rename_coordinate_systems({"my_space11": "my_space11"})
    assert get_transformation(full_sdata.shapes["poly"], get_all=True)["my_space11"] == Identity()

    # check that all the elements with coordinate system global are still there
    elements_in_global_after = {name for _, name, _ in full_sdata.filter_by_coordinate_system("global")._gen_elements()}
    assert elements_in_global_before == elements_in_global_after


def test_concatenate_tables() -> None:
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


def test_concatenate_sdatas(full_sdata: SpatialData) -> None:
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


def test_locate_spatial_element(full_sdata: SpatialData) -> None:
    assert full_sdata.locate_element(full_sdata.images["image2d"])[0] == "images/image2d"
    im = full_sdata.images["image2d"]
    del full_sdata.images["image2d"]
    assert len(full_sdata.locate_element(im)) == 0
    full_sdata.images["image2d"] = im
    full_sdata.images["image2d_again"] = im
    paths = full_sdata.locate_element(im)
    assert len(paths) == 2


def test_get_item(points: SpatialData) -> None:
    assert id(points["points_0"]) == id(points.points["points_0"])

    # removed this test after this change: https://github.com/scverse/spatialdata/pull/145#discussion_r1133122720
    # to be uncommented/removed/modified after this is closed: https://github.com/scverse/spatialdata/issues/186
    # # this should be illegal: https://github.com/scverse/spatialdata/issues/176
    # points.images["points_0"] = Image2DModel.parse(np.array([[[1]]]), dims=("c", "y", "x"))
    # with pytest.raises(AssertionError):
    #     _ = points["points_0"]

    with pytest.raises(KeyError):
        _ = points["not_present"]


def test_set_item(full_sdata: SpatialData) -> None:
    for name in ["image2d", "labels2d", "points_0", "circles", "poly"]:
        full_sdata[name + "_again"] = full_sdata[name]
        with pytest.warns(UserWarning):
            full_sdata[name] = full_sdata[name]


def test_no_shared_transformations() -> None:
    """Test transformation dictionary copy for transformations not to be shared."""
    sdata = blobs()
    element_name = "blobs_image"
    test_space = "test"
    set_transformation(sdata.images[element_name], Identity(), to_coordinate_system=test_space)

    gen = sdata._gen_elements()
    for _, name, obj in gen:
        if name != element_name:
            assert test_space not in get_transformation(obj, get_all=True)
        else:
            assert test_space in get_transformation(obj, get_all=True)


def test_init_from_elements(full_sdata: SpatialData) -> None:
    all_elements = {name: el for _, name, el in full_sdata._gen_elements()}
    sdata = SpatialData.init_from_elements(all_elements, table=full_sdata.table)
    for element_type in ["images", "labels", "points", "shapes"]:
        assert set(getattr(sdata, element_type).keys()) == set(getattr(full_sdata, element_type).keys())


def test_subset(full_sdata: SpatialData) -> None:
    element_names = ["image2d", "labels2d", "points_0", "circles", "poly"]
    subset0 = full_sdata.subset(element_names)
    unique_names = set()
    for _, k, _ in subset0._gen_elements():
        unique_names.add(k)
    assert "image3d_xarray" in full_sdata.images
    assert unique_names == set(element_names)
    assert subset0.table is None

    adata = AnnData(
        shape=(10, 0),
        obs={"region": ["circles"] * 5 + ["poly"] * 5, "instance_id": [0, 1, 2, 3, 4, "a", "b", "c", "d", "e"]},
    )
    del full_sdata.table
    full_sdata.table = TableModel.parse(
        adata, region=["circles", "poly"], region_key="region", instance_key="instance_id"
    )
    subset1 = full_sdata.subset(["poly"])
    assert subset1.table is not None
    assert len(subset1.table) == 5
    assert subset1.table.obs["region"].unique().tolist() == ["poly"]


def test_transform_to_data_extent(full_sdata: SpatialData):
    theta = math.pi / 6
    rotation = Affine(
        [
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta), math.cos(theta), 0],
            [0, 0, 1],
        ],
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )
    scale = Scale([2.0], axes=("x",))
    translation = Translation([-100.0, 200.0], axes=("x", "y"))
    sequence = Sequence([rotation, scale, translation])
    for el in full_sdata._gen_elements_values():
        set_transformation(el, sequence, "global")
    full_sdata = full_sdata.subset(
        ["image2d", "image2d_multiscale", "labels2d", "labels2d_multiscale", "points_0", "circles", "multipoly", "poly"]
    )
    sdata = transform_to_data_extent(full_sdata, "global", target_width=1000)
