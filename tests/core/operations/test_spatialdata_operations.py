import math

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from geopandas import GeoDataFrame

from spatialdata._core.concatenate import _concatenate_tables, concatenate
from spatialdata._core.data_extent import are_extents_equal, get_extent
from spatialdata._core.operations._utils import transform_to_data_extent
from spatialdata._core.spatialdata import SpatialData
from spatialdata._types import ArrayLike
from spatialdata.datasets import blobs
from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, ShapesModel, TableModel, get_table_keys
from spatialdata.testing import assert_elements_dict_are_identical, assert_spatial_data_objects_are_identical
from spatialdata.transformations.operations import get_transformation, set_transformation
from spatialdata.transformations.transformations import (
    Affine,
    BaseTransformation,
    Identity,
    Scale,
    Sequence,
    Translation,
)
from tests.conftest import _get_shapes, _get_table


def test_element_names_unique() -> None:
    shapes = ShapesModel.parse(np.array([[0, 0]]), geometry=0, radius=1)
    points = PointsModel.parse(np.array([[0, 0]]))
    labels = Labels2DModel.parse(np.array([[0, 0], [0, 0]]), dims=["y", "x"])
    image = Image2DModel.parse(np.array([[[0, 0], [0, 0]]]), dims=["c", "y", "x"])
    table = TableModel.parse(AnnData(shape=(1, 0)))

    with pytest.raises(KeyError):
        SpatialData(images={"image": image}, points={"image": points})
    with pytest.raises(KeyError):
        SpatialData(images={"image": image}, shapes={"image": shapes})
    with pytest.raises(KeyError):
        SpatialData(images={"image": image}, labels={"image": labels})
    with pytest.raises(KeyError):
        SpatialData(images={"image": image}, labels={"image": table})

    sdata = SpatialData(
        images={"image": image},
        points={"points": points},
        shapes={"shapes": shapes},
        labels={"labels": labels},
        tables={"table": table},
    )

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
    with pytest.raises(KeyError):
        sdata.tables["labels"] = table

    # add elements with the case-variant of an existing name
    # of element of same type
    with pytest.raises(KeyError):
        sdata.images["Image"] = image
    with pytest.raises(KeyError):
        sdata.points["POINTS"] = points
    with pytest.raises(KeyError):
        sdata.shapes["Shapes"] = shapes
    with pytest.raises(KeyError):
        sdata.labels["Labels"] = labels
    with pytest.raises(KeyError):
        sdata.tables["Table"] = table

    assert sdata["image"].shape == image.shape
    assert sdata["labels"].shape == labels.shape
    assert len(sdata["points"]) == len(points)
    assert sdata["shapes"].shape == shapes.shape
    assert len(sdata["table"]) == len(table)

    # add elements with the same name, test only couples of elements
    with pytest.raises(KeyError):
        sdata["labels"] = image

    # this should not raise warnings because it's a different (new) name
    sdata["image2"] = image

    # test replacing complete attribute
    sdata = SpatialData(
        images={"image": image},
        points={"points": points},
        shapes={"shapes": shapes},
        labels={"labels": labels},
        tables={"table": table},
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
    # test for shapes
    sdata.shapes = {"shapes2": shapes}
    assert set(sdata.shapes.keys()) == {"shapes2"}
    assert "shapes2" in sdata._shared_keys
    assert "shapes" not in sdata._shared_keys
    # test for tables
    sdata.tables = {"table2": table}
    assert set(sdata.tables.keys()) == {"table2"}
    assert "table2" in sdata._shared_keys
    assert "table" not in sdata._shared_keys


def test_element_type_from_element_name(points: SpatialData) -> None:
    with pytest.raises(ValueError, match="not found in SpatialData object."):
        points._element_type_from_element_name("invalid")
    points["copy"] = points["points_0"]
    assert points._element_type_from_element_name("points_0") == "points"


def test_filter_by_coordinate_system(full_sdata: SpatialData) -> None:
    sdata = full_sdata.filter_by_coordinate_system(coordinate_system="global", filter_tables=False)
    assert_spatial_data_objects_are_identical(sdata, full_sdata)

    scale = Scale([2.0], axes=("x",))
    set_transformation(full_sdata.images["image2d"], scale, "my_space0")
    set_transformation(full_sdata.shapes["circles"], Identity(), "my_space0")
    set_transformation(full_sdata.shapes["poly"], Identity(), "my_space1")

    sdata_my_space = full_sdata.filter_by_coordinate_system(coordinate_system="my_space0", filter_tables=False)
    assert len(list(sdata_my_space.gen_elements())) == 3
    assert_elements_dict_are_identical(sdata_my_space.tables, full_sdata.tables)

    sdata_my_space1 = full_sdata.filter_by_coordinate_system(
        coordinate_system=["my_space0", "my_space1", "my_space2"], filter_tables=False
    )
    assert len(list(sdata_my_space1.gen_elements())) == 4


def test_filter_by_coordinate_system_also_table(full_sdata: SpatialData) -> None:
    from spatialdata.models import TableModel

    rng = np.random.default_rng(seed=0)
    full_sdata["table"].obs["annotated_shapes"] = pd.Categorical(
        rng.choice(["circles", "poly"], size=full_sdata["table"].shape[0])
    )
    adata = full_sdata["table"]
    del adata.uns[TableModel.ATTRS_KEY]
    full_sdata["table"] = TableModel.parse(
        adata,
        region=["circles", "poly"],
        region_key="annotated_shapes",
        instance_key="instance_id",
    )

    scale = Scale([2.0], axes=("x",))
    set_transformation(full_sdata.shapes["circles"], scale, "my_space0")
    set_transformation(full_sdata.shapes["poly"], scale, "my_space1")

    filtered_sdata0 = full_sdata.filter_by_coordinate_system(coordinate_system="my_space0")
    filtered_sdata1 = full_sdata.filter_by_coordinate_system(coordinate_system="my_space1")
    filtered_sdata2 = full_sdata.filter_by_coordinate_system(coordinate_system="my_space0", filter_tables=False)

    assert len(filtered_sdata0["table"]) + len(filtered_sdata1["table"]) == len(full_sdata["table"])
    assert len(filtered_sdata2["table"]) == len(full_sdata["table"])


def test_rename_coordinate_systems(full_sdata: SpatialData) -> None:
    # all the elements point to global, add new coordinate systems
    set_transformation(
        element=full_sdata.shapes["circles"],
        transformation=Identity(),
        to_coordinate_system="my_space0",
    )
    set_transformation(
        element=full_sdata.shapes["poly"],
        transformation=Identity(),
        to_coordinate_system="my_space1",
    )
    set_transformation(
        element=full_sdata.shapes["multipoly"],
        transformation=Identity(),
        to_coordinate_system="my_space2",
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
            {
                "my_space00": "my_space3",
                "my_space11": "my_space3",
                "my_space3": "my_space4",
            }
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

    table3 = _get_table(
        region="shapes/circles",
        region_key="annotated_shapes_other",
        instance_key="instance_id",
    )
    with pytest.raises(ValueError):
        _concatenate_tables([table0, table3], region_key="region")

    table4 = _get_table(
        region=["shapes/circles1", "shapes/poly1"],
        region_key="annotated_shape0",
        instance_key="instance_id",
    )
    table5 = _get_table(
        region=["shapes/circles2", "shapes/poly2"],
        region_key="annotated_shape0",
        instance_key="instance_id",
    )
    table6 = _get_table(
        region=["shapes/circles3", "shapes/poly3"],
        region_key="annotated_shape1",
        instance_key="instance_id",
    )
    with pytest.raises(
        ValueError,
        match="`region_key` must be specified if tables have different region keys",
    ):
        _concatenate_tables([table4, table5, table6])
    assert len(_concatenate_tables([table4, table5, table6], region_key="region")) == len(table4) + len(table5) + len(
        table6
    )


def test_concatenate_custom_table_metadata() -> None:
    # test for https://github.com/scverse/spatialdata/issues/349
    shapes0 = _get_shapes()
    shapes1 = _get_shapes()
    n = len(shapes0["poly"])
    table0 = TableModel.parse(
        AnnData(obs={"my_region": pd.Categorical(["poly0"] * n), "my_instance_id": list(range(n))}),
        region="poly0",
        region_key="my_region",
        instance_key="my_instance_id",
    )
    table1 = TableModel.parse(
        AnnData(obs={"my_region": pd.Categorical(["poly1"] * n), "my_instance_id": list(range(n))}),
        region="poly1",
        region_key="my_region",
        instance_key="my_instance_id",
    )
    sdata0 = SpatialData.init_from_elements({"poly0": shapes0["poly"], "table": table0})
    sdata1 = SpatialData.init_from_elements({"poly1": shapes1["poly"], "table": table1})
    sdata = concatenate([sdata0, sdata1], concatenate_tables=True)
    assert len(sdata["table"]) == 2 * n


def test_concatenate_sdatas(full_sdata: SpatialData) -> None:
    with pytest.raises(KeyError):
        concatenate([full_sdata, SpatialData(images={"image2d": full_sdata.images["image2d"]})])
    with pytest.raises(KeyError):
        concatenate(
            [
                full_sdata,
                SpatialData(labels={"labels2d": full_sdata.labels["labels2d"]}),
            ]
        )
    with pytest.raises(KeyError):
        concatenate(
            [
                full_sdata,
                SpatialData(points={"points_0": full_sdata.points["points_0"]}),
            ]
        )
    with pytest.raises(KeyError):
        concatenate([full_sdata, SpatialData(shapes={"circles": full_sdata.shapes["circles"]})])

    assert concatenate([full_sdata, SpatialData()])["table"] is not None

    set_transformation(full_sdata.shapes["circles"], Identity(), "my_space0")
    set_transformation(full_sdata.shapes["poly"], Identity(), "my_space1")
    filtered = full_sdata.filter_by_coordinate_system(coordinate_system=["my_space0", "my_space1"], filter_tables=False)
    assert len(list(filtered.gen_elements())) == 3
    filtered0 = filtered.filter_by_coordinate_system(coordinate_system="my_space0", filter_tables=False)
    filtered1 = filtered.filter_by_coordinate_system(coordinate_system="my_space1", filter_tables=False)
    # this is needed cause we can't handle regions with same name.
    # TODO: fix this
    new_region = "sample2"
    table_new = filtered1["table"].copy()
    filtered1["table"] = table_new
    filtered1["table"].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] = new_region
    filtered1["table"].obs[filtered1["table"].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]] = new_region
    concatenated = concatenate([filtered0, filtered1], concatenate_tables=True)
    assert len(list(concatenated.gen_elements())) == 3


@pytest.mark.parametrize("concatenate_tables", [True, False])
@pytest.mark.parametrize("obs_names_make_unique", [True, False])
def test_concatenate_sdatas_from_iterable(concatenate_tables: bool, obs_names_make_unique: bool) -> None:
    sdata0 = blobs()
    sdata1 = blobs()

    sdatas = {"sample0": sdata0, "sample1": sdata1}
    with pytest.raises(KeyError, match="Images must have unique names across the SpatialData objects"):
        _ = concatenate(
            sdatas.values(),
            concatenate_tables=concatenate_tables,
            obs_names_make_unique=obs_names_make_unique,
        )
    merged = concatenate(
        sdatas,
        obs_names_make_unique=obs_names_make_unique,
        concatenate_tables=concatenate_tables,
    )

    if concatenate_tables:
        assert len(merged.tables) == 1
        table = merged["table"]
        if obs_names_make_unique:
            assert table.obs_names[0] == "1-sample0"
            assert table.obs_names[-1] == "30-sample1"
        else:
            assert table.obs_names[0] == "1"
    else:
        assert merged["table-sample0"].obs_names[0] == "1"
    assert sdata0["table"].obs_names[0] == "1"


def test_concatenate_two_tables_each_annotating_two_elements() -> None:
    # let's define 4 polygon elements. Two of them are annotated by the first table and the other two by the second
    # table. The two tables have the same region key and instance key.
    def _get_table_and_poly(i: int) -> tuple[AnnData, GeoDataFrame]:
        poly = _get_shapes()["poly"]
        n = len(poly)
        region = f"poly{i}"
        table = TableModel.parse(
            AnnData(
                obs=pd.DataFrame(
                    {"region": pd.Categorical([region] * n), "instance_id": list(range(n))},
                    index=[f"{i}" for i in range(n)],
                )
            ),
            region=region,
            region_key="region",
            instance_key="instance_id",
        )
        return table, poly

    table0_a, poly0_a = _get_table_and_poly(0)
    table1_a, poly1_a = _get_table_and_poly(1)
    table0_b, poly0_b = _get_table_and_poly(0)
    table1_b, poly1_b = _get_table_and_poly(1)

    table01_a = _concatenate_tables([table0_a, table1_a])
    table01_b = _concatenate_tables([table0_b, table1_b])

    sdata01_a = SpatialData.init_from_elements({"poly0": poly0_a, "poly1": poly1_a, "table": table01_a})
    sdata01_b = SpatialData.init_from_elements({"poly0": poly0_b, "poly1": poly1_b, "table": table01_b})
    sdata = concatenate({"a": sdata01_a, "b": sdata01_b}, concatenate_tables=True)
    region, _, _ = get_table_keys(sdata["table"])
    assert region == ["poly0-a", "poly1-a", "poly0-b", "poly1-b"]
    assert len(sdata["table"]) == 4 * len(poly0_a)


def test_concatenate_sdatas_single_item() -> None:
    sdata = blobs()

    def _n_elements(sdata: SpatialData) -> int:
        return len([0 for _, _, _ in sdata.gen_elements()])

    n = _n_elements(sdata)
    assert n == _n_elements(concatenate([sdata]))
    assert n == _n_elements(concatenate({"sample": sdata}.values()))
    c = concatenate({"sample": sdata})
    assert n == _n_elements(c)
    assert "blobs_image-sample" in c.images


@pytest.mark.parametrize("merge_coordinate_systems_on_name", [True, False])
def test_concatenate_merge_coordinate_systems_on_name(merge_coordinate_systems_on_name):
    blob1 = blobs()
    blob2 = blobs()

    if merge_coordinate_systems_on_name:
        with pytest.raises(
            ValueError,
            match="`merge_coordinate_systems_on_name` can only be used if `sdatas` is a dictionary",
        ):
            concatenate((blob1, blob2), merge_coordinate_systems_on_name=merge_coordinate_systems_on_name)

    sdata_keys = ["blob1", "blob2"]
    sdata = concatenate(
        dict(zip(sdata_keys, [blob1, blob2], strict=True)),
        merge_coordinate_systems_on_name=merge_coordinate_systems_on_name,
    )

    if merge_coordinate_systems_on_name:
        assert set(sdata.coordinate_systems) == {"global"}
    else:
        assert set(sdata.coordinate_systems) == {"global-blob1", "global-blob2"}

    # extra checks not specific to this test, we could remove them or leave them just
    # in case
    expected_images = ["blobs_image", "blobs_multiscale_image"]
    expected_labels = ["blobs_labels", "blobs_multiscale_labels"]
    expected_points = ["blobs_points"]
    expected_shapes = ["blobs_circles", "blobs_polygons", "blobs_multipolygons"]

    expected_suffixed_images = [f"{name}-{key}" for key in sdata_keys for name in expected_images]
    expected_suffixed_labels = [f"{name}-{key}" for key in sdata_keys for name in expected_labels]
    expected_suffixed_points = [f"{name}-{key}" for key in sdata_keys for name in expected_points]
    expected_suffixed_shapes = [f"{name}-{key}" for key in sdata_keys for name in expected_shapes]

    assert set(sdata.images.keys()) == set(expected_suffixed_images)
    assert set(sdata.labels.keys()) == set(expected_suffixed_labels)
    assert set(sdata.points.keys()) == set(expected_suffixed_points)
    assert set(sdata.shapes.keys()) == set(expected_suffixed_shapes)


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
    assert points["points_0"] is points.points["points_0"]

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


def test_del_item(full_sdata: SpatialData) -> None:
    for name in ["image2d", "labels2d", "points_0", "circles", "poly"]:
        del full_sdata[name]
        with pytest.raises(KeyError):
            del full_sdata[name]
    with pytest.raises(KeyError, match="Could not find element with name"):
        _ = full_sdata["not_present"]


def test_no_shared_transformations() -> None:
    """Test transformation dictionary copy for transformations not to be shared."""
    sdata = blobs()
    element_name = "blobs_image"
    test_space = "test"
    set_transformation(sdata.images[element_name], Identity(), to_coordinate_system=test_space)

    gen = sdata._gen_elements()
    for element_type, name, obj in gen:
        if element_type != "tables":
            if name != element_name:
                assert test_space not in get_transformation(obj, get_all=True)
            else:
                assert test_space in get_transformation(obj, get_all=True)


def test_init_from_elements(full_sdata: SpatialData) -> None:
    # this first code block needs to be removed when the tables argument is removed from init_from_elements()
    all_elements = {name: el for _, name, el in full_sdata._gen_elements()}
    sdata = SpatialData.init_from_elements(all_elements | {"table": full_sdata["table"]})
    for element_type in ["images", "labels", "points", "shapes", "tables"]:
        assert set(getattr(sdata, element_type).keys()) == set(getattr(full_sdata, element_type).keys())

    all_elements = {name: el for _, name, el in full_sdata._gen_elements(include_tables=True)}
    sdata = SpatialData.init_from_elements(all_elements)
    for element_type in ["images", "labels", "points", "shapes", "tables"]:
        assert set(getattr(sdata, element_type).keys()) == set(getattr(full_sdata, element_type).keys())


def test_subset(full_sdata: SpatialData) -> None:
    element_names = ["image2d", "points_0", "circles", "poly"]
    subset0 = full_sdata.subset(element_names)
    unique_names = set()
    for _, k, _ in subset0.gen_spatial_elements():
        unique_names.add(k)
    assert "image3d_xarray" in full_sdata.images
    assert unique_names == set(element_names)
    # no table since the labels are not present in the subset
    assert "table" not in subset0.tables

    adata = AnnData(
        shape=(10, 0),
        obs={
            "region": pd.Categorical(["circles"] * 5 + ["poly"] * 5),
            "instance_id": [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
        },
    )
    sdata_table = TableModel.parse(
        adata,
        region=["circles", "poly"],
        region_key="region",
        instance_key="instance_id",
    )
    full_sdata["table"] = sdata_table
    full_sdata.tables["second_table"] = sdata_table
    subset1 = full_sdata.subset(["poly", "second_table"])
    assert subset1["table"] is not None
    assert len(subset1["table"]) == 5
    assert subset1["table"].obs["region"].unique().tolist() == ["poly"]
    assert len(subset1["second_table"]) == 10


@pytest.mark.parametrize("maintain_positioning", [True, False])
def test_transform_to_data_extent(full_sdata: SpatialData, maintain_positioning: bool) -> None:
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
    for el in full_sdata._gen_spatial_element_values():
        set_transformation(el, sequence, "global")
    elements = [
        "image2d",
        "image2d_multiscale",
        "labels2d",
        "labels2d_multiscale",
        "points_0",
        "circles",
        "multipoly",
        "poly",
    ]
    full_sdata = full_sdata.subset(elements)
    points = full_sdata["points_0"].compute()
    points["z"] = points["x"]
    points = PointsModel.parse(points)
    full_sdata["points_0_3d"] = points
    sdata = transform_to_data_extent(
        full_sdata,
        "global",
        target_width=1000,
        maintain_positioning=maintain_positioning,
    )

    first_a: ArrayLike | None = None
    for _, name, el in sdata.gen_spatial_elements():
        t = get_transformation(el, to_coordinate_system="global")
        assert isinstance(t, BaseTransformation)
        a = t.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
        if first_a is None:
            first_a = a
        else:
            # we are not pixel perfect because of this bug: https://github.com/scverse/spatialdata/issues/165
            if maintain_positioning and name in [
                "points_0_3d",
                "points_0",
                "poly",
                "circles",
                "multipoly",
            ]:
                # Again, due to the "pixel perfect" bug, the 0.5 translation forth and back in the z axis that is added
                # by rasterize() (like the one in the example belows), amplifies the error also for x and y beyond the
                # rtol threshold below. So, let's skip that check and to an absolute check up to 0.5 (due to the
                # half-pixel offset).
                # Sequence
                #     Translation (z, y, x)
                #         [-0.5 -0.5 -0.5]
                #     Scale (y, x)
                #         [0.17482681 0.17485125]
                #     Translation (y, x)
                #         [  -3.13652607 -164.        ]
                #     Translation (z, y, x)
                #         [0.5 0.5 0.5]
                assert np.allclose(a, first_a, atol=0.5)
            else:
                assert np.allclose(a, first_a, rtol=0.005)

    if not maintain_positioning:
        assert np.allclose(first_a, np.eye(3))
    else:
        data_extent_before = get_extent(full_sdata, coordinate_system="global")
        data_extent_after = get_extent(sdata, coordinate_system="global")
        # again, due to the "pixel perfect" bug, we use an absolute tolerance of 0.5
        assert are_extents_equal(data_extent_before, data_extent_after, atol=0.5)


def test_validate_table_in_spatialdata(full_sdata):
    table = full_sdata["table"]
    region, region_key, _ = get_table_keys(table)
    assert region == "labels2d"

    full_sdata.validate_table_in_spatialdata(table)

    # region not found
    del full_sdata.labels["labels2d"]
    with pytest.warns(UserWarning, match="in the SpatialData object"):
        full_sdata.validate_table_in_spatialdata(table)

    table.obs[region_key] = pd.Categorical(["points_0"] * table.n_obs)
    full_sdata.set_table_annotates_spatialelement("table", region="points_0")

    full_sdata.validate_table_in_spatialdata(table)

    # region not found
    del full_sdata.points["points_0"]
    with pytest.warns(UserWarning, match="in the SpatialData object"):
        full_sdata.validate_table_in_spatialdata(table)
