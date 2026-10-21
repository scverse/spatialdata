# pyright: standard

from __future__ import annotations

from pathlib import PurePosixPath

import numpy as np
import xarray as xr
import ome_zarr as oz
import ome_zarr_models.v06.coordinate_transforms as ozm06trans

from spatialdata._io.io_raster import ElementKey, FutureSpatialData, try_parse_ngff06_multiscale
from spatialdata.transformations.graph.edge import ScaleEdge
from spatialdata.transformations.graph.vert import NgffCsPath
from tests.conftest import SEED


def test_parse_multiscale():
    full_shape = (256, 256)
    rng = np.random.default_rng(seed=SEED)

    data = rng.random(full_shape)
    omz_img = oz.OMEZarrImage(data=data, axes="yx")

    labels_name = "my_labels"
    label_data = oz.OMEZarrImage(
        data=rng.integers(0, 256, size=(256, 256), dtype=np.uint8), axes="yx", name=labels_name
    )
    omz_label = oz.OMEZarrLabels(
        image=label_data,
    )

    scale_factors = (2, 4, 8, 16)
    ms = oz.OMEZarrMultiscale(
        image=omz_img,
        scale_factors=scale_factors,
        default_coordinate_system_name="intrinsic_cs",
        coordinate_systems=[
            ozm06trans.CoordinateSystem(
                name="another_cs",
                axes=(
                    ozm06trans.Axis(name="i", type="space"),
                    ozm06trans.Axis(name="j", type="space"),
                ),
            )
        ],
        coordinate_transformations=(
            ozm06trans.Scale(
                scale=(10.0, 20.0),
                input=ozm06trans.CoordinateSystemIdentifier(name="intrinsic_cs"),
                output=ozm06trans.CoordinateSystemIdentifier(name="another_cs"),
            ),
        ),
        # default_coordinate_system_name="physical",
        labels=[omz_label],
    )

    sdata = try_parse_ngff06_multiscale(group_path=PurePosixPath("."), multiscale=ms)

    base_path = PurePosixPath(".")
    intrinsic_cs = sdata.find_node(
        ngff_path=NgffCsPath(group_path=base_path, name="intrinsic_cs")
    )
    another_cs = sdata.find_node(
        ngff_path=NgffCsPath(group_path=base_path, name="another_cs")
    )
    transfs = list(sdata.edges().directed(fro=intrinsic_cs, to=another_cs))
    assert len(transfs) == 1
    scale = transfs.pop().data["transform"]
    assert isinstance(scale, ScaleEdge)
    assert scale.input == intrinsic_cs.coord_system
    assert scale.output == another_cs.coord_system
    np.testing.assert_equal(scale.scale, [10.0, 20.0])

    _ = sdata.find_node(
        ngff_path=NgffCsPath(group_path=base_path / "labels" / labels_name, name="physical")
    )

    # labels_data_tree = data_trees[base_path / "labels" / labels_name]
    image_data_tree = sdata.get_image(ElementKey(namespace=base_path, name=omz_img.name))
    assert image_data_tree is not None
    for xr_scale_node, ngff_scale, ngff_meta in zip(
        image_data_tree.children.values(), ms.images, ms.metadata.datasets, strict=True
    ):
        xr_scale = xr_scale_node["image"]
        assert not isinstance(xr_scale, xr.DataTree)
        assert xr_scale.shape == ngff_scale.data.shape

        coords_x = xr_scale.coords["x"]
        coords_y = xr_scale.coords["y"]

        seq = ngff_meta.coordinateTransformations[0]
        assert isinstance(seq, ozm06trans.Sequence)
        scale = seq.transformations[0]
        assert isinstance(scale, ozm06trans.Scale)
        translate = seq.transformations[1]
        assert isinstance(translate, ozm06trans.Translation)

        start_indices = (0, 0)
        start = (coords_x[0], coords_y[0])

        end_indices = (xr_scale.shape[0] - 1, xr_scale.shape[1] - 1)
        end = (coords_x[-1], coords_y[-1])

        for indices, point in [(start_indices, start), (end_indices, end)]:
            expected = np.asarray(scale.scale) * indices + translate.translation
            assert np.allclose(expected, point)

    sliced_data_tree = image_data_tree.sel(x=slice(0, 256), y=slice(0, 256), method="nearest")
    assert sliced_data_tree.equals(image_data_tree)

    quarter_tree = image_data_tree.sel(x=slice(0, 128), y=slice(0, 128), method="nearest")
    for xr_scale_node, scale_factor in zip(quarter_tree.children.values(), [1, *scale_factors], strict=True):
        xr_scale = xr_scale_node["image"]
        expected_shape = np.asarray(full_shape) / 2 / scale_factor
        np.testing.assert_equal(xr_scale.shape, expected_shape)

def test_dumping_multiscales_to_ngff():
    full_shape = (256, 256)
    rng = np.random.default_rng(seed=SEED)

    img_data1 = rng.random(full_shape)
    omz_img1 = oz.OMEZarrImage(data=img_data1, axes="yx")

    labels_name1 = "my_labels"
    label_data1 = oz.OMEZarrImage(
        data=rng.integers(0, 256, size=(256, 256), dtype=np.uint8), axes="yx", name=labels_name1
    )
    omz_label1 = oz.OMEZarrLabels(
        image=label_data1,
    )

    scale_factors = (2, 4, 8, 16)
    multiscales1 = oz.OMEZarrMultiscale(
        image=omz_img1,
        scale_factors=scale_factors,
        default_coordinate_system_name="intrinsic_cs",
        coordinate_systems=[
            ozm06trans.CoordinateSystem(
                name="another_cs",
                axes=(
                    ozm06trans.Axis(name="i", type="space"),
                    ozm06trans.Axis(name="j", type="space"),
                ),
            )
        ],
        coordinate_transformations=(
            ozm06trans.Scale(
                scale=(10.0, 20.0),
                input=ozm06trans.CoordinateSystemIdentifier(name="intrinsic_cs"),
                output=ozm06trans.CoordinateSystemIdentifier(name="another_cs"),
            ),
        ),
        # default_coordinate_system_name="physical",
        labels=[omz_label1],
    )


    img_data2 = rng.random((1, 256, 256))
    omz_img2 = oz.OMEZarrImage(data=img_data2, axes="cyx")
    multiscales2 = oz.OMEZarrMultiscale(
        image=omz_img2,
        scale_factors=[
            {"x": s, "y": s} for s in scale_factors
        ],
        default_coordinate_system_name="intrinsic_cs2",
        coordinate_systems=[
            ozm06trans.CoordinateSystem(
                name="another_cs2",
                axes=(
                    ozm06trans.Axis(name="c", type="channel"),
                    ozm06trans.Axis(name="i", type="space"),
                    ozm06trans.Axis(name="j", type="space"),
                ),
            )
        ],
        coordinate_transformations=(
            ozm06trans.Scale(
                scale=(10.0, 20.0, 1.0),
                input=ozm06trans.CoordinateSystemIdentifier(name="intrinsic_cs2"),
                output=ozm06trans.CoordinateSystemIdentifier(name="another_cs2"),
            ),
        ),
        labels=[omz_label1],
    )

    scene_cs1 = ozm06trans.CoordinateSystem(
        name="scene_cs1",
        axes=(
            ozm06trans.Axis(name="axis_0", type="space"),
            ozm06trans.Axis(name="axis_1", type="space"),
            ozm06trans.Axis(name="axis_2", type="channel"),
        ),
    )
    scene_transf1 = ozm06trans.Identity(
        input=ozm06trans.CoordinateSystemIdentifier(name=scene_cs1.name),
        output=ozm06trans.CoordinateSystemIdentifier(path="multiscales2", name="another_cs2"),
    )
    scene = oz.OMEZarrScene(
        coordinate_systems=[scene_cs1],
        coordinate_transformations=[scene_transf1],
        images={"multiscales1": multiscales1, "multiscales2": multiscales2},
    )
