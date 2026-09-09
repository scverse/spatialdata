from __future__ import annotations

import numpy as np
from ome_zarr import OMEZarrImage, OMEZarrMultiscale
from ome_zarr_models.v06.coordinate_transforms import Scale as ModelsScale
from ome_zarr_models.v06.coordinate_transforms import Sequence as ModelsSequence
from ome_zarr_models.v06.coordinate_transforms import Translation as ModelsTranslation

from spatialdata._io.io_raster import try_parse_ngff06_multiscale
from tests.conftest import SEED


def test_parse_multiscale():
    data = np.random.default_rng(seed=SEED).random((256, 256))
    omz_img = OMEZarrImage(data=data, axes="yx")
    ms = OMEZarrMultiscale(image=omz_img, scale_factors=(2, 4, 8, 16))

    data_tree, _transforms = try_parse_ngff06_multiscale(ms)
    for xr_scale_node, ngff_scale, ngff_meta in zip(
        data_tree.children.values(), ms.images, ms.metadata.datasets, strict=True
    ):
        xr_scale = xr_scale_node["image"]
        assert xr_scale.shape == ngff_scale.data.shape

        coords_x = xr_scale.coords["x"]
        coords_y = xr_scale.coords["y"]

        seq = ngff_meta.coordinateTransformations[0]
        assert isinstance(seq, ModelsSequence)
        scale = seq.transformations[0]
        assert isinstance(scale, ModelsScale)
        translate = seq.transformations[1]
        assert isinstance(translate, ModelsTranslation)

        start_indices = (0, 0)
        start = (coords_x[0], coords_y[0])

        end_indices = (xr_scale.shape[0] - 1, xr_scale.shape[1] - 1)
        end = (coords_x[-1], coords_y[-1])

        for indices, point in [(start_indices, start), (end_indices, end)]:
            expected = np.asarray(scale.scale) * indices + translate.translation
            assert np.allclose(expected, point)

    sliced_data_tree = data_tree.sel(x=slice(0, 256), y=slice(0, 256), method="nearest")
    assert sliced_data_tree.equals(data_tree)

    return data_tree, ms
