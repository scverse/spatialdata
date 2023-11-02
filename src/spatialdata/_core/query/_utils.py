from __future__ import annotations

import geopandas as gpd
from xarray import DataArray

from spatialdata._types import ArrayLike
from spatialdata._utils import Number, _parse_list_into_array


def circles_to_polygons(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # We should only be buffering points, not polygons. Unfortunately this is an expensive check.
    from spatialdata.models import ShapesModel

    values_geotypes = list(df.geom_type.unique())
    if values_geotypes == ["Point"]:
        buffered_df = df.set_geometry(df.geometry.buffer(df[ShapesModel.RADIUS_KEY]))
        # TODO replace with a function to copy the metadata (the parser could also do this): https://github.com/scverse/spatialdata/issues/258
        buffered_df.attrs[ShapesModel.TRANSFORM_KEY] = df.attrs[ShapesModel.TRANSFORM_KEY]
        return buffered_df
    if "Point" in values_geotypes:
        raise TypeError("Geometry contained shapes and polygons.")
    return df


def get_bounding_box_corners(
    axes: tuple[str, ...],
    min_coordinate: list[Number] | ArrayLike,
    max_coordinate: list[Number] | ArrayLike,
) -> DataArray:
    """Get the coordinates of the corners of a bounding box from the min/max values.

    Parameters
    ----------
    axes
        The axes that min_coordinate and max_coordinate refer to.
    min_coordinate
        The upper left hand corner of the bounding box (i.e., minimum coordinates
        along all dimensions).
    max_coordinate
        The lower right hand corner of the bounding box (i.e., the maximum coordinates
        along all dimensions

    Returns
    -------
    (N, D) array of coordinates of the corners. N = 4 for 2D and 8 for 3D.
    """
    min_coordinate = _parse_list_into_array(min_coordinate)
    max_coordinate = _parse_list_into_array(max_coordinate)

    if len(min_coordinate) not in (2, 3):
        raise ValueError("bounding box must be 2D or 3D")

    if len(min_coordinate) == 2:
        # 2D bounding box
        assert len(axes) == 2
        return DataArray(
            [
                [min_coordinate[0], min_coordinate[1]],
                [min_coordinate[0], max_coordinate[1]],
                [max_coordinate[0], max_coordinate[1]],
                [max_coordinate[0], min_coordinate[1]],
            ],
            coords={"corner": range(4), "axis": list(axes)},
        )

    # 3D bounding cube
    assert len(axes) == 3
    return DataArray(
        [
            [min_coordinate[0], min_coordinate[1], min_coordinate[2]],
            [min_coordinate[0], min_coordinate[1], max_coordinate[2]],
            [min_coordinate[0], max_coordinate[1], max_coordinate[2]],
            [min_coordinate[0], max_coordinate[1], min_coordinate[2]],
            [max_coordinate[0], min_coordinate[1], min_coordinate[2]],
            [max_coordinate[0], min_coordinate[1], max_coordinate[2]],
            [max_coordinate[0], max_coordinate[1], max_coordinate[2]],
            [max_coordinate[0], max_coordinate[1], min_coordinate[2]],
        ],
        coords={"corner": range(8), "axis": list(axes)},
    )
