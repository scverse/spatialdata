from __future__ import annotations

from typing import Any

from anndata import AnnData
from xarray import DataArray

from spatialdata._core._elements import Tables
from spatialdata._core.spatialdata import SpatialData
from spatialdata._types import ArrayLike
from spatialdata._utils import Number, _parse_list_into_array


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


def _get_filtered_or_unfiltered_tables(
    filter_table: bool, elements: dict[str, Any], sdata: SpatialData
) -> dict[str, AnnData] | Tables:
    """
    Get the tables in a SpatialData object.

    The tables of the SpatialData object can either be filtered to only include the tables that annotate an element in
    elements or all tables are returned.

    Parameters
    ----------
    filter_table
        Specifies whether to filter the tables to only include tables that annotate elements in the retrieved
        SpatialData object of the query.
    elements
        A dictionary containing the elements to use for filtering the tables.
    sdata
        The SpatialData object that contains the tables to filter.

    Returns
    -------
    A dictionary containing the filtered or unfiltered tables based on the value of the 'filter_table' parameter.

    """
    if filter_table:
        from spatialdata._core.query.relational_query import _filter_table_by_elements

        return {
            name: filtered_table
            for name, table in sdata.tables.items()
            if (filtered_table := _filter_table_by_elements(table, elements)) and len(filtered_table) != 0
        }

    return sdata.tables
