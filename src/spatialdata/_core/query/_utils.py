from typing import Any

import numba as nb
import numpy as np
from anndata import AnnData
from xarray import DataArray, Dataset, DataTree

from spatialdata._core._elements import Tables
from spatialdata._core.spatialdata import SpatialData
from spatialdata._types import ArrayLike
from spatialdata._utils import Number, _parse_list_into_array
from spatialdata.transformations._utils import compute_coordinates
from spatialdata.transformations.transformations import BaseTransformation, Sequence, Translation


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

    if min_coordinate.ndim == 1:
        min_coordinate = min_coordinate[np.newaxis, :]
        max_coordinate = max_coordinate[np.newaxis, :]

    if min_coordinate.shape[1] not in (2, 3):
        raise ValueError("bounding box must be 2D or 3D")

    num_boxes = min_coordinate.shape[0]
    num_dims = min_coordinate.shape[1]

    if num_dims == 2:
        # 2D bounding box
        assert len(axes) == 2
        corners = np.array(
            [
                [min_coordinate[:, 0], min_coordinate[:, 1]],
                [min_coordinate[:, 0], max_coordinate[:, 1]],
                [max_coordinate[:, 0], max_coordinate[:, 1]],
                [max_coordinate[:, 0], min_coordinate[:, 1]],
            ]
        )
        corners = np.transpose(corners, (2, 0, 1))
    else:
        # 3D bounding cube
        assert len(axes) == 3
        corners = np.array(
            [
                [min_coordinate[:, 0], min_coordinate[:, 1], min_coordinate[:, 2]],
                [min_coordinate[:, 0], min_coordinate[:, 1], max_coordinate[:, 2]],
                [min_coordinate[:, 0], max_coordinate[:, 1], max_coordinate[:, 2]],
                [min_coordinate[:, 0], max_coordinate[:, 1], min_coordinate[:, 2]],
                [max_coordinate[:, 0], min_coordinate[:, 1], min_coordinate[:, 2]],
                [max_coordinate[:, 0], min_coordinate[:, 1], max_coordinate[:, 2]],
                [max_coordinate[:, 0], max_coordinate[:, 1], max_coordinate[:, 2]],
                [max_coordinate[:, 0], max_coordinate[:, 1], min_coordinate[:, 2]],
            ]
        )
        corners = np.transpose(corners, (2, 0, 1))
    output = DataArray(
        corners,
        coords={
            "box": range(num_boxes),
            "corner": range(corners.shape[1]),
            "axis": list(axes),
        },
    )
    if num_boxes > 1:
        return output
    return output.squeeze().drop_vars("box")


@nb.jit(parallel=False, nopython=True)
def _create_slices_and_translation(
    min_values: nb.types.Array,
    max_values: nb.types.Array,
) -> tuple[nb.types.Array, nb.types.Array]:
    n_boxes, n_dims = min_values.shape
    slices = np.empty((n_boxes, n_dims, 2), dtype=np.float64)  # (n_boxes, n_dims, [min, max])
    translation_vectors = np.empty((n_boxes, n_dims), dtype=np.float64)  # (n_boxes, n_dims)

    for i in range(n_boxes):
        for j in range(n_dims):
            slices[i, j, 0] = min_values[i, j]
            slices[i, j, 1] = max_values[i, j]
            translation_vectors[i, j] = np.ceil(max(min_values[i, j], 0))

    return slices, translation_vectors


def _process_data_tree_query_result(query_result: DataTree) -> DataTree | None:
    d = {}
    for k, data_tree in query_result.items():
        v = data_tree.values()
        assert len(v) == 1
        xdata = v.__iter__().__next__()
        if 0 in xdata.shape:
            if k == "scale0":
                return None
        else:
            d[k] = xdata

    # Remove scales after finding a missing scale
    scales_to_keep = []
    for i, scale_name in enumerate(d.keys()):
        if scale_name == f"scale{i}":
            scales_to_keep.append(scale_name)
        else:
            break

    # Case in which scale0 is not present but other scales are
    if len(scales_to_keep) == 0:
        return None

    d = {k: Dataset({"image": d[k]}) for k in scales_to_keep}
    result = DataTree.from_dict(d)

    from dask.array.core import _check_regular_chunks

    # rechunk to avoid irregular chunks
    for scale in result:
        data = result[scale]["image"].data
        chunks = data.chunks
        if not _check_regular_chunks(chunks):
            data = data.rechunk(data.chunksize)
            if not _check_regular_chunks(data.chunks):
                raise ValueError(
                    f"Chunks are not regular for {scale} of the queried data: {chunks} "
                    "and could also not be rechunked regularly. Please report this bug."
                )
            result[scale]["image"].data = data

    return result


def _process_query_result(
    result: DataArray | DataTree, translation_vector: ArrayLike, axes: tuple[str, ...]
) -> DataArray | DataTree | None:
    from dask.array.core import _check_regular_chunks

    from spatialdata.transformations import get_transformation, set_transformation

    if isinstance(result, DataArray):
        if 0 in result.shape:
            return None
        # rechunk to avoid irregular chunks
        if not _check_regular_chunks(result.data.chunks):
            result.data = result.data.rechunk(result.data.chunksize)
    elif isinstance(result, DataTree):
        result = _process_data_tree_query_result(result)
        if result is None:
            return None

    result = compute_coordinates(result)

    if not np.allclose(np.array(translation_vector), 0):
        translation_transform = Translation(translation=translation_vector, axes=axes)

        transformations = get_transformation(result, get_all=True)
        assert isinstance(transformations, dict)

        new_transformations = {}
        for coordinate_system, initial_transform in transformations.items():
            new_transformation: BaseTransformation = Sequence(
                [translation_transform, initial_transform],
            )
            new_transformations[coordinate_system] = new_transformation
        set_transformation(result, new_transformations, set_all=True)

    # let's make a copy of the transformations so that we don't modify the original object
    t = get_transformation(result, get_all=True)
    assert isinstance(t, dict)
    set_transformation(result, t.copy(), set_all=True)

    return result


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
