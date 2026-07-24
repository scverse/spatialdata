from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
from dask.array import Array as DaskArray
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from shapely import Point, box
from xarray import DataArray, DataTree

if TYPE_CHECKING:
    import datashader as ds
    import pandas as pd

from spatialdata._core.operations._utils import _parse_element
from spatialdata._core.operations.transform import transform
from spatialdata._core.operations.vectorize import to_polygons
from spatialdata._core.query.relational_query import get_values
from spatialdata._core.spatialdata import SpatialData
from spatialdata._types import ArrayLike
from spatialdata._utils import Number, _parse_list_into_array
from spatialdata.models import (
    Image2DModel,
    Image3DModel,
    Labels2DModel,
    Labels3DModel,
    PointsModel,
    ShapesModel,
    SpatialElement,
    get_axes_names,
    get_model,
)
from spatialdata.models._utils import _get_uint_dtype, get_spatial_axes
from spatialdata.transformations._utils import _get_scale, compute_coordinates
from spatialdata.transformations.operations import get_transformation, remove_transformation, set_transformation
from spatialdata.transformations.transformations import (
    Affine,
    BaseTransformation,
    Identity,
    Scale,
    Sequence,
    Translation,
    _get_affine_for_element,
)

VALUES_COLUMN = "__values_column"


def _filter_points_for_tile(
    data: pd.DataFrame,
    *,
    x_range: tuple[Number, Number],
    y_range: tuple[Number, Number],
    include_x_max: bool,
    include_y_max: bool,
) -> pd.DataFrame:
    x_upper_bound = data["x"] <= x_range[1] if include_x_max else data["x"] < x_range[1]
    y_upper_bound = data["y"] <= y_range[1] if include_y_max else data["y"] < y_range[1]
    return data[(data["x"] >= x_range[0]) & x_upper_bound & (data["y"] >= y_range[0]) & y_upper_bound]


def _rasterize_tile(
    partitions: list[pd.DataFrame | GeoDataFrame],
    *,
    is_shapes: bool,
    shape_positions: np.ndarray | None,
    plot_height: int,
    plot_width: int,
    x_range: tuple[Number, Number],
    y_range: tuple[Number, Number],
    agg_func: ds.reductions.Reduction,
    crop: tuple[slice, ...],
    empty_shape: tuple[int, ...],
    empty_dtype: np.dtype,
    empty_fill_value: Number,
) -> np.ndarray:
    import datashader as ds
    import pandas as pd

    canvas = ds.Canvas(plot_height=plot_height, plot_width=plot_width, x_range=x_range, y_range=y_range)
    if is_shapes:
        assert len(partitions) == 1
        data = partitions[0]
        assert isinstance(data, GeoDataFrame)
        assert shape_positions is not None
        visible_data = data.iloc[shape_positions].copy()
        if len(visible_data) == 0:
            return np.full(empty_shape, empty_fill_value, dtype=empty_dtype)[crop]
        aggregate = canvas.polygons(visible_data, "geometry", agg=agg_func)
    else:
        data = pd.concat(partitions)
        aggregate = canvas.points(data, x="x", y="y", agg=agg_func)
    return np.asarray(aggregate.data)[crop]


def _rasterize_tiled(
    data: DaskDataFrame | GeoDataFrame,
    *,
    is_shapes: bool,
    plot_height: int,
    plot_width: int,
    x_range: tuple[Number, Number],
    y_range: tuple[Number, Number],
    agg_func: ds.reductions.Reduction,
    tile_size: int,
) -> DataArray:
    import dask
    import dask.array as da
    import datashader as ds

    sample = data.iloc[:1].copy() if is_shapes else data._meta
    sample_canvas = ds.Canvas(plot_height=1, plot_width=1, x_range=x_range, y_range=y_range)
    if is_shapes:
        sample_aggregate = sample_canvas.polygons(sample, "geometry", agg=agg_func)
    else:
        sample_aggregate = sample_canvas.points(sample, x="x", y="y", agg=agg_func)
    y_axis = sample_aggregate.dims.index("y")
    x_axis = sample_aggregate.dims.index("x")

    partitions = list(data.to_delayed()) if isinstance(data, DaskDataFrame) else [dask.delayed(data, pure=True)]
    spatial_index = data.sindex if is_shapes else None
    x_scale = (x_range[1] - x_range[0]) / plot_width
    y_scale = (y_range[1] - y_range[0]) / plot_height
    rows = []
    for y_start in range(0, plot_height, tile_size):
        y_stop = min(y_start + tile_size, plot_height)
        row = []
        for x_start in range(0, plot_width, tile_size):
            x_stop = min(x_start + tile_size, plot_width)
            render_x_start = max(0, x_start - 1) if is_shapes else x_start
            render_x_stop = min(plot_width, x_stop + 1) if is_shapes else x_stop
            render_y_start = max(0, y_start - 1) if is_shapes else y_start
            render_y_stop = min(plot_height, y_stop + 1) if is_shapes else y_stop
            tile_x_range = (
                x_range[0] + render_x_start * x_scale,
                x_range[0] + render_x_stop * x_scale,
            )
            tile_y_range = (
                y_range[0] + render_y_start * y_scale,
                y_range[0] + render_y_stop * y_scale,
            )
            crop = [slice(None)] * sample_aggregate.ndim
            crop[y_axis] = slice(
                y_start - render_y_start,
                y_stop - render_y_start,
            )
            crop[x_axis] = slice(
                x_start - render_x_start,
                x_stop - render_x_start,
            )
            render_shape = list(sample_aggregate.shape)
            render_shape[y_axis] = render_y_stop - render_y_start
            render_shape[x_axis] = render_x_stop - render_x_start
            empty_fill_value = 0 if sample_aggregate.dtype.kind in "uib" else np.nan
            tile_partitions = partitions
            shape_positions = None
            if is_shapes:
                assert spatial_index is not None
                shape_positions = np.sort(
                    spatial_index.query(
                        box(tile_x_range[0], tile_y_range[0], tile_x_range[1], tile_y_range[1]),
                        predicate="intersects",
                    )
                )
            if not is_shapes:
                tile_partitions = [
                    dask.delayed(_filter_points_for_tile, pure=True)(
                        partition,
                        x_range=tile_x_range,
                        y_range=tile_y_range,
                        include_x_max=x_stop == plot_width,
                        include_y_max=y_stop == plot_height,
                    )
                    for partition in partitions
                ]
            tile = dask.delayed(_rasterize_tile, pure=True)(
                tile_partitions,
                is_shapes=is_shapes,
                shape_positions=shape_positions,
                plot_height=render_y_stop - render_y_start,
                plot_width=render_x_stop - render_x_start,
                x_range=tile_x_range,
                y_range=tile_y_range,
                agg_func=agg_func,
                crop=tuple(crop),
                empty_shape=tuple(render_shape),
                empty_dtype=sample_aggregate.dtype,
                empty_fill_value=empty_fill_value,
            )
            shape = list(sample_aggregate.shape)
            shape[y_axis] = y_stop - y_start
            shape[x_axis] = x_stop - x_start
            row.append(da.from_delayed(tile, shape=tuple(shape), dtype=sample_aggregate.dtype))
        rows.append(da.concatenate(row, axis=x_axis))
    aggregate = da.concatenate(rows, axis=y_axis)

    coords = {
        "y": y_range[0] + (np.arange(plot_height) + 0.5) * y_scale,
        "x": x_range[0] + (np.arange(plot_width) + 0.5) * x_scale,
    }
    for dim in sample_aggregate.dims:
        if dim not in coords:
            coords[dim] = sample_aggregate.coords[dim].values
    return DataArray(
        aggregate,
        coords=coords,
        dims=sample_aggregate.dims,
        name=sample_aggregate.name,
        attrs=sample_aggregate.attrs,
    )


def _compute_target_dimensions(
    spatial_axes: tuple[str, ...],
    min_coordinate: list[Number] | ArrayLike,
    max_coordinate: list[Number] | ArrayLike,
    target_unit_to_pixels: float | None,
    target_width: float | None,
    target_height: float | None,
    target_depth: float | None,
) -> tuple[float, float, float | None]:
    """
    Compute the pixel sizes (width, height, depth) of the image that will be produced by the rasterization.

    Parameters
    ----------
    spatial_axes
        The axes that min_coordinate and max_coordinate refer to.
    min_coordinate
        The minimum coordinates of the bounding box.
    max_coordinate
        The maximum coordinates of the bounding box.
    target_unit_to_pixels
        The number of pixels per unit that the target image should have. It is mandatory to specify precisely one of
        the following options: target_unit_to_pixels, target_width, target_height, target_depth.
    target_width
        The width of the target image in units. It is mandatory to specify precisely one of the following options:
        target_unit_to_pixels, target_width, target_height, target_depth.
    target_height
        The height of the target image in units. It is mandatory to specify precisely one of the following options:
        target_unit_to_pixels, target_width, target_height, target_depth.
    target_depth
        The depth of the target image in units. It is mandatory to specify precisely one of the following options:
        target_unit_to_pixels, target_width, target_height, target_depth.

    Returns
    -------
    target_width, target_height, target_depth
        The pixel sizes (width, height, depth) of the image that will be produced by the rasterization.
        If spatial_axes does not contain "z", target_depth will be None.
    """
    if isinstance(target_width, int):
        target_width = float(target_width)
    if isinstance(target_height, int):
        target_height = float(target_height)
    if isinstance(target_depth, int):
        target_depth = float(target_depth)
    assert (
        np.sum(
            [
                target_unit_to_pixels is not None,
                target_width is not None,
                target_height is not None,
                target_depth is not None,
            ]
        )
        == 1
    ), "you must specify only one of: target_unit_to_pixels, target_width, target_height, target_depth"
    assert set(spatial_axes) == {"x", "y"} or set(spatial_axes) == {"x", "y", "z"}
    if "z" not in spatial_axes:
        assert target_depth is None, "you cannot specify a target depth for 2D data"

    x_index = spatial_axes.index("x")
    y_index = spatial_axes.index("y")
    w_bb = max_coordinate[x_index] - min_coordinate[x_index]
    h_bb = max_coordinate[y_index] - min_coordinate[y_index]
    assert w_bb > 0, "all max_coordinate values must be greater than all min_coordinate values"
    assert h_bb > 0, "all max_coordinate values must be greater than all min_coordinate values"
    w_to_h_bb = w_bb / h_bb

    d_bb = None
    d_to_h_bb = None
    if "z" in spatial_axes:
        z_index = spatial_axes.index("z")
        d_bb = max_coordinate[z_index] - min_coordinate[z_index]
        assert d_bb > 0, "all max_coordinate values must be greater than all min_coordinate values"
        d_to_h_bb = d_bb / h_bb

    if target_unit_to_pixels is not None:
        target_width = w_bb * target_unit_to_pixels
        target_height = h_bb * target_unit_to_pixels
        if "z" in spatial_axes:
            assert d_bb is not None
            target_depth = d_bb * target_unit_to_pixels
    elif target_width is not None:
        target_height = target_width / w_to_h_bb
        if "z" in spatial_axes:
            assert d_to_h_bb is not None
            target_depth = target_height * d_to_h_bb
    elif target_height is not None:
        target_width = target_height * w_to_h_bb
        if "z" in spatial_axes:
            assert d_to_h_bb is not None
            target_depth = target_height * d_to_h_bb
    elif target_depth is not None:
        assert d_to_h_bb is not None
        target_height = target_depth / d_to_h_bb
        target_width = target_height * w_to_h_bb
    else:
        raise RuntimeError("Should not reach here")
    assert target_width is not None
    assert isinstance(target_width, float)
    assert target_height is not None
    assert isinstance(target_height, float)
    return np.round(target_width), np.round(target_height), np.round(target_depth) if target_depth is not None else None


def rasterize(
    # required arguments
    data: SpatialData | SpatialElement | str,
    axes: tuple[str, ...],
    min_coordinate: list[Number] | ArrayLike,
    max_coordinate: list[Number] | ArrayLike,
    target_coordinate_system: str,
    target_unit_to_pixels: float | None = None,
    target_width: float | None = None,
    target_height: float | None = None,
    target_depth: float | None = None,
    # extra arguments
    sdata: SpatialData | None = None,
    value_key: str | None = None,
    table_name: str | None = None,
    return_regions_as_labels: bool = False,
    agg_func: str | ds.reductions.Reduction | None = None,
    return_single_channel: bool | None = None,
    tile_size: int | None = None,
) -> SpatialData | DataArray:
    """
    Rasterize a `SpatialData` object or a `SpatialElement` (image, labels, points, shapes).

    Parameters
    ----------
    data
        The `SpatialData` object or `SpatialElement` to rasterize. In alternative, the name of the `SpatialElement` in
        the `SpatialData` object, when the `SpatialData` object is passed to `values_sdata`.
    axes
        The axes that `min_coordinate` and `max_coordinate` refer to.
    min_coordinate
        The minimum coordinates of the bounding box.
    max_coordinate
        The maximum coordinates of the bounding box.
    target_coordinate_system
        The coordinate system in which we define the bounding box. This will also be the coordinate system of the
        produced rasterized image.
    target_unit_to_pixels
        The number of pixels per unit that the target image should have. It is mandatory to specify precisely one of
        the following options: `target_unit_to_pixels`, `target_width`, `target_height`, `target_depth`.
    target_width
        The width of the target image in units. It is mandatory to specify precisely one of the following options:
        `target_unit_to_pixels`, `target_width`, `target_height`, `target_depth`.
    target_height
        The height of the target image in units. It is mandatory to specify precisely one of the following options:
        `target_unit_to_pixels`, `target_width`, `target_height`, `target_depth`.
    target_depth
        The depth of the target image in units. It is mandatory to specify precisely one of the following options:
        `target_unit_to_pixels`, `target_width`, `target_height`, `target_depth`.
    sdata
        `SpatialData` object containing the values to aggregate if `value_key` refers to values from a table. Must
        be `None` when `data` is a `SpatialData` object.
    value_key
        Name of the column containing the values to aggregate; can refer both to numerical or
        categorical values.

        The key can be:

        - the name of a column(s) in the dataframe (Dask `DataFrame` for points or `GeoDataFrame` for shapes);
        - the name of obs column(s) in the associated `AnnData` table (for points, shapes, and labels);
        - the name of a var(s), referring to the column(s) of the X matrix in the table (for points, shapes, and
          labels).

        See the notes for more details on the default behavior.
        Must be `None` when `data` is a `SpatialData` object.
    table_name
        The table optionally containing the `value_key` and the name of the table in the returned `SpatialData` object.
        Must be `None` when `data` is a `SpatialData` object, otherwise it assumes the default value of `'table'`.
    return_regions_as_labels
        By default, single-scale images of shape `(c, y, x)` are returned. If `True`, returns labels, shapes and points
        as labels of shape `(y, x)` as opposed to an image of shape `(c, y, x)`. Images are always returned as images,
        and multiscale raster data is always returned as single-scale data.
    agg_func
        Available only when rasterizing points and shapes. A reduction function from datashader (its name, or a
        `Callable`). See the notes for more details on the default behavior.
        Must be `None` when `data` is a `SpatialData` object.
    return_single_channel
        Only used when rasterizing points and shapes and when `value_key` refers to a categorical column. If `False`,
        each category will be rasterized in a separate channel.
    tile_size
        Maximum size, in pixels, of each spatial output chunk. For points and shapes, this controls the Datashader
        canvas size. For images and labels, this controls the Dask output chunks.

    Returns
    -------
    The rasterized `SpatialData` object or SpatialData supported `DataArray`. Each `SpatialElement` will be rasterized
    into a `DataArray` (not a `DataTree`). So if a `SpatialData` object with elements is passed, a `SpatialData` object
    with single-scale images and labels will be returned.

    When `return_regions_as_labels` is `True`, the returned `DataArray` object will have an attribute called
    `label_index_to_category` that maps the label index to the category name. You can access it via
    `returned_data.attrs["label_index_to_category"]`. The returned labels will start from 1 (0 is reserved for the
    background), and will be contiguous.

    Notes
    -----
    For images and labels, the parameters `value_key`, `table_name`, `agg_func`, and `return_single_channel` are not
    used.

    Instead, when rasterizing shapes and points, the following table clarifies the default datashader reduction used
    for various combinations of parameters.

    In particular, the first two rows refer to the default behavior when the parameters (`value_key`, 'table_name',
    `returned_single_channel`, `agg_func`) are kept to their default values.

    +------------+----------------------------+---------------------+---------------------+------------+
    | value_key  | Shapes or Points           | return_single_chan  | datashader reduct.  | table_name |
    +============+============================+=====================+=====================+============+
    | None*      | Point (default)            | NA                  | count               | 'table'    |
    +------------+----------------------------+---------------------+---------------------+------------+
    | None**     | Shapes (default)           | True                | first               | 'table'    |
    +------------+----------------------------+---------------------+---------------------+------------+
    | None**     | Shapes                     | False               | count_cat           | 'table'    |
    +------------+----------------------------+---------------------+---------------------+------------+
    | category   | NA                         | True                | first               | 'table'    |
    +------------+----------------------------+---------------------+---------------------+------------+
    | category   | NA                         | False               | count_cat           | 'table'    |
    +------------+----------------------------+---------------------+---------------------+------------+
    | int/float  | NA                         | NA                  | sum                 | 'table'    |
    +------------+----------------------------+---------------------+---------------------+------------+

    Explicitly, the default behaviors are as follows.

    - for points, each pixel counts the number of points belonging to it, (the `count` function is applied to an
      artificial column of ones);
    - for shapes, each pixel gets a single index among the ones of the shapes that intersect it (the index of the
      shapes is interpreted as a categorical column and then the `first` function is used).
    """
    if tile_size is not None and (isinstance(tile_size, bool) or not isinstance(tile_size, Integral) or tile_size <= 0):
        raise ValueError("tile_size must be a positive integer.")

    if isinstance(data, SpatialData):
        if sdata is not None:
            raise ValueError("When data is a SpatialData object, sdata must be None.")
        if value_key is not None:
            raise ValueError("When data is a SpatialData object, value_key must be None.")
        if table_name is not None:
            raise ValueError("When data is a SpatialData object, table_name must be None.")
        if agg_func is not None:
            raise ValueError("When data is a SpatialData object, agg_func must be None.")
        new_images = {}
        new_labels = {}
        for element_type in ["points", "images", "labels", "shapes"]:
            elements = getattr(data, element_type)
            for name in elements:
                rasterized = rasterize(
                    data=name,
                    axes=axes,
                    min_coordinate=min_coordinate,
                    max_coordinate=max_coordinate,
                    target_coordinate_system=target_coordinate_system,
                    target_unit_to_pixels=target_unit_to_pixels,
                    target_width=target_width,
                    target_height=target_height,
                    target_depth=target_depth,
                    sdata=data,
                    return_regions_as_labels=return_regions_as_labels,
                    return_single_channel=return_single_channel if element_type in ("points", "shapes") else None,
                    tile_size=tile_size,
                )
                new_name = f"{name}_rasterized_{element_type}"
                model = get_model(rasterized)
                if model in (Image2DModel, Image3DModel):
                    new_images[new_name] = rasterized
                elif model in (Labels2DModel, Labels3DModel):
                    new_labels[new_name] = rasterized
                else:
                    raise RuntimeError(f"Unsupported model {model} detected as return type of rasterize().")
        return SpatialData(images=new_images, labels=new_labels, tables=data.tables, attrs=data.attrs)

    parsed_data = _parse_element(element=data, sdata=sdata, element_var_name="data", sdata_var_name="sdata")
    model = get_model(parsed_data)
    if model in (Image2DModel, Image3DModel, Labels2DModel, Labels3DModel):
        if agg_func is not None:
            raise ValueError("agg_func must be None when data is an image or labels.")
        if return_single_channel is not None:
            raise ValueError("return_single_channel must be None when data is an image or labels.")
        rasterized = rasterize_images_labels(
            data=parsed_data,
            axes=axes,
            min_coordinate=min_coordinate,
            max_coordinate=max_coordinate,
            target_coordinate_system=target_coordinate_system,
            target_unit_to_pixels=target_unit_to_pixels,
            target_width=target_width,
            target_height=target_height,
            target_depth=target_depth,
            tile_size=tile_size,
        )
        transformations = get_transformation(rasterized, get_all=True)
        assert isinstance(transformations, dict)
        # adjust the return type
        if model in (Labels2DModel, Labels3DModel) and not return_regions_as_labels:
            model = Image2DModel if model == Labels2DModel else Image3DModel
            rasterized = model.parse(rasterized.expand_dims("c", axis=0))
        # eventually color the raster data by the specified value column
        if value_key is not None:
            element_name = data if isinstance(data, str) else None
            kwargs = {"sdata": sdata, "element_name": element_name} if element_name is not None else {"element": data}
            values = get_values(value_key, table_name=table_name, **kwargs).iloc[:, 0]  # type: ignore[arg-type, union-attr]
            max_index: int = np.max(values.index)
            assigner = np.zeros(max_index + 1, dtype=values.dtype)
            assigner[values.index] = values
            # call-arg is ignored because model is never TableModel (the error is that the transformation param is not
            # accepted by TableModel.parse)
            rasterized = model.parse(assigner[rasterized], transformations=transformations)  # type: ignore[call-arg]
        return rasterized
    if model in (PointsModel, ShapesModel):
        return rasterize_shapes_points(
            data=parsed_data,
            axes=axes,
            min_coordinate=min_coordinate,
            max_coordinate=max_coordinate,
            target_coordinate_system=target_coordinate_system,
            target_unit_to_pixels=target_unit_to_pixels,
            target_width=target_width,
            target_height=target_height,
            target_depth=target_depth,
            element_name=data if isinstance(data, str) else None,
            sdata=sdata,
            value_key=value_key,
            table_name=table_name,
            return_regions_as_labels=return_regions_as_labels,
            agg_func=agg_func,
            return_single_channel=return_single_channel,
            tile_size=tile_size,
        )
    raise ValueError(f"Unsupported model {model}.")


def _get_xarray_data_to_rasterize(
    data: DataArray | DataTree,
    axes: tuple[str, ...],
    min_coordinate: list[Number] | ArrayLike,
    max_coordinate: list[Number] | ArrayLike,
    target_sizes: dict[str, float | None],
    target_coordinate_system: str,
) -> tuple[DataArray, Scale | None]:
    """Make the DataArray to rasterize from either a Spatialdata supported DataArray or DataTree.

    If from a pyramid level, computes scale factor.

    Parameters
    ----------
    data
        The input data to be rasterized.
    axes
       The axes that min_coordinate and max_coordinate refer to.
    min_coordinate
        The minimum coordinates of the bounding box for the data to be rasterized.
    max_coordinate
        The maximum coordinates of the bounding box for the data to be rasterized.
    target_sizes
        A dictionary containing the target size (in pixels) of each axis after rasterization.

    Returns
    -------
    A tuple containing the DataArray to be rasterized and its scale, if the selected DataArray comes from a pyramid
    level that is not the full resolution but has been scaled.
    """
    min_coordinate = _parse_list_into_array(min_coordinate)
    max_coordinate = _parse_list_into_array(max_coordinate)
    if isinstance(data, DataArray):
        xdata = data
        pyramid_scale = None
    elif isinstance(data, DataTree):
        latest_scale: str | None = None
        for scale in reversed(list(data.keys())):
            data_tree = data[scale]
            latest_scale = scale
            v = data_tree.values()
            assert len(v) == 1
            xdata = next(iter(v))
            assert set(get_spatial_axes(tuple(xdata.sizes.keys()))) == set(axes)

            corrected_affine, _ = _get_corrected_affine_matrix(
                data=xdata,
                axes=axes,
                target_coordinate_system=target_coordinate_system,
            )
            m = corrected_affine.inverse().matrix  # type: ignore[attr-defined]
            m_linear = m[:-1, :-1]
            m_translation = m[:-1, -1]
            from spatialdata._core.query._utils import get_bounding_box_corners

            bb_corners = get_bounding_box_corners(
                min_coordinate=min_coordinate, max_coordinate=max_coordinate, axes=axes
            )
            assert tuple(bb_corners.axis.data.tolist()) == axes
            bb_in_xdata = bb_corners.data @ m_linear + m_translation
            bb_in_xdata_sizes = {
                ax: bb_in_xdata[:, axes.index(ax)].max() - bb_in_xdata[:, axes.index(ax)].min() for ax in axes
            }
            for ax in axes:
                # TLDR; the sqrt selects a pyramid level in which the requested bounding box is a bit larger than the
                # size of the data we want to obtain
                #
                # Intuition: the sqrt is to account for the fact that the bounding box could be rotated in the
                # intrinsic space, so bb_in_xdata_sizes is actually measuring the size of the bounding box of the
                # inverse-transformed bounding box. The sqrt comes from the ratio of the side of a square,
                # and the maximum diagonal of a square containing the original square, if the original square is
                # rotated.
                if bb_in_xdata_sizes[ax] < target_sizes[ax] * np.sqrt(len(axes)):
                    break
            else:
                # when this code is reached, latest_scale is selected
                break
        assert latest_scale is not None
        xdata = next(iter(data[latest_scale].values()))
        if latest_scale != "scale0":
            transformations = xdata.attrs["transform"]
            pyramid_scale = _get_scale(transformations)
        else:
            pyramid_scale = None
    else:
        raise RuntimeError("Should not reach here")
    return xdata, pyramid_scale


def _get_corrected_affine_matrix(
    data: DataArray | DataTree,
    axes: tuple[str, ...],
    target_coordinate_system: str,
) -> tuple[Affine, tuple[str, ...]]:
    """Get the affine matrix that maps the intrinsic coordinates of the data to the target_coordinate_system.

    In addition:

        - restricting the domain to the axes specified in axes (i.e. the axes for which the bounding box is specified),
            in particular axes never contains c;
        - restricting the codomain to the spatial axes of the target coordinate system (i.e. excluding c).

    We do this because:

        - we don't need to consider c
        - when we create the target rasterized object, we need to have axes in the order that is requires by the schema

    """
    transformation = get_transformation(data, target_coordinate_system)
    assert isinstance(transformation, BaseTransformation)
    affine = _get_affine_for_element(data, transformation)
    target_axes_unordered = affine.output_axes
    assert set(target_axes_unordered) in [{"x", "y", "z"}, {"x", "y"}, {"c", "x", "y", "z"}, {"c", "x", "y"}]
    target_axes: tuple[str, ...]
    if "z" in target_axes_unordered:
        target_axes = ("c", "z", "y", "x") if "c" in target_axes_unordered else ("z", "y", "x")
    else:
        target_axes = ("c", "y", "x") if "c" in target_axes_unordered else ("y", "x")
    target_spatial_axes = get_spatial_axes(target_axes)
    assert len(target_spatial_axes) == len(axes)
    assert len(target_spatial_axes) == len(axes)
    corrected_affine = affine.to_affine(input_axes=axes, output_axes=target_spatial_axes)
    return corrected_affine, target_axes


# TODO: rename this function to an internatl function and invoke this function from a function that has arguments
#  values, values_sdata
def rasterize_images_labels(
    data: SpatialElement,
    axes: tuple[str, ...],
    min_coordinate: list[Number] | ArrayLike,
    max_coordinate: list[Number] | ArrayLike,
    target_coordinate_system: str,
    target_unit_to_pixels: float | None = None,
    target_width: float | None = None,
    target_height: float | None = None,
    target_depth: float | None = None,
    tile_size: int | None = None,
) -> DataArray:
    import dask_image.ndinterp

    min_coordinate = _parse_list_into_array(min_coordinate)
    max_coordinate = _parse_list_into_array(max_coordinate)
    # get dimensions of the target image
    spatial_axes = get_spatial_axes(axes)
    target_width, target_height, target_depth = _compute_target_dimensions(
        spatial_axes=spatial_axes,
        min_coordinate=min_coordinate,
        max_coordinate=max_coordinate,
        target_unit_to_pixels=target_unit_to_pixels,
        target_width=target_width,
        target_height=target_height,
        target_depth=target_depth,
    )
    target_sizes = {
        "x": target_width,
        "y": target_height,
        "z": target_depth,
    }

    bb_sizes = {ax: max_coordinate[axes.index(ax)] - min_coordinate[axes.index(ax)] for ax in axes}
    scale_vector = [bb_sizes[ax] / target_sizes[ax] for ax in axes]
    scale = Scale(scale_vector, axes=axes)

    offset = [min_coordinate[axes.index(ax)] for ax in axes]
    translation = Translation(offset, axes=axes)

    xdata, pyramid_scale = _get_xarray_data_to_rasterize(
        data=data,
        axes=axes,
        min_coordinate=min_coordinate,
        max_coordinate=max_coordinate,
        target_sizes=target_sizes,
        target_coordinate_system=target_coordinate_system,
    )

    extra = [pyramid_scale.inverse()] if pyramid_scale is not None else []

    # get inverse transformation
    corrected_affine, target_axes = _get_corrected_affine_matrix(
        data=data,
        axes=axes,
        target_coordinate_system=target_coordinate_system,
    )

    if "z" in spatial_axes:
        half_pixel_offset = Translation([0.5, 0.5, 0.5], axes=("z", "y", "x"))
    else:
        half_pixel_offset = Translation([0.5, 0.5], axes=("y", "x"))
    sequence = Sequence(
        [
            # half_pixel_offset.inverse(),
            scale,
            translation,
            corrected_affine.inverse(),
            # half_pixel_offset,
        ]
        + extra
    )
    dims = get_axes_names(data)
    matrix = sequence.to_affine_matrix(input_axes=target_axes, output_axes=dims)

    # get output shape
    output_shape_ = []
    for ax in dims:
        f = xdata.sizes[ax] if ax == "c" else target_sizes[ax]
        if f is not None:
            output_shape_.append(int(f))
    output_shape = tuple(output_shape_)
    output_chunks = None
    if tile_size is not None:
        output_chunks = tuple(
            min(tile_size, output_shape[i])
            if ax in spatial_axes
            else min(xdata.data.chunksize[xdata.get_axis_num(ax)], output_shape[i])
            for i, ax in enumerate(dims)
        )

    # get kwargs and schema
    schema = get_model(data)
    # labels need to be preserved after the resizing of the image
    if schema in (Labels2DModel, Labels3DModel):
        kwargs = {"prefilter": False, "order": 0}
    elif schema in (Image2DModel, Image3DModel):
        kwargs = {"order": 0}
    else:
        raise ValueError(f"Unsupported schema {schema}")

    # resample the image
    transformed_dask = dask_image.ndinterp.affine_transform(
        xdata.data,
        matrix=matrix,
        output_shape=output_shape,
        output_chunks=output_chunks,
        **kwargs,
    )
    assert isinstance(transformed_dask, DaskArray)
    channels = xdata.coords["c"].values if schema in (Image2DModel, Image3DModel) else None
    transformed_data = schema.parse(transformed_dask, dims=xdata.dims, c_coords=channels)  # type: ignore[call-arg]

    if target_coordinate_system != "global":
        remove_transformation(transformed_data, "global")

    sequence = Sequence([half_pixel_offset.inverse(), scale, translation, half_pixel_offset])
    set_transformation(transformed_data, sequence, target_coordinate_system)

    transformed_data = compute_coordinates(transformed_data)
    schema.validate(transformed_data)
    return transformed_data


def rasterize_shapes_points(
    data: DaskDataFrame | GeoDataFrame,
    axes: tuple[str, ...],
    min_coordinate: list[Number] | ArrayLike,
    max_coordinate: list[Number] | ArrayLike,
    target_coordinate_system: str,
    target_unit_to_pixels: float | None = None,
    target_width: float | None = None,
    target_height: float | None = None,
    target_depth: float | None = None,
    element_name: str | None = None,
    sdata: SpatialData | None = None,
    value_key: str | None = None,
    table_name: str | None = None,
    return_regions_as_labels: bool = False,
    agg_func: str | ds.reductions.Reduction | None = None,
    return_single_channel: bool | None = None,
    tile_size: int | None = None,
) -> DataArray:
    import datashader as ds

    min_coordinate = _parse_list_into_array(min_coordinate)
    max_coordinate = _parse_list_into_array(max_coordinate)
    target_width, target_height, target_depth = _compute_target_dimensions(
        spatial_axes=axes,
        min_coordinate=min_coordinate,
        max_coordinate=max_coordinate,
        target_unit_to_pixels=target_unit_to_pixels,
        target_width=target_width,
        target_height=target_height,
        target_depth=target_depth,
    )

    GEOMETRY_COLUMNS = ["x", "y"] if isinstance(data, DaskDataFrame) else ["geometry"]
    columns = GEOMETRY_COLUMNS + [value_key] if value_key in data else GEOMETRY_COLUMNS
    if isinstance(data, GeoDataFrame) and isinstance(data.iloc[0].geometry, Point):
        assert isinstance(columns, list)
        columns += ["radius"]
    data = data[columns]

    plot_width, plot_height = int(target_width), int(target_height)
    y_range = (min_coordinate[axes.index("y")], max_coordinate[axes.index("y")])
    x_range = (min_coordinate[axes.index("x")], max_coordinate[axes.index("x")])
    t = get_transformation(data, target_coordinate_system)
    if not isinstance(t, Identity):
        data = transform(data, to_coordinate_system=target_coordinate_system)

    table_name = table_name if table_name is not None else "table"

    index = False
    if value_key is not None:
        kwargs = {"sdata": sdata, "element_name": element_name} if element_name is not None else {"element": data}
        data[VALUES_COLUMN] = get_values(value_key, table_name=table_name, **kwargs).iloc[:, 0]  # type: ignore[arg-type, union-attr]
    elif isinstance(data, GeoDataFrame) or isinstance(data, DaskDataFrame) and return_regions_as_labels is True:
        value_key = VALUES_COLUMN
        data[VALUES_COLUMN] = data.index.astype("category")
        index = True
    else:
        value_key = VALUES_COLUMN
        data[VALUES_COLUMN] = 1

    label_index_to_category = None
    if VALUES_COLUMN in data and data[VALUES_COLUMN].dtype == "category":
        if isinstance(data, DaskDataFrame):
            # We have to do this because as_known() does not preserve the order anymore in latest dask versions
            # TODO discuss whether we can always expect the index from before to be monotonically increasing, because
            # then we don't have to check order.
            if index:
                data[VALUES_COLUMN] = data[VALUES_COLUMN].cat.set_categories(data.index, ordered=True)
            else:
                data[VALUES_COLUMN] = data[VALUES_COLUMN].cat.as_known()
        label_index_to_category = dict(enumerate(data[VALUES_COLUMN].cat.categories, start=1))

    if return_single_channel is None:
        return_single_channel = True
    if agg_func is None:
        agg_func = _default_agg_func(data, value_key, return_single_channel)
    elif isinstance(agg_func, str):
        AGGREGATIONS = ["sum", "count", "count_cat", "first"]

        assert np.isin(agg_func, AGGREGATIONS), (
            f"Aggregation function must be one of {', '.join(AGGREGATIONS)}. Found {agg_func}"
        )

        assert agg_func == "count" or value_key is not None, f"value_key cannot be done for agg_func={agg_func}"

        agg_func = getattr(ds, agg_func)(column=value_key)

    is_shapes = isinstance(data, GeoDataFrame)
    if is_shapes:
        data = to_polygons(data)
    if tile_size is None:
        cnv = ds.Canvas(plot_height=plot_height, plot_width=plot_width, x_range=x_range, y_range=y_range)
        if is_shapes:
            agg = cnv.polygons(data, "geometry", agg=agg_func)
        else:
            agg = cnv.points(data, x="x", y="y", agg=agg_func)
    else:
        agg = _rasterize_tiled(
            data,
            is_shapes=is_shapes,
            plot_height=plot_height,
            plot_width=plot_width,
            x_range=x_range,
            y_range=y_range,
            agg_func=agg_func,
            tile_size=int(tile_size),
        )

    if label_index_to_category is not None and isinstance(agg_func, ds.first):
        agg.attrs["label_index_to_category"] = label_index_to_category

    scale = Scale([(y_range[1] - y_range[0]) / plot_height, (x_range[1] - x_range[0]) / plot_width], axes=("y", "x"))
    translation = Translation([y_range[0], x_range[0]], axes=("y", "x"))
    transformations: dict[str, BaseTransformation] = {target_coordinate_system: Sequence([scale, translation])}

    if isinstance(agg_func, ds.count_cat):
        if return_single_channel:
            raise ValueError("Cannot return single channel when using count_cat aggregation")
        if return_regions_as_labels:
            raise ValueError("Cannot return labels when using count_cat aggregation")

        agg = agg.rename({VALUES_COLUMN: "c"}).transpose("c", "y", "x")

        return Image2DModel.parse(agg, transformations=transformations)

    agg = agg.fillna(0)

    if return_regions_as_labels:
        if label_index_to_category is not None:
            max_label = next(iter(reversed(label_index_to_category.keys())))
        else:
            max_label = int(agg.max().values)
        agg = agg.astype(_get_uint_dtype(max_label))
        return Labels2DModel.parse(agg, transformations=transformations)

    agg = agg.expand_dims(dim={"c": 1}).transpose("c", "y", "x")
    return Image2DModel.parse(agg, transformations=transformations)


def _default_agg_func(
    data: DaskDataFrame | GeoDataFrame, value_key: str | None, return_single_channel: bool
) -> ds.reductions.Reduction:
    import datashader as ds

    if value_key is None:
        return ds.count()

    if data[VALUES_COLUMN].dtype != "category":
        return ds.sum(VALUES_COLUMN)

    if return_single_channel:
        data[VALUES_COLUMN] = data[VALUES_COLUMN].cat.codes + 1
        return ds.first(VALUES_COLUMN)

    return ds.count_cat(VALUES_COLUMN)
