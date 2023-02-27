from functools import singledispatch
from typing import Optional, Union

import dask_image.ndinterp
import numpy as np
from dask.array.core import Array as DaskArray
from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage

from spatialdata import SpatialData
from spatialdata._core._spatialdata_ops import (
    get_transformation,
    remove_transformation,
    set_transformation,
)
from spatialdata._core.core_utils import (
    SpatialElement,
    _get_scale,
    compute_coordinates,
    get_dims,
    get_spatial_axes,
)
from spatialdata._core.models import (
    Image2DModel,
    Image3DModel,
    Labels2DModel,
    Labels3DModel,
    get_schema,
)
from spatialdata._core.transformations import (
    BaseTransformation,
    Scale,
    Sequence,
    Translation,
    _get_affine_for_element,
)
from spatialdata._types import ArrayLike


def _compute_target_dimensions(
    spatial_axes: tuple[str, ...],
    min_coordinate: ArrayLike,
    max_coordinate: ArrayLike,
    target_unit_to_pixels: Optional[float],
    target_width: Optional[float],
    target_height: Optional[float],
    target_depth: Optional[float],
) -> tuple[float, float, Optional[float]]:
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
    )
    assert set(spatial_axes) == {"x", "y"} or set(spatial_axes) == {"x", "y", "z"}
    if "z" not in spatial_axes:
        assert target_depth is None

    x_index = spatial_axes.index("x")
    y_index = spatial_axes.index("y")
    w_bb = max_coordinate[x_index] - min_coordinate[x_index]
    h_bb = max_coordinate[y_index] - min_coordinate[y_index]
    assert w_bb > 0
    assert h_bb > 0
    w_to_h_bb = w_bb / h_bb

    d_bb = None
    d_to_h_bb = None
    if "z" in spatial_axes:
        z_index = spatial_axes.index("z")
        d_bb = max_coordinate[z_index] - min_coordinate[z_index]
        assert d_bb > 0
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
    return target_width, target_height, target_depth


@singledispatch
def rasterize(
    data: Union[SpatialData, SpatialElement],
    axes: tuple[str, ...],
    min_coordinate: ArrayLike,
    max_coordinate: ArrayLike,
    target_coordinate_system: str,
    target_unit_to_pixels: Optional[float] = None,
    target_width: Optional[float] = None,
    target_height: Optional[float] = None,
    target_depth: Optional[float] = None,
) -> Union[SpatialData, SpatialImage]:
    raise RuntimeError("Unsupported type: {type(data)}")


@rasterize.register(SpatialData)
def _(
    sdata: SpatialData,
    axes: tuple[str, ...],
    min_coordinate: ArrayLike,
    max_coordinate: ArrayLike,
    target_coordinate_system: str,
    target_unit_to_pixels: Optional[float] = None,
    target_width: Optional[float] = None,
    target_height: Optional[float] = None,
    target_depth: Optional[float] = None,
) -> SpatialData:
    from spatialdata import SpatialData

    new_images = {}
    for element_type in ["points", "images", "labels", "shapes"]:
        elements = getattr(sdata, element_type)
        for name, element in elements:
            rasterized = rasterize(
                data=element,
                axes=axes,
                min_coordinate=min_coordinate,
                max_coordinate=max_coordinate,
                target_coordinate_system=target_coordinate_system,
                target_unit_to_pixels=target_unit_to_pixels,
                target_width=target_width,
                target_height=target_height,
                target_depth=target_depth,
            )
            new_name = f"{name}_rasterized_{element_type}"
            new_images[new_name] = rasterized
    return SpatialData(images=new_images, table=sdata.table)


@rasterize.register(SpatialImage)
@rasterize.register(MultiscaleSpatialImage)
def _(
    data: SpatialImage,
    axes: tuple[str, ...],
    min_coordinate: ArrayLike,
    max_coordinate: ArrayLike,
    target_coordinate_system: str,
    target_unit_to_pixels: Optional[float] = None,
    target_width: Optional[float] = None,
    target_height: Optional[float] = None,
    target_depth: Optional[float] = None,
) -> SpatialImage:
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

    # get inverse transformation
    transformation = get_transformation(data, target_coordinate_system)
    dims = get_dims(data)
    assert isinstance(transformation, BaseTransformation)
    affine = _get_affine_for_element(data, transformation)
    target_axes_unordered = affine.output_axes
    assert set(target_axes_unordered) in [{"x", "y", "z"}, {"x", "y"}, {"c", "x", "y", "z"}, {"c", "x", "y"}]
    target_axes: tuple[str, ...]
    if "z" in target_axes_unordered:
        if "c" in target_axes_unordered:
            target_axes = ("c", "z", "y", "x")
        else:
            target_axes = ("z", "y", "x")
    else:
        if "c" in target_axes_unordered:
            target_axes = ("c", "y", "x")
        else:
            target_axes = ("y", "x")
    target_spatial_axes = get_spatial_axes(target_axes)
    assert len(target_spatial_axes) == len(min_coordinate)
    assert len(target_spatial_axes) == len(max_coordinate)
    corrected_affine = affine.to_affine(input_axes=axes, output_axes=target_spatial_axes)

    # get xdata
    if isinstance(data, SpatialImage):
        xdata = data
        pyramid_scale = None
    elif isinstance(data, MultiscaleSpatialImage):
        latest_scale: Optional[str] = None
        for scale in reversed(list(data.keys())):
            data_tree = data[scale]
            latest_scale = scale
            v = data_tree.values()
            assert len(v) == 1
            xdata = next(iter(v))
            assert set(get_spatial_axes(tuple(xdata.sizes.keys()))) == set(axes)

            m = corrected_affine.inverse().matrix  # type: ignore[attr-defined]
            m_linear = m[:-1, :-1]
            m_translation = m[:-1, -1]
            from spatialdata._core._spatial_query import get_bounding_box_corners

            bb_corners = get_bounding_box_corners(
                min_coordinate=min_coordinate, max_coordinate=max_coordinate, axes=axes
            )
            assert tuple(bb_corners.axis.data.tolist()) == axes
            bb_in_xdata = bb_corners.data @ m_linear + m_translation
            bb_in_xdata_sizes = {
                ax: bb_in_xdata[axes.index(ax)].max() - bb_in_xdata[axes.index(ax)].min() for ax in axes
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
                if bb_in_xdata_sizes[ax] * np.sqrt(len(axes)) < target_sizes[ax]:
                    break
            else:
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

    bb_sizes = {ax: max_coordinate[axes.index(ax)] - min_coordinate[axes.index(ax)] for ax in axes}
    scale_vector = [bb_sizes[ax] / target_sizes[ax] for ax in axes]
    scale = Scale(scale_vector, axes=axes)

    offset = [min_coordinate[axes.index(ax)] for ax in axes]
    translation = Translation(offset, axes=axes)

    if pyramid_scale is not None:
        extra = [pyramid_scale.inverse()]
    else:
        extra = []

    half_pixel_offset = Translation([0.5, 0.5, 0.5], axes=("z", "y", "x"))
    sequence = Sequence(
        [
            half_pixel_offset.inverse(),
            scale,
            translation,
            corrected_affine.inverse(),
            half_pixel_offset,
        ]
        + extra
    )
    matrix = sequence.to_affine_matrix(input_axes=target_axes, output_axes=dims)

    # get output shape
    output_shape_ = []
    for ax in dims:
        if ax == "c":
            f = xdata.sizes[ax]
        else:
            f = target_sizes[ax]
        if f is not None:
            output_shape_.append(int(f))
    output_shape = tuple(output_shape_)

    # get kwargs and schema
    schema = get_schema(data)
    # labels need to be preserved after the resizing of the image
    if schema == Labels2DModel or schema == Labels3DModel:
        kwargs = {"prefilter": False, "order": 0}
    elif schema == Image2DModel or schema == Image3DModel:
        kwargs = {}
    else:
        raise ValueError(f"Unsupported schema {schema}")

    # TODO: adjust matrix
    # TODO: add c
    # resample the image
    transformed_dask = dask_image.ndinterp.affine_transform(
        xdata.data,
        matrix=matrix,
        output_shape=output_shape,
        # output_chunks=xdata.data.chunks,
        **kwargs,
    )
    assert isinstance(transformed_dask, DaskArray)
    transformed_data = schema.parse(transformed_dask, dims=xdata.dims)  # type: ignore[call-arg,arg-type]
    if target_coordinate_system != "global":
        remove_transformation(transformed_data, "global")

    sequence = Sequence([half_pixel_offset.inverse(), scale, translation])
    set_transformation(transformed_data, sequence, target_coordinate_system)

    transformed_data = compute_coordinates(transformed_data)
    schema().validate(transformed_data)
    return transformed_data


@rasterize.register(DaskDataFrame)
def _(
    data: DaskDataFrame,
    axes: tuple[str, ...],
    min_coordinate: ArrayLike,
    max_coordinate: ArrayLike,
    target_coordinate_system: str,
    target_unit_to_pixels: Optional[float] = None,
    target_width: Optional[float] = None,
    target_height: Optional[float] = None,
    target_depth: Optional[float] = None,
) -> SpatialImage:
    target_width, target_height, target_depth = _compute_target_dimensions(
        spatial_axes=axes,
        min_coordinate=min_coordinate,
        max_coordinate=max_coordinate,
        target_unit_to_pixels=target_unit_to_pixels,
        target_width=target_width,
        target_height=target_height,
        target_depth=target_depth,
    )
    raise NotImplementedError()


@rasterize.register(GeoDataFrame)
def _(
    data: GeoDataFrame,
    axes: tuple[str, ...],
    min_coordinate: ArrayLike,
    max_coordinate: ArrayLike,
    target_coordinate_system: str,
    target_unit_to_pixels: Optional[float] = None,
    target_width: Optional[float] = None,
    target_height: Optional[float] = None,
    target_depth: Optional[float] = None,
) -> SpatialImage:
    target_width, target_height, target_depth = _compute_target_dimensions(
        spatial_axes=axes,
        min_coordinate=min_coordinate,
        max_coordinate=max_coordinate,
        target_unit_to_pixels=target_unit_to_pixels,
        target_width=target_width,
        target_height=target_height,
        target_depth=target_depth,
    )
    raise NotImplementedError()
