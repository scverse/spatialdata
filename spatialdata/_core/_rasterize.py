import numpy as np
from functools import singledispatch

from spatialdata import SpatialData
from typing import Union, Optional
from spatial_image import SpatialImage
from multiscale_spatial_image import MultiscaleSpatialImage
from spatialdata._types import ArrayLike
from spatialdata._core.core_utils import SpatialElement, get_dims, compute_coordinates, _get_scale
from spatialdata._core._spatialdata_ops import get_transformation, set_transformation, remove_transformation
from spatialdata._core.transformations import _get_affine_for_element, Sequence, Translation, Scale, Affine
from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
import dask_image.ndinterp
from spatialdata._core.models import Image2DModel, Image3DModel, Labels2DModel, Labels3DModel, get_schema
from dask.array.core import Array as DaskArray


def _compute_target_dimensions(
    axes: tuple[str, ...],
    min_coordinate: ArrayLike,
    max_coordinate: ArrayLike,
    target_unit_to_pixels: Optional[float],
    target_width: Optional[float],
    target_height: Optional[float],
    target_depth: Optional[float],
) -> tuple[float, float, float]:
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
    assert set(axes) == {"x", "y"} or set(axes) == {"x", "y", "z"}
    if "z" not in axes:
        assert target_depth is None

    x_index = axes.index("x")
    y_index = axes.index("y")
    w_bb = max_coordinate[x_index] - min_coordinate[x_index]
    h_bb = max_coordinate[y_index] - min_coordinate[y_index]
    assert w_bb > 0
    assert h_bb > 0
    w_to_h_bb = w_bb / h_bb

    d_bb = None
    d_to_h_bb = None
    if "z" in axes:
        z_index = axes.index("z")
        d_bb = max_coordinate[z_index] - min_coordinate[z_index]
        assert d_bb > 0
        d_to_h_bb = d_bb / h_bb

    if target_unit_to_pixels is not None:
        target_width = w_bb * target_unit_to_pixels
        target_height = h_bb * target_unit_to_pixels
        if "z" in axes:
            assert d_bb is not None
            target_depth = d_bb * target_unit_to_pixels
    elif target_width is not None:
        target_height = target_width / w_to_h_bb
        if "z" in axes:
            assert d_to_h_bb is not None
            target_depth = target_height * d_to_h_bb
    elif target_height is not None:
        target_width = target_height * w_to_h_bb
        if "z" in axes:
            assert d_to_h_bb is not None
            target_depth = target_height * d_to_h_bb
    elif target_depth is not None:
        target_height = target_depth / d_to_h_bb
        target_width = target_height * w_to_h_bb
    else:
        raise RuntimeError("Should not reach here")
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
    target_width, target_height, target_depth = _compute_target_dimensions(
        axes=axes,
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
    # get xdata
    if isinstance(data, SpatialImage):
        xdata = data.data
        pyramid_scale = None
    elif isinstance(data, MultiscaleSpatialImage):
        latest_scale: Optional[str] = None
        for scale in reversed(list(data.keys())):
            data_tree = data[scale]
            latest_scale = scale
            v = data_tree.values()
            assert len(v) == 1
            xdata = next(iter(v))
            assert set(xdata.sizes.keys()) == set(axes)
            for ax in axes:
                if xdata.sizes[ax] < target_sizes[ax]:
                    break
            else:
                break
        assert latest_scale is not None
        xdata = next(iter(data[latest_scale].values()))
        if latest_scale != 'scale0':
            transformations = xdata.attrs['transform']
            pyramid_scale = _get_scale(transformations)
        else:
            pyramid_scale = None
    else:
        raise RuntimeError("Should not reach here")

    # get transformation
    transformation = get_transformation(data, target_coordinate_system)
    dims = get_dims(data)
    affine = _get_affine_for_element(data, transformation)
    target_axes_unordered = affine.output_axes
    assert set(target_axes_unordered) in [{"x", "y", "z"}, {"x", "y"}]
    assert len(target_axes_unordered) == len(min_coordinate)
    assert len(target_axes_unordered) == len(max_coordinate)
    if "z" in target_axes_unordered:
        target_axes = ("z", "y", "x")
    else:
        target_axes = ("y", "x")
    corrected_affine = Affine(
        affine.to_affine_matrix(input_axes=dims, output_axes=target_axes), input_axes=dims, output_axes=target_axes
    )

    bb_sizes = {
        ax: max_coordinate[axes.index(ax)] - min_coordinate[axes.index(ax)]
        for ax in target_axes
    }
    scale_vector = [bb_sizes[ax] / target_sizes[ax] for ax in target_axes]
    scale = Scale(scale_vector, axes=target_axes)

    offset = [min_coordinate[target_axes_unordered.index(ax)] for ax in target_axes]
    translation = Translation(offset, axes=target_axes)

    if pyramid_scale is not None:
        extra = [pyramid_scale.inverse()]
    else:
        extra = []
    sequence = Sequence([scale, translation, corrected_affine.inverse()] + extra)
    matrix = sequence.to_affine_matrix(input_axes=target_axes, output_axes=dims)

    # get output shape
    output_shape = tuple(int(target_sizes[ax]) for ax in dims)

    # get kwargs and schema
    schema = get_schema(data)
    # labels need to be preserved after the resizing of the image
    if schema == Labels2DModel or schema == Labels3DModel:
        # TODO: this should work, test better
        kwargs = {"prefilter": False}
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
    transformed_data = schema.parse(transformed_dask, dims=axes)  # type: ignore[call-arg,arg-type]
    if target_coordinate_system != "global":
        remove_transformation(transformed_data, "global")

    sequence = Sequence([scale, translation])
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
        axes=axes,
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
        axes=axes,
        min_coordinate=min_coordinate,
        max_coordinate=max_coordinate,
        target_unit_to_pixels=target_unit_to_pixels,
        target_width=target_width,
        target_height=target_height,
        target_depth=target_depth,
    )
    raise NotImplementedError()
