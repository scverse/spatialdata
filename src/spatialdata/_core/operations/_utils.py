from __future__ import annotations

from typing import TYPE_CHECKING

from xarray import DataArray, DataTree

from spatialdata.models import SpatialElement, get_axes_names, get_spatial_axes

if TYPE_CHECKING:
    from spatialdata._core.spatialdata import SpatialData


def transform_to_data_extent(
    sdata: SpatialData,
    coordinate_system: str,
    maintain_positioning: bool = True,
    target_unit_to_pixels: float | None = None,
    target_width: float | None = None,
    target_height: float | None = None,
    target_depth: float | None = None,
) -> SpatialData:
    """
    Transform the spatial data to match the data extent, so that pixels and vector coordinates correspond.

    Given a selected coordinate system, this function will transform the spatial data in that coordinate system, and
    will resample images, so that the pixels and vector coordinates correspond.
    In other words, the vector coordinate (x, y) (or (x, y, z)) will correspond to the pixel (y, x) (or (z, y, x)).

    When `maintain_positioning` is `False`, each transformation will be set to Identity. When `maintain_positioning` is
    `True` (default value), each element of the data will also have a transformation that will maintain the positioning
    of the element, as it was before calling this function.
    Note that in this case the correspondence between pixels and vector coordinates is true in the intrinsic coordinate
    system, not in the target coordinate system.

    Parameters
    ----------
    sdata
        The spatial data to transform.
    coordinate_system
        The coordinate system to use to compute the extent and to transform the data to.
    maintain_positioning
        If `True`, the transformation will maintain the positioning of the elements, as it was before calling this
        function. If `False`, each transformation will be set to Identity.
    target_unit_to_pixels
        The required number of pixels per unit (units in the target coordinate system) of the data that will be
        produced.
    target_width
        The width of the data extent, in pixels, for the data that will be produced.
    target_height
        The height of the data extent, in pixels, for the data that will be produced.
    target_depth
        The depth of the data extent, in pixels, for the data that will be produced.

    Returns
    -------
    SpatialData
        The transformed spatial data with downscaled and padded images and adjusted vector coordinates; all the
        transformations will set to Identity and the coordinates of the vector data will be aligned to the pixel
        coordinates.

    Notes
    -----
        - The data extent is the smallest rectangle that contains all the images and geometries.
        - DataTree objects (multiscale images) will be converted to DataArray (single-scale images) objects.
        - This helper function will be deprecated when https://github.com/scverse/spatialdata/issues/308 is closed,
          as this function will be easily recovered by `transform_to_coordinate_system()`
    """
    from spatialdata._core.data_extent import get_extent
    from spatialdata._core.operations.rasterize import _compute_target_dimensions, rasterize
    from spatialdata._core.spatialdata import SpatialData
    from spatialdata.transformations.operations import get_transformation, set_transformation
    from spatialdata.transformations.transformations import BaseTransformation, Identity, Scale, Sequence, Translation

    sdata = sdata.filter_by_coordinate_system(coordinate_system=coordinate_system)
    # calling transform_to_coordinate_system will likely decrease the resolution, let's use rasterize() instead
    sdata_vector = SpatialData(shapes=dict(sdata.shapes), points=dict(sdata.points))
    sdata_raster = SpatialData(images=dict(sdata.images), labels=dict(sdata.labels))
    sdata_vector_transformed = sdata_vector.transform_to_coordinate_system(coordinate_system)

    data_extent = get_extent(sdata, coordinate_system=coordinate_system)
    data_extent_axes = tuple(data_extent.keys())
    translation_to_origin = Translation([-data_extent[ax][0] for ax in data_extent_axes], axes=data_extent_axes)

    sizes = [data_extent[ax][1] - data_extent[ax][0] for ax in data_extent_axes]
    target_width, target_height, target_depth = _compute_target_dimensions(
        spatial_axes=data_extent_axes,
        min_coordinate=[0 for _ in data_extent_axes],
        max_coordinate=sizes,
        target_unit_to_pixels=target_unit_to_pixels,
        target_width=target_width,
        target_height=target_height,
        target_depth=target_depth,
    )
    scale_to_target_d = {
        "x": target_width / sizes[data_extent_axes.index("x")],
        "y": target_height / sizes[data_extent_axes.index("y")],
    }
    if target_depth is not None:
        scale_to_target_d["z"] = target_depth / sizes[data_extent_axes.index("z")]
    scale_to_target = Scale([scale_to_target_d[ax] for ax in data_extent_axes], axes=data_extent_axes)

    for el in sdata_vector_transformed._gen_spatial_element_values():
        t = get_transformation(el, to_coordinate_system=coordinate_system)
        assert isinstance(t, BaseTransformation)
        sequence = Sequence([t, translation_to_origin, scale_to_target])
        set_transformation(el, transformation=sequence, to_coordinate_system=coordinate_system)
    sdata_vector_transformed_inplace = sdata_vector_transformed.transform_to_coordinate_system(
        coordinate_system, maintain_positioning=True
    )

    sdata_to_return_elements = {
        **sdata_vector_transformed_inplace.shapes,
        **sdata_vector_transformed_inplace.points,
    }

    for _, element_name, element in sdata_raster.gen_spatial_elements():
        element_axes = get_spatial_axes(get_axes_names(element))
        if isinstance(element, DataArray | DataTree):
            rasterized = rasterize(
                element,
                axes=element_axes,
                min_coordinate=[data_extent[ax][0] for ax in element_axes],
                max_coordinate=[data_extent[ax][1] for ax in element_axes],
                target_coordinate_system=coordinate_system,
                target_unit_to_pixels=None,
                target_width=target_width,
                target_height=None,
                target_depth=None,
                return_regions_as_labels=True,
            )
            sdata_to_return_elements[element_name] = rasterized
        else:
            sdata_to_return_elements[element_name] = element
    if not maintain_positioning:
        for el in sdata_to_return_elements.values():
            set_transformation(el, transformation={coordinate_system: Identity()}, set_all=True)
    for k, v in sdata.tables.items():
        sdata_to_return_elements[k] = v.copy()
    return SpatialData.init_from_elements(sdata_to_return_elements, attrs=sdata.attrs)


def _parse_element(
    element: str | SpatialElement, sdata: SpatialData | None, element_var_name: str, sdata_var_name: str
) -> SpatialElement:
    if not ((sdata is not None and isinstance(element, str)) ^ (not isinstance(element, str))):
        raise ValueError(
            f"To specify the {element_var_name!r} SpatialElement, please do one of the following: "
            f"- either pass a SpatialElement to the {element_var_name!r} parameter (and keep "
            f"`{sdata_var_name}` = None);"
            f"- either `{sdata_var_name}` needs to be a SpatialData object, and {element_var_name!r} needs "
            f"to be the string name of the element."
        )
    if sdata is not None:
        assert isinstance(element, str)
        return sdata[element]
    assert element is not None
    return element
