from __future__ import annotations

from typing import TYPE_CHECKING

from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage

if TYPE_CHECKING:
    from spatialdata._core.spatialdata import SpatialData


def transform_to_data_extent(
    sdata: SpatialData,
    coordinate_system: str,
    target_unit_to_pixels: float | None = None,
    target_width: float | None = None,
    target_height: float | None = None,
    target_depth: float | None = None,
) -> SpatialData:
    """
    Transform the spatial data to match the data extent, keeping the positioning, and making all transformations equal.

    Parameters
    ----------
    sdata
        The spatial data to transform.
    coordinate_system
        The coordinate system to use to compute the extent and to transform the data to.
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
        - MultiscaleSpatialImage objects will be converted to SpatialImage objects.
        - This helper function will be deprecated when https://github.com/scverse/spatialdata/issues/308 is closed,
          as this function will be easily recovered by `transform_to_coordinate_system()`
    """
    # TODO: change "all the elements have idenity" with "all the elements have the same transformation"
    from spatialdata._core.data_extent import get_extent
    from spatialdata._core.operations.rasterize import _compute_target_dimensions, rasterize
    from spatialdata._core.spatialdata import SpatialData
    from spatialdata.transformations.operations import get_transformation, set_transformation
    from spatialdata.transformations.transformations import BaseTransformation, Scale, Sequence, Translation

    sdata = sdata.filter_by_coordinate_system(coordinate_system=coordinate_system)
    # calling transform_to_coordinate_system will likely decrease the resolution, let's use rasterize() instead
    sdata_vector = SpatialData(shapes=dict(sdata.shapes), points=dict(sdata.points))
    sdata_raster = SpatialData(images=dict(sdata.images), labels=dict(sdata.labels))
    sdata_vector_transformed = sdata_vector.transform_to_coordinate_system(coordinate_system)

    de = get_extent(sdata, coordinate_system=coordinate_system)
    de_axes = tuple(de.keys())
    translation_to_origin = Translation([-de[ax][0] for ax in de_axes], axes=de_axes)

    sizes = [de[ax][1] - de[ax][0] for ax in de_axes]
    target_width, target_height, target_depth = _compute_target_dimensions(
        spatial_axes=de_axes,
        min_coordinate=[0 for _ in de_axes],
        max_coordinate=sizes,
        target_unit_to_pixels=target_unit_to_pixels,
        target_width=target_width,
        target_height=target_height,
        target_depth=target_depth,
    )
    scale_to_target_d = {"x": target_width / sizes[de_axes.index("x")], "y": target_height / sizes[de_axes.index("y")]}
    if target_depth is not None:
        scale_to_target_d["z"] = target_depth / sizes[de_axes.index("z")]
    scale_to_target = Scale([scale_to_target_d[ax] for ax in de_axes], axes=de_axes)

    for el in sdata_vector_transformed._gen_elements_values():
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

    for _, element_name, element in sdata_raster._gen_elements():
        if isinstance(element, (MultiscaleSpatialImage, SpatialImage)):
            rasterized = rasterize(
                element,
                axes=de_axes,
                min_coordinate=[de[ax][0] for ax in de_axes],
                max_coordinate=[de[ax][1] for ax in de_axes],
                target_coordinate_system=coordinate_system,
                target_unit_to_pixels=None,
                target_width=target_width,
                target_height=None,
                target_depth=None,
            )
            sdata_to_return_elements[element_name] = rasterized
        else:
            sdata_to_return_elements[element_name] = element
    if sdata.table is not None:
        sdata_to_return_elements["table"] = sdata.table
    return SpatialData.from_elements_dict(sdata_to_return_elements)
