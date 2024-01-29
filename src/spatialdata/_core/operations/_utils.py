from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

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
    Transform the spatial data to match the data extent and setting all the transformations to Identity.

    Parameters
    ----------
    sdata
        The spatial data to transform.
    coordinate_system
        The coordinate system to use to compute the extent and to transform the data.
    target_unit_to_pixels
        The number of pixels per unit of the target coordinate system.
    target_width
        The width of the data extent in the units of the selected coordinate system
    target_height
        The height of the data extent in the units of the selected coordinate system
    target_depth
        The depth of the data extent in the units of the selected coordinate system


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
    from spatialdata._core.operations.rasterize import _compute_target_dimensions
    from spatialdata._core.spatialdata import SpatialData
    from spatialdata.transformations.operations import get_transformation, set_transformation
    from spatialdata.transformations.transformations import BaseTransformation, Scale, Sequence, Translation

    sdata = sdata.filter_by_coordinate_system(coordinate_system=coordinate_system)
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

    sdata_transformed = sdata.transform_to_coordinate_system(coordinate_system)

    random_uuid = str(uuid.uuid4())
    for el in sdata_transformed._gen_elements_values():
        t = get_transformation(el, to_coordinate_system=coordinate_system)
        assert isinstance(t, BaseTransformation)
        sequence = Sequence([t, translation_to_origin, scale_to_target])
        set_transformation(el, transformation=sequence, to_coordinate_system=random_uuid)
    sdata_transformed_inplace = sdata_transformed.transform_to_coordinate_system(random_uuid, maintain_positioning=True)
    print(get_extent(sdata_transformed_inplace, coordinate_system=random_uuid))

    sdata_to_return_elements = {}

    for _, element_name, element in sdata_transformed._gen_elements():
        # debugging
        sdata_to_return_elements[element_name] = element
    #     # the call of transform_to_coordinate_system() above didn't remove any coordinate system but we just need
    #     # the coordinate system `random_uuid`; let's remove all the other coordinate systems
    #     t = get_transformation(element, to_coordinate_system=random_uuid)
    #     assert isinstance(t, BaseTransformation)
    #     set_transformation(element, transformation={random_uuid: t}, set_all=True)
    #
    #     # let's resample (with rasterize()) the raster data into the target dimensions, let's keep the vector data as
    #     # it is (since it has been already transformed)
    #     if isinstance(element, (MultiscaleSpatialImage, SpatialImage)):
    #         rasterized = rasterize(
    #             element,
    #             axes=de_axes,
    #             min_coordinate=[de[ax][0] for ax in de_axes],
    #             max_coordinate=[de[ax][1] for ax in de_axes],
    #             target_coordinate_system=random_uuid,
    #             target_unit_to_pixels=None,
    #             target_width=target_width,
    #             target_height=None,
    #             target_depth=None,
    #         )
    #         sdata_to_return_elements[element_name] = rasterized
    #     else:
    #         sdata_to_return_elements[element_name] = element
    if sdata_transformed_inplace.table is not None:
        sdata_to_return_elements["table"] = sdata_transformed_inplace.table
    sdata_to_return = SpatialData.from_elements_dict(sdata_to_return_elements)

    from napari_spatialdata import Interactive

    Interactive([sdata_to_return, sdata, sdata_transformed])
    pass
