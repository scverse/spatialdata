from functools import partial

import numpy as np
import xarray as xr
from geopandas import GeoDataFrame
from xarray.core.dataarray import DataArray
from xarray.core.datatree import DataTree

from spatialdata.models import Labels2DModel, ShapesModel


def _mask_block(block: xr.DataArray, ids_to_remove: list[int]) -> xr.DataArray:
    # Use apply_ufunc for efficient processing
    # Create a copy to avoid modifying read-only array
    result = block.copy()
    result[np.isin(result, ids_to_remove)] = 0
    return result


def _set_instance_ids_in_labels_to_zero(image: xr.DataArray, ids_to_remove: list[int]) -> xr.DataArray:
    processed = xr.apply_ufunc(
        partial(_mask_block, ids_to_remove=ids_to_remove),
        image,
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[image.dtype],
        dataset_fill_value=0,
        dask_gufunc_kwargs={"allow_rechunk": True},
    )

    # Force computation to ensure the changes are materialized
    computed_result = processed.compute()

    # Create a new DataArray to ensure persistence
    return xr.DataArray(
        data=computed_result.data,
        coords=image.coords,
        dims=image.dims,
        attrs=image.attrs.copy(),  # Preserve all attributes
    )


def _get_scale_factors(labels_element: Labels2DModel) -> list[tuple[float, float]]:
    scales = list(labels_element.keys())

    # Calculate relative scale factors between consecutive scales
    scale_factors = []
    for i in range(len(scales) - 1):
        y_size_current = labels_element[scales[i]].image.shape[0]
        x_size_current = labels_element[scales[i]].image.shape[1]
        y_size_next = labels_element[scales[i + 1]].image.shape[0]
        x_size_next = labels_element[scales[i + 1]].image.shape[1]
        y_factor = y_size_current / y_size_next
        x_factor = x_size_current / x_size_next

        scale_factors.append((y_factor, x_factor))

    return scale_factors


def filter_shapesmodel_by_instance_ids(element: ShapesModel, ids_to_remove: list[str]) -> GeoDataFrame:
    """
    Filter a ShapesModel by instance ids.

    Parameters
    ----------
    element
        The ShapesModel to filter.
    ids_to_remove
        The instance ids to remove.

    Returns
    -------
    The filtered ShapesModel.
    """
    element2: GeoDataFrame = element[~element.index.isin(ids_to_remove)]  # type: ignore[index, attr-defined]
    return ShapesModel.parse(element2)


def filter_labels2dmodel_by_instance_ids(element: Labels2DModel, ids_to_remove: list[int]) -> DataArray | DataTree:
    """
    Filter a Labels2DModel by instance ids.

    This function works for both DataArray and DataTree and sets the
    instance ids to zero.

    Parameters
    ----------
    element
        The Labels2DModel to filter.
    ids_to_remove
        The instance ids to remove.

    Returns
    -------
    The filtered Labels2DModel.
    """
    if isinstance(element, xr.DataArray):
        return Labels2DModel.parse(_set_instance_ids_in_labels_to_zero(element, ids_to_remove))

    if isinstance(element, DataTree):
        # we extract the info to just reconstruct
        # the DataTree after filtering the max scale
        max_scale = list(element.keys())[0]
        scale_factors_temp = _get_scale_factors(element)
        scale_factors = [int(sf[0]) for sf in scale_factors_temp]

        return Labels2DModel.parse(
            data=_set_instance_ids_in_labels_to_zero(element[max_scale].image, ids_to_remove),
            scale_factors=scale_factors,
        )
    raise ValueError(f"Unknown element type: {type(element)}")
