from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from anndata import AnnData
from dask.array import Array
from dask.dataframe import DataFrame
from geopandas import GeoDataFrame
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage

import spatialdata
from spatialdata._logging import logger
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import BaseTransformation, set_transformation

if TYPE_CHECKING:
    from spatialdata import SpatialData


class LayerManager(ABC):
    def add_layer(
        self,
        sdata: SpatialData,
        arr: Array,
        output_layer: str,
        dims: tuple[str, ...] | None = None,
        chunks: str | tuple[int, ...] | int | None = None,
        transformation: BaseTransformation | dict[str, BaseTransformation] = None,
        scale_factors: ScaleFactors_t | None = None,
        overwrite: bool = False,
        **kwargs: Any,  # kwargs passed to create_spatial_element
    ) -> SpatialData:
        chunks = chunks or arr.chunksize
        if dims is None:
            logger.warning("No dims parameter specified. "
                           "Assuming order of dimension of provided array is (c, (z), y, x)")
            dims = self.get_dims(arr)

        intermediate_output_layer = None
        if scale_factors is not None:
            if sdata.is_backed():
                spatial_element = self.create_spatial_element(
                    arr,
                    dims=dims,
                    scale_factors=None,
                    chunks=chunks,
                    **kwargs,
                )
                if transformation is not None:
                    set_transformation(spatial_element, transformation)

                intermediate_output_layer = f"{uuid.uuid4()}_{output_layer}"
                logger.info(f"Writing intermediate non-multiscale results to layer '{intermediate_output_layer}'")
                sdata = self.add_to_sdata(
                    sdata,
                    output_layer=intermediate_output_layer,
                    spatial_element=spatial_element,
                    overwrite=False,
                )
                arr = self.retrieve_data_from_sdata(sdata, intermediate_output_layer)
            else:
                arr = arr.persist()

        elif not sdata.is_backed():
            # if sdata is not backed, and if no scale factors, we also need to do a persist
            # to prevent recomputation
            arr = arr.persist()

        spatial_element = self.create_spatial_element(
            arr,
            dims=dims,
            scale_factors=scale_factors,
            chunks=chunks,
            **kwargs,
        )

        if transformation is not None:
            set_transformation(spatial_element, transformation)

        logger.info(f"Writing results to layer '{output_layer}'")

        sdata = self.add_to_sdata(
            sdata,
            output_layer=output_layer,
            spatial_element=spatial_element,
            overwrite=overwrite,
        )

        if intermediate_output_layer:
            logger.info(
                f"Removing intermediate output layer '{intermediate_output_layer}' "
                f"from .zarr store at path {sdata.path}."
            )
            del sdata[intermediate_output_layer]
            if sdata.is_backed():
                sdata.delete_element_from_disk(intermediate_output_layer)

        return sdata

    @abstractmethod
    def create_spatial_element(
        self,
        arr: Array,
        dims: tuple[int, ...],
        scale_factors: ScaleFactors_t | None = None,
        chunks: str | tuple[int, ...] | int | None = None,
        **kwargs: Any,
    ) -> SpatialImage | MultiscaleSpatialImage:
        pass

    @abstractmethod
    def get_dims(self) -> tuple[str, ...]:
        pass

    @abstractmethod
    def add_to_sdata(
        self,
        sdata: SpatialData,
        output_layer: str,
        spatial_element: SpatialImage | MultiscaleSpatialImage,
        overwrite: bool = False,
    ) -> SpatialData:
        pass

    @abstractmethod
    def retrieve_data_from_sdata(self, sdata: SpatialData, name: str) -> SpatialData:
        pass


class ImageLayerManager(LayerManager):
    def create_spatial_element(
        self,
        arr: Array,
        dims: tuple[str, ...],
        scale_factors: ScaleFactors_t | None = None,
        chunks: str | tuple[int, ...] | int | None = None,
        c_coords: list[str] | None = None,
    ) -> SpatialImage | MultiscaleSpatialImage:
        if len(dims) == 3:
            return spatialdata.models.Image2DModel.parse(
                arr,
                dims=dims,
                scale_factors=scale_factors,
                chunks=chunks,
                c_coords=c_coords,
            )
        if len(dims) == 4:
            return spatialdata.models.Image3DModel.parse(
                arr,
                dims=dims,
                scale_factors=scale_factors,
                chunks=chunks,
                c_coords=c_coords,
            )
        raise ValueError(
            f"Provided dims is {dims}, which is not supported, "
            "please provide dims parameter that only contains c, (z), y, and x."
        )

    def get_dims(self, arr) -> tuple[str, ...]:
        if len(arr.shape) == 3:
            return ("c", "y", "x")
        if len(arr.shape) == 4:
            return ("c", "z", "y", "x")
        raise ValueError("Only 2D and 3D images (c, (z), y, x) are currently supported.")

    def add_to_sdata(
        self,
        sdata: SpatialData,
        output_layer: str,
        spatial_element: SpatialImage | MultiscaleSpatialImage,
        overwrite: bool = False,
    ) -> SpatialData:
        from spatialdata import read_zarr
        # given a spatial_element with some graph defined on it.
        if output_layer in [*sdata.images]:
            if sdata.is_backed():
                if overwrite:
                    sdata = _incremental_io_on_disk(sdata, output_layer=output_layer, element=spatial_element)
                else:
                    raise ValueError(
                        f"Attempting to overwrite sdata.images[{output_layer}], but overwrite is set to False. "
                        "Set overwrite to True to overwrite the .zarr store."
                    )
            else:
                sdata[output_layer] = spatial_element

        else:
            sdata[output_layer] = spatial_element
            if sdata.is_backed():
                sdata.write_element(output_layer)
                sdata = read_zarr(sdata.path)

        return sdata

    def retrieve_data_from_sdata(self, sdata: SpatialData, name: str) -> Array:
        return sdata.images[name].data


class LabelLayerManager(LayerManager):
    def create_spatial_element(
        self,
        arr: Array,
        dims: tuple[str, ...],
        scale_factors: ScaleFactors_t | None = None,
        chunks: str | tuple[int, ...] | int | None = None,
    ) -> SpatialImage | MultiscaleSpatialImage:
        if len(dims) == 2:
            return spatialdata.models.Labels2DModel.parse(
                arr,
                dims=dims,
                scale_factors=scale_factors,
                chunks=chunks,
            )
        if len(dims) == 3:
            return spatialdata.models.Labels3DModel.parse(
                arr,
                dims=dims,
                scale_factors=scale_factors,
                chunks=chunks,
            )
        raise ValueError(
            f"Provided dims is {dims}, which is not supported, "
            "please provide dims parameter that only contains (z), y and x."
        )

    def get_dims(self, arr) -> tuple[str, ...]:
        if len(arr.shape) == 2:
            return ("y", "x")
        if len(arr.shape) == 3:
            return ("z", "y", "x")
        raise ValueError("Only 2D and 3D labels layers ( (z), y, x) are currently supported.")

    def add_to_sdata(
        self,
        sdata: SpatialData,
        output_layer: str,
        spatial_element: SpatialImage | MultiscaleSpatialImage,
        overwrite: bool = False,
    ) -> SpatialData:
        from spatialdata import read_zarr
        # given a spatial_element with some graph defined on it.
        if output_layer in [*sdata.labels]:
            if sdata.is_backed():
                if overwrite:
                    sdata = _incremental_io_on_disk(sdata, output_layer=output_layer, element=spatial_element)
                else:
                    raise ValueError(
                        f"Attempting to overwrite sdata.labels[{output_layer}], "
                        "but overwrite is set to False. Set overwrite to True to overwrite the .zarr store."
                    )
            else:
                sdata[output_layer] = spatial_element
        else:
            sdata[output_layer] = spatial_element
            if sdata.is_backed():
                sdata.write_element(output_layer)
                sdata = read_zarr(sdata.path)

        return sdata

    def retrieve_data_from_sdata(self, sdata: SpatialData, name: str) -> Array:
        return sdata.labels[name].data


def _incremental_io_on_disk(
    sdata: SpatialData,
    output_layer: str,
    element: SpatialImage | MultiscaleSpatialImage | DataFrame | GeoDataFrame | AnnData,
)->spatialdata:
    from spatialdata import read_zarr
    new_output_layer = f"{output_layer}_{uuid.uuid4()}"
    # a. write a backup copy of the data
    sdata[new_output_layer] = element
    try:
        sdata.write_element(new_output_layer)
    except Exception as e:
        if new_output_layer in sdata[new_output_layer]:
            del sdata[new_output_layer]
        raise e
    # a2. remove the in-memory copy from the SpatialData object (note,
    # at this point the backup copy still exists on-disk)
    del sdata[new_output_layer]
    del sdata[output_layer]
    # a3 load the backup copy into memory
    sdata_copy = read_zarr(sdata.path)
    # b1. rewrite the original data
    sdata.delete_element_from_disk(output_layer)
    sdata[output_layer] = sdata_copy[new_output_layer]
    logger.warning(f"layer with name '{output_layer}' already exists. Overwriting...")
    sdata.write_element(output_layer)
    # b2. reload the new data into memory (because it has been written but in-memory it still points
    # from the backup location)
    sdata = read_zarr(sdata.path)
    # c. remove the backup copy
    del sdata[new_output_layer]
    sdata.delete_element_from_disk(new_output_layer)

    return sdata
