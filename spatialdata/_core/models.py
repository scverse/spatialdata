"""This file contains models and schema for SpatialData"""

from functools import singledispatchmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from anndata import AnnData
from dask.array.core import Array as DaskArray
from dask.array.core import from_array
from geopandas import GeoDataFrame
from geopandas.array import GeometryDtype
from multiscale_spatial_image import to_multiscale
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from multiscale_spatial_image.to_multiscale.to_multiscale import Methods
from numpy.typing import ArrayLike
from pandas.api.types import is_categorical_dtype
from pandera import SchemaModel
from pandera.typing.geopandas import GeoSeries
from shapely import GeometryType
from shapely.io import from_geojson, from_ragged_array
from spatial_image import SpatialImage, to_spatial_image
from xarray_schema.components import (
    ArrayTypeSchema,
    AttrSchema,
    AttrsSchema,
    DimsSchema,
)
from xarray_schema.dataarray import DataArraySchema

from spatialdata._core.transformations import BaseTransformation, Identity
from spatialdata._logging import logger

# Types
Chunks_t = Union[
    int,
    Tuple[int, ...],
    Tuple[Tuple[int, ...], ...],
    Mapping[Any, Union[None, int, Tuple[int, ...]]],
]
ScaleFactors_t = Sequence[Union[Dict[str, int], int]]

Z, C, Y, X = "z", "c", "y", "x"

Transform_s = AttrSchema(BaseTransformation, None)


class RasterSchema(DataArraySchema):
    """Base schema for raster data."""

    @classmethod
    def parse(
        cls,
        data: ArrayLike,
        transform: Optional[BaseTransformation] = None,
        scale_factors: Optional[ScaleFactors_t] = None,
        method: Optional[Methods] = None,
        chunks: Optional[Chunks_t] = None,
        **kwargs: Any,
    ) -> Union[SpatialImage, MultiscaleSpatialImage]:
        """
        Validate (or parse) raster data.

        Parameters
        ----------
        data
            Data to validate.
        transform
            Transformation to apply to the data.
        scale_factors
            Scale factors to apply for multiscale.
            If not None, a :class:`multiscale_spatial_image.multiscale_spatial_image.MultiscaleSpatialImage` is returned.
        method
            Method to use for multiscale.
        chunks
            Chunks to use for dask array.

        Returns
        -------
        :class:`spatial_image.SpatialImage` or
        :class:`multiscale_spatial_image.multiscale_spatial_image.MultiscaleSpatialImage`.
        """
        if not isinstance(data, DaskArray):
            data = from_array(data)
        print(cls.dims.dims)
        data = to_spatial_image(array_like=data, dims=cls.dims.dims, **kwargs)
        if transform is None:
            transform = Identity()
        if TYPE_CHECKING:
            assert isinstance(data, SpatialImage) or isinstance(data, MultiscaleSpatialImage)
        data.attrs = {"transform": transform}
        if scale_factors is not None:
            data = to_multiscale(
                data,
                scale_factors=scale_factors,
                method=method,
                chunks=chunks,
            )
        return data

    def validate(self, data: Union[SpatialImage, MultiscaleSpatialImage]) -> None:
        if isinstance(data, SpatialImage):
            super().validate(data)
        elif isinstance(data, MultiscaleSpatialImage):
            name = {list(data[i].data_vars.keys())[0] for i in data.keys()}
            if len(name) > 1:
                raise ValueError(f"Wrong name for datatree: {name}.")
            name = list(name)[0]
            for d in data:
                super().validate(data[d][name])


class Label2D(RasterSchema):
    dims = DimsSchema((Y, X))
    array_type = ArrayTypeSchema(DaskArray)
    attrs = AttrsSchema({"transform": Transform_s})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            dims=self.dims,
            array_type=self.array_type,
            attrs=self.attrs,
            *args,
            **kwargs,
        )


class Label3D(RasterSchema):
    dims = DimsSchema((Z, Y, X))
    array_type = ArrayTypeSchema(DaskArray)
    attrs = AttrsSchema({"transform": Transform_s})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            dims=self.dims,
            array_type=self.array_type,
            attrs=self.attrs,
            *args,
            **kwargs,
        )


class Image2D(RasterSchema):
    dims = DimsSchema((C, Y, X))
    array_type = ArrayTypeSchema(DaskArray)
    attrs = AttrsSchema({"transform": Transform_s})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            dims=self.dims,
            array_type=self.array_type,
            attrs=self.attrs,
            *args,
            **kwargs,
        )


class Image3D(RasterSchema):
    dims = DimsSchema((Z, Y, X, C))
    array_type = ArrayTypeSchema(DaskArray)
    attrs = AttrsSchema({"transform": Transform_s})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            dims=self.dims,
            array_type=self.array_type,
            attrs=self.attrs,
            *args,
            **kwargs,
        )


# TODO: should check for column be strict?
# TODO: validate attrs for transform.
class Polygon(SchemaModel):
    geometry: GeoSeries[GeometryDtype]

    @singledispatchmethod
    @classmethod
    def parse(cls, data: Any, **kwargs: Any) -> GeoDataFrame:
        raise NotImplementedError

    @parse.register
    @classmethod
    def _(
        cls,
        data: np.ndarray,  # type: ignore[type-arg]
        offsets: Tuple[np.ndarray, ...],  # type: ignore[type-arg]
        geometry: Literal[3, 6],
        transform: Optional[Any] = None,
        **kwargs: Any,
    ) -> GeoDataFrame:

        geometry = GeometryType(geometry)

        data = from_ragged_array(geometry, data, offsets)
        geo_df = GeoDataFrame({"geometry": data})
        if transform is None:
            transform = Identity()
        geo_df.attrs = {"transform": transform}
        cls.validate(data)
        return geo_df

    @parse.register
    @classmethod
    def _(
        cls,
        data: str,
        transform: Optional[Any] = None,
        **kwargs: Any,
    ) -> GeoDataFrame:

        data = from_geojson(data)
        geo_df = GeoDataFrame({"geometry", data})
        if transform is None:
            transform = Identity()
        geo_df.attrs = {"transform": transform}
        cls.validate(data)
        return geo_df

    @parse.register
    @classmethod
    def _(
        cls,
        data: GeoDataFrame,
        transform: Optional[Any] = None,
        **kwargs: Any,
    ) -> GeoDataFrame:

        if transform is None:
            transform = Identity()
        data.attrs = {"transform": transform}
        cls.validate(data)
        return data


class Shape:
    coords_key = "spatial"
    transform_key = "transform"
    attrs_key = "spatialdata_attrs"

    def validate(self, data: AnnData) -> None:
        if self.coords_key not in data.obsm:
            raise ValueError(f"AnnData does not contain shapes coordinates in `adata.obsm['{self.coords_key}']`.")
        if self.transform_key not in data.uns:
            raise ValueError(f"AnnData does not contain `{self.transform_key}`.")
        if self.attrs_key not in data.uns:
            raise ValueError(f"AnnData does not contain `{self.attrs_key}`.")
        if "type" not in data.uns[self.attrs_key]:
            raise ValueError(f"AnnData does not contain `{self.attrs_key}['type']`.")
        if "size" not in data.uns[self.attrs_key]:
            raise ValueError(f"AnnData does not contain `{self.attrs_key}['size']`.")

    @classmethod
    def parse(
        cls,
        coords: np.ndarray,  # type: ignore[type-arg]
        shape_type: Literal["Circle", "Square"],
        shape_size: np.float_,
        transform: Optional[Any] = None,
        **kwargs: Any,
    ) -> AnnData:
        """
        Parse shape data into SpatialData.

        Parameters
        ----------
        coords
            Coordinates of shapes.
        shape_type
            Type of shape.
        shape_size
            Size of shape.
        transform
            Transform of shape.
        kwargs
            Additional arguments for shapes.

        Returns
        -------
        :class:`anndata.AnnData` formatted for shapes elements.
        """

        adata = AnnData(
            None,
            obs=pd.DataFrame(index=np.arange(coords.shape[0])),
            var=pd.DataFrame(index=np.arange(1)),
            **kwargs,
        )
        adata.obsm[cls.coords_key] = coords
        if transform is None:
            transform = Identity()
        adata.uns[cls.transform_key] = transform
        adata.uns[cls.attrs_key] = {"type": shape_type, "size": shape_size}
        return adata


class Point:
    coords_key = "spatial"
    transform_key = "transform"

    def validate(self, data: AnnData) -> None:
        if self.coords_key not in data.obsm:
            raise ValueError(f"AnnData does not contain points coordinates in `adata.obsm['{self.coords_key}']`.")
        if self.transform_key not in data.uns:
            raise ValueError(f"AnnData does not contain `{self.transform_key}`.")

    @classmethod
    def parse(
        cls,
        coords: np.ndarray,  # type: ignore[type-arg]
        var_names: Sequence[str],
        transform: Optional[Any] = None,
        **kwargs: Any,
    ) -> AnnData:

        adata = AnnData(
            None,
            obs=pd.DataFrame(index=np.arange(coords.shape[0])),
            var=pd.DataFrame(index=var_names),
            **kwargs,
        )
        adata.obsm[cls.coords_key] = coords
        if transform is None:
            transform = Identity()
        adata.uns[cls.transform_key] = transform
        return adata


class Table:
    def validate(
        data: AnnData,
        region: Optional[Union[str, Sequence[str]]] = None,
        region_key: Optional[str] = None,
        instance_key: Optional[str] = None,
        **kwargs: Any,
    ) -> AnnData:
        # TODO: is there enough validation?
        if "spatialdata_attr" not in data.uns:
            if isinstance(region, str):
                if region_key is not None or instance_key is not None:
                    logger.warning(
                        "`region` is of type `str` but `region_key` or `instance_key` found. They will be discarded."
                    )

            elif isinstance(region, list):
                if region_key is None:
                    raise ValueError("`region_key` must be provided if `region` is of type `Sequence`.")
                if region_key not in data.obs:
                    raise ValueError(f"Region key {region_key} not found in `adata.obs`.")
                if instance_key is None:
                    raise ValueError("`instance_key` must be provided if `region` is of type `Sequence`.")
                if instance_key not in data.obs:
                    raise ValueError(f"Instance key {instance_key} not found in `adata.obs`.")
                if not data.obs[region_key].isin(region).all():
                    raise ValueError(f"`Region key: {region_key}` values do not match with `region` values.")
                if not is_categorical_dtype(data.obs[region_key]):
                    logger.warning(f"Converting `region_key: {region_key}` to categorical dtype.")
                    data.obs["region_key"] = pd.Categorical(data.obs[region_key])
                # TODO: should we check for `instance_key` values?

            attr = {"region": region, "region_key": region_key, "instance_key": instance_key}
            data.uns["spatialdata_attr"] = attr
            return data
        else:
            attr = data.uns["spatialdata_attr"]
            if "region" not in attr:
                raise ValueError("`region` not found in `adata.uns['spatialdata_attr']`.")
            if isinstance(attr["region"], list):
                if "region_key" not in attr:
                    raise ValueError(
                        "`region` is of type `list` but `region_key` not found in `adata.uns['spatialdata_attr']`."
                    )
                if "instance_key" not in attr:
                    raise ValueError(
                        "`region` is of type `list` but `instance_key` not found in `adata.uns['spatialdata_attr']`."
                    )

            elif isinstance(attr["region"], str):
                attr["region_key"] = None
                attr["instance_key"] = None

            return data
