"""This file contains models and schema for SpatialData"""
from functools import singledispatchmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
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
from numpy.typing import ArrayLike, NDArray
from pandas.api.types import is_categorical_dtype
from pandera import SchemaModel
from pandera.typing.geopandas import GeoSeries
from scipy.sparse import csr_matrix
from shapely._geometry import GeometryType
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

C, Z, Y, X = "c", "z", "y", "x"

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
        data = to_spatial_image(array_like=data, dims=cls.dims.dims, **kwargs)
        # TODO(giovp): drop coordinates for now until solution with IO.
        if TYPE_CHECKING:
            assert isinstance(data, SpatialImage)
        data = data.drop(data.coords.keys())
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


class Label2DModel(RasterSchema):
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


class Labels3DModel(RasterSchema):
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


class Image2DModel(RasterSchema):
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


class Image3DModel(RasterSchema):
    dims = DimsSchema((C, Z, Y, X))
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
class PolygonModel(SchemaModel):
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
        # these correspond to GeometryType.POLYGON, GeometryType.MULTIPOINT
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


class ShapesModel:
    COORDS_KEY = "spatial"
    TRANSFORM_KEY = "transform"
    ATTRS_KEY = "spatialdata_attrs"

    def validate(self, data: AnnData) -> None:
        if self.COORDS_KEY not in data.obsm:
            raise ValueError(f"AnnData does not contain shapes coordinates in `adata.obsm['{self.COORDS_KEY}']`.")
        if self.TRANSFORM_KEY not in data.uns:
            raise ValueError(f"AnnData does not contain `{self.TRANSFORM_KEY}`.")
        if self.ATTRS_KEY not in data.uns:
            raise ValueError(f"AnnData does not contain `{self.ATTRS_KEY}`.")
        if "type" not in data.uns[self.ATTRS_KEY]:
            raise ValueError(f"AnnData does not contain `{self.ATTRS_KEY}['type']`.")
        if "size" not in data.uns[self.ATTRS_KEY]:
            raise ValueError(f"AnnData does not contain `{self.ATTRS_KEY}['size']`.")

    @classmethod
    def parse(
        cls,
        coords: np.ndarray,  # type: ignore[type-arg]
        shape_type: Literal["Circle", "Square"],
        shape_size: float,
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
        adata.obsm[cls.COORDS_KEY] = coords
        if transform is None:
            transform = Identity()
        adata.uns[cls.TRANSFORM_KEY] = transform
        adata.uns[cls.ATTRS_KEY] = {"type": shape_type, "size": shape_size}
        return adata


class PointsModel:
    COORDS_KEY = "spatial"
    TRANSFORM_KEY = "transform"

    def validate(self, data: AnnData) -> None:
        if self.COORDS_KEY not in data.obsm:
            raise ValueError(f"AnnData does not contain points coordinates in `adata.obsm['{self.COORDS_KEY}']`.")
        if self.TRANSFORM_KEY not in data.uns:
            raise ValueError(f"AnnData does not contain `{self.TRANSFORM_KEY}`.")

    @classmethod
    def parse(
        cls,
        coords: np.ndarray,  # type: ignore[type-arg]
        var_names: Sequence[str],
        points_assignment: np.ndarray,  # type: ignore[type-arg]
        transform: Optional[Any] = None,
        **kwargs: Any,
    ) -> AnnData:
        n_obs = coords.shape[0]
        sparse = _sparse_matrix_from_assignment(n_obs=n_obs, var_names=list(var_names), assignment=points_assignment)
        adata = AnnData(
            sparse,
            obs=pd.DataFrame(index=np.arange(n_obs)),
            var=pd.DataFrame(index=var_names),
            **kwargs,
        )
        adata.obsm[cls.COORDS_KEY] = coords
        if transform is None:
            transform = Identity()
        adata.uns[cls.TRANSFORM_KEY] = transform
        return adata


class TableModel:
    ATTRS_KEY = "spatialdata_attrs"

    def validate(
        self,
        data: AnnData,
    ) -> AnnData:
        if self.ATTRS_KEY in data.uns:
            attr = data.uns[self.ATTRS_KEY]
            if "region" not in attr:
                raise ValueError("`region` not found in `adata.uns['spatialdata_attr']`.")
            if isinstance(attr["region"], list):
                if "region_key" not in attr:
                    raise ValueError(
                        "`region` is of type `list` but `region_key` not found in `adata.uns['spatialdata_attr']`."
                    )
                if "instance_key" not in attr:
                    raise ValueError("`instance_key` not found in `adata.uns['spatialdata_attr']`.")
            elif isinstance(attr["region"], str):
                assert attr["region_key"] is None
                if "instance_key" not in attr:
                    raise ValueError("`instance_key` not found in `adata.uns['spatialdata_attr']`.")
        return data

    @classmethod
    def parse(
        cls,
        data: AnnData,
        region: Optional[Union[str, List[str]]] = None,
        region_key: Optional[str] = None,
        instance_key: Optional[str] = None,
    ) -> AnnData:
        # region, region_key and instance_key should either all live in adata.uns or all be passed in as argument
        n_args = sum([region is not None, region_key is not None, instance_key is not None])
        if n_args > 0:
            if cls.ATTRS_KEY in data.uns:
                raise ValueError(
                    f"Either pass `region`, `region_key` and `instance_key` as arguments or have them in `adata.uns['{cls.ATTRS_KEY}']`."
                )
        elif cls.ATTRS_KEY in data.uns:
            attr = data.uns[cls.ATTRS_KEY]
            region = attr["region"]
            region_key = attr["region_key"]
            instance_key = attr["instance_key"]

        if isinstance(region, str):
            if region_key is not None:
                raise ValueError("If `region` is of type `str`, `region_key` must be `None` as it is redundant.")
            if instance_key is None:
                raise ValueError("`instance_key` must be provided if `region` is of type `str`.")
        elif isinstance(region, list):
            if region_key is None:
                raise ValueError("`region_key` must be provided if `region` is of type `List`.")
            if not data.obs[region_key].isin(region).all():
                raise ValueError(f"`Region key: {region_key}` values do not match with `region` values.")
            if not is_categorical_dtype(data.obs[region_key]):
                logger.warning(f"Converting `region_key: {region_key}` to categorical dtype.")
                data.obs[region_key] = pd.Categorical(data.obs[region_key])
            if instance_key is None:
                raise ValueError("`instance_key` must be provided if `region` is of type `List`.")
        else:
            if region is not None:
                raise ValueError("`region` must be of type `str` or `List`.")
        # TODO: check for `instance_key` values?
        attr = {"region": region, "region_key": region_key, "instance_key": instance_key}
        data.uns[cls.ATTRS_KEY] = attr
        return data


def _sparse_matrix_from_assignment(
    n_obs: int, var_names: Union[List[str], ArrayLike], assignment: np.ndarray  # type: ignore[type-arg]
) -> csr_matrix:
    """Create a sparse matrix from an assignment array."""
    data: NDArray[np.bool_] = np.ones(len(assignment), dtype=bool)
    row = np.arange(len(assignment))
    if type(var_names) == np.ndarray:
        assert len(var_names.shape) == 1
        col = np.array([np.where(var_names == p)[0][0] for p in assignment])
    elif type(var_names) == list:
        col = np.array([var_names.index(p) for p in assignment])
    else:
        raise TypeError(f"var_names must be either np.array or List, but got {type(var_names)}")
    sparse = csr_matrix((data, (row, col)), shape=(n_obs, len(var_names)))
    return sparse
