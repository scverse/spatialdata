"""This file contains models and schema for SpatialData"""
import time
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
from geopandas import GeoDataFrame, GeoSeries
from multiscale_spatial_image import to_multiscale
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from multiscale_spatial_image.to_multiscale.to_multiscale import Methods
from numpy.typing import ArrayLike, NDArray
from pandas.api.types import is_categorical_dtype
from scipy.sparse import csr_matrix
from shapely._geometry import GeometryType
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.collection import GeometryCollection
from shapely.io import from_geojson, from_ragged_array
from spatial_image import SpatialImage, to_spatial_image
from tqdm import tqdm
from xarray import DataArray
from xarray_schema.components import (
    ArrayTypeSchema,
    AttrSchema,
    AttrsSchema,
    DimsSchema,
)
from xarray_schema.dataarray import DataArraySchema

from spatialdata._core.core_utils import (
    TRANSFORM_KEY,
    C,
    SpatialElement,
    X,
    Y,
    Z,
    get_default_coordinate_system,
    get_dims,
    set_transform,
)
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

Transform_s = AttrSchema(BaseTransformation, None)


def _parse_transform(element: SpatialElement, transform: Optional[BaseTransformation]) -> None:
    t: BaseTransformation
    if transform is None:
        t = Identity()
    else:
        t = transform
    if t.output_coordinate_system is None:
        dims = get_dims(element)
        t.output_coordinate_system = get_default_coordinate_system(dims)
    set_transform(element, t)


class RasterSchema(DataArraySchema):
    """Base schema for raster data."""

    @classmethod
    def parse(
        cls,
        data: ArrayLike,
        dims: Optional[Sequence[str]] = None,
        transform: Optional[BaseTransformation] = None,
        multiscale_factors: Optional[ScaleFactors_t] = None,
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
        multiscale_factors
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
        # if dims is specified inside the data, get the value of dims from the data
        if isinstance(data, DataArray) or isinstance(data, SpatialImage):
            if dims is not None:
                if dims != data.dims:
                    raise ValueError(
                        f"dims {dims} does not match data.dims {data.dims}, please specify the dims only once."
                    )
                else:
                    logger.warning(
                        "dims is specified redundantly: found also inside the data and the two values " "coincide."
                    )
            dims = data.dims

        # check if dims is spcified and if it has correct values
        if dims is None:
            dims = cls.dims.dims
            logger.info("dims is not specified, using default value: %s", dims)
        if len(set(dims).symmetric_difference(cls.dims.dims)) > 0:
            raise ValueError(f"Wrong dimensions: {dims}. Expected {cls.dims.dims} or a permutation of them.")

        # transpose the data if needed
        if isinstance(data, DataArray) or isinstance(data, SpatialImage):
            if data.dims != cls.dims.dims:
                data = data.transpose(*cls.dims.dims)
                logger.info(f"Transposing DataArray/SpatialImage data to {cls.dims.dims}.")
        elif isinstance(data, np.ndarray):
            if dims is None:
                raise ValueError("If data is a numpy array, dims must be provided.")
            if dims != cls.dims.dims:
                data = np.transpose(data, axes=[dims.index(d) for d in cls.dims.dims])
                logger.info(f"Transposing np.ndarray data to {cls.dims.dims}.")
        elif isinstance(data, DaskArray):
            if dims is None:
                raise ValueError("If data is a dask array, dims must be provided.")
            if dims != cls.dims.dims:
                data = data.transpose(*[dims.index(d) for d in cls.dims.dims])
                logger.info(f"Transposing DaskArray data to {cls.dims.dims}.")
        else:
            raise ValueError(f"Unsupported data type: {type(data)}.")

        # convert the data to a dask array
        if not isinstance(data, DaskArray):
            data = from_array(data)

        data = to_spatial_image(array_like=data, dims=cls.dims.dims, **kwargs)
        # TODO(giovp): drop coordinates for now until solution with IO.
        if TYPE_CHECKING:
            assert isinstance(data, SpatialImage)
        data = data.drop(data.coords.keys())
        if TYPE_CHECKING:
            assert isinstance(data, SpatialImage) or isinstance(data, MultiscaleSpatialImage)
        _parse_transform(data, transform)
        if multiscale_factors is not None:
            data = to_multiscale(
                data,
                scale_factors=multiscale_factors,
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


class Labels2DModel(RasterSchema):
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


# TODO: should check for columns be strict?
# TODO: validate attrs for transform.
class PolygonsModel:
    GEOMETRY_KEY = "geometry"
    ATTRS_KEY = "spatialdata_attrs"
    GEOS_KEY = "geos"
    TYPE_KEY = "type"
    NAME_KEY = "name"
    TRANSFORM_KEY = "transform"

    @classmethod
    def validate(cls, data: GeoDataFrame) -> None:
        if cls.GEOMETRY_KEY not in data:
            raise KeyError(f"GeoDataFrame must have a column named `{cls.GEOMETRY_KEY}`.")
        if not isinstance(data[cls.GEOMETRY_KEY], GeoSeries):
            raise ValueError(f"Column `{cls.GEOMETRY_KEY}` must be a GeoSeries.")
        if not isinstance(data[cls.GEOMETRY_KEY].values[0], Polygon) and not isinstance(
            data[cls.GEOMETRY_KEY].values[0], MultiPolygon
        ):
            # TODO: should check for all values?
            raise ValueError(f"Column `{cls.GEOMETRY_KEY}` must contain Polygon or MultiPolygon objects.")
        if cls.TRANSFORM_KEY not in data.attrs:
            raise ValueError(f":class:`geopandas.GeoDataFrame` does not contain `{TRANSFORM_KEY}`.")

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
        geometry: Literal[3, 6],  # [GeometryType.POLYGON, GeometryType.MULTIPOLYGON]
        transform: Optional[Any] = None,
        **kwargs: Any,
    ) -> GeoDataFrame:

        geometry = GeometryType(geometry)
        data = from_ragged_array(geometry, data, offsets)
        geo_df = GeoDataFrame({"geometry": data})
        _parse_transform(data, transform)
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

        gc: GeometryCollection = from_geojson(data)
        geo_df = GeoDataFrame({"geometry": gc.geoms})
        _parse_transform(data, transform)
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

        _parse_transform(data, transform)
        cls.validate(data)
        return data


class ShapesModel:
    COORDS_KEY = "spatial"
    ATTRS_KEY = "spatialdata_attrs"
    TYPE_KEY = "type"
    SIZE_KEY = "size"
    TRANSFORM_KEY = "transform"

    @classmethod
    def validate(cls, data: AnnData) -> None:
        if cls.COORDS_KEY not in data.obsm:
            raise ValueError(f":attr:`anndata.AnnData.obsm` does not contain shapes coordinates `{cls.COORDS_KEY}`.")
        if cls.TRANSFORM_KEY not in data.uns:
            raise ValueError(f":attr:`anndata.AnnData.uns` does not contain `{cls.TRANSFORM_KEY}`.")
        if cls.ATTRS_KEY not in data.uns:
            raise ValueError(f":attr:`anndata.AnnData.uns` does not contain `{cls.ATTRS_KEY}`.")
        if cls.TYPE_KEY not in data.uns[cls.ATTRS_KEY]:
            raise ValueError(f":attr:`anndata.AnnData.uns[`{cls.ATTRS_KEY}`]` does not contain `{cls.TYPE_KEY}`.")
        if cls.SIZE_KEY not in data.obs:
            raise ValueError(f":attr:`anndata.AnnData.obs` does not contain `{cls.SIZE_KEY}`.")

    @classmethod
    def parse(
        cls,
        coords: np.ndarray,  # type: ignore[type-arg]
        shape_type: Literal["Circle", "Square"],
        shape_size: Union[float, Sequence[float]],
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
        if isinstance(shape_size, list):
            if len(shape_size) != len(coords):
                raise ValueError("Length of `shape_size` must match length of `coords`.")
        shape_size_ = np.repeat(shape_size, len(coords)) if isinstance(shape_size, float) else shape_size
        adata = AnnData(
            None,
            obs=pd.DataFrame({cls.SIZE_KEY: shape_size_}, index=map(str, np.arange(coords.shape[0]))),
            **kwargs,
        )
        adata.obsm[cls.COORDS_KEY] = coords

        _parse_transform(adata, transform)
        adata.uns[cls.ATTRS_KEY] = {cls.TYPE_KEY: shape_type}
        return adata


class PointsModel:
    COORDS_KEY = "spatial"

    def validate(self, data: AnnData) -> None:
        if self.COORDS_KEY not in data.obsm:
            raise ValueError(f"AnnData does not contain points coordinates in `adata.obsm['{self.COORDS_KEY}']`.")
        if TRANSFORM_KEY not in data.uns:
            raise ValueError(f"AnnData does not contain `{TRANSFORM_KEY}`.")

    @classmethod
    def parse(
        cls,
        coords: np.ndarray,  # type: ignore[type-arg]
        points_assignment: Optional[np.ndarray] = None,  # type: ignore[type-arg]
        transform: Optional[Any] = None,
        **kwargs: Any,
    ) -> AnnData:
        n_obs = coords.shape[0]
        var_index: List[str]
        if points_assignment is not None:
            if not is_categorical_dtype(points_assignment):
                logger.warning(f"Converting `points_assignment: {points_assignment}` to categorical dtype.")
                points_assignment = pd.Categorical(points_assignment)
            var_names = points_assignment.cat.categories.tolist()
            var_index = var_names
            sparse = _sparse_matrix_from_assignment(n_obs=n_obs, var_names=var_names, assignment=points_assignment)
            kwargs["X"] = sparse
            kwargs["dtype"] = sparse.dtype
        else:
            kwargs["shape"] = (n_obs, 0)
            var_index = []
        start = time.time()
        adata = AnnData(
            obs=pd.DataFrame(index=np.arange(n_obs)),
            var=pd.DataFrame(index=var_index),
            **kwargs,
        )
        print(f"creating anndata: {time.time() - start}")
        adata.obsm[cls.COORDS_KEY] = coords
        _parse_transform(adata, transform)
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
        adata: AnnData,
        region: Optional[Union[str, List[str]]] = None,
        region_key: Optional[str] = None,
        instance_key: Optional[str] = None,
        region_values: Optional[Union[str, Sequence[str]]] = None,
        instance_values: Optional[Sequence[Any]] = None,
    ) -> AnnData:
        # region, region_key and instance_key should either all live in adata.uns or all be passed in as argument
        n_args = sum([region is not None, region_key is not None, instance_key is not None])
        if n_args > 0:
            if cls.ATTRS_KEY in adata.uns:
                raise ValueError(
                    f"Either pass `region`, `region_key` and `instance_key` as arguments or have them in `adata.uns['{cls.ATTRS_KEY}']`."
                )
        elif cls.ATTRS_KEY in adata.uns:
            attr = adata.uns[cls.ATTRS_KEY]
            region = attr["region"]
            region_key = attr["region_key"]
            instance_key = attr["instance_key"]

        if isinstance(region, str):
            if region_key is not None:
                raise ValueError("If `region` is of type `str`, `region_key` must be `None` as it is redundant.")
            if region_values is not None:
                raise ValueError("If `region` is of type `str`, `region_values` must be `None` as it is redundant.")
            if instance_key is None:
                raise ValueError("`instance_key` must be provided if `region` is of type `str`.")
        elif isinstance(region, list):
            if region_key is None:
                raise ValueError("`region_key` must be provided if `region` is of type `List`.")
            if not adata.obs[region_key].isin(region).all():
                raise ValueError(f"`Region key: {region_key}` values do not match with `region` values.")
            if not is_categorical_dtype(adata.obs[region_key]):
                logger.warning(f"Converting `region_key: {region_key}` to categorical dtype.")
                adata.obs[region_key] = pd.Categorical(adata.obs[region_key])
            if instance_key is None:
                raise ValueError("`instance_key` must be provided if `region` is of type `List`.")
        else:
            if region is not None:
                raise ValueError("`region` must be of type `str` or `List`.")
        # TODO: check for `instance_key` values?
        attr = {"region": region, "region_key": region_key, "instance_key": instance_key}
        adata.uns[cls.ATTRS_KEY] = attr

        if region_values is not None:
            if region_key in adata.obs:
                raise ValueError(f"this annotation table already contains the {region_key} (region_key) column")
            assert isinstance(region_values, str) or len(adata) == len(region_values)
            adata.obs[region_key] = region_values
        if instance_values is not None:
            if instance_key in adata.obs:
                raise ValueError(f"this annotation table already contains the {instance_key} (instance_key) column")
            assert len(adata) == len(instance_values)
            adata.obs[instance_key] = instance_values
        return adata


def _sparse_matrix_from_assignment(
    n_obs: int, var_names: Union[List[str], ArrayLike], assignment: np.ndarray  # type: ignore[type-arg]
) -> csr_matrix:
    """Create a sparse matrix from an assignment array."""
    data: NDArray[np.bool_] = np.ones(len(assignment), dtype=bool)
    row = np.arange(len(assignment))
    # if type(var_names) == np.ndarray:
    #     assert len(var_names.shape) == 1
    #     col = np.array([np.where(var_names == p)[0][0] for p in assignment])
    if type(var_names) == list:
        # naive way, slow
        # values = []
        # for p in tqdm(assignment, desc='creating sparse matrix'):
        #     values.append(var_names.index(p))
        # col = np.array(values)

        # better way, ~10 times faster
        col = np.full((len(assignment),), np.nan)
        for cat in tqdm(assignment.cat.categories, desc="creating sparse matrix"):
            value = var_names.index(cat)
            col[assignment == cat] = value
        assert np.sum(np.isnan(col)) == 0
    else:
        raise TypeError(f"var_names must be either np.array or List, but got {type(var_names)}")
    sparse = csr_matrix((data, (row, col)), shape=(n_obs, len(var_names)))
    return sparse
