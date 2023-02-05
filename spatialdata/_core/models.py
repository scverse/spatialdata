"""This file contains models and schema for SpatialData"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import singledispatchmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa
from anndata import AnnData
from dask.array.core import Array as DaskArray
from dask.array.core import from_array
from geopandas import GeoDataFrame, GeoSeries
from multiscale_spatial_image import to_multiscale
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from multiscale_spatial_image.to_multiscale.to_multiscale import Methods
from numpy.typing import NDArray
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
    DEFAULT_COORDINATE_SYSTEM,
    TRANSFORM_KEY,
    C,
    MappingToCoordinateSystem_t,
    SpatialElement,
    X,
    Y,
    Z,
    _get_transformations,
    _set_transformations,
    _validate_mapping_to_coordinate_system_type,
    get_dims,
)
from spatialdata._core.transformations import BaseTransformation, Identity
from spatialdata._logging import logger
from spatialdata._types import ArrayLike

# Types
Chunks_t = Union[
    int,
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
    Mapping[Any, Union[None, int, tuple[int, ...]]],
]
ScaleFactors_t = Sequence[Union[dict[str, int], int]]

Transform_s = AttrSchema(BaseTransformation, None)


__all__ = [
    "Labels2DModel",
    "Labels3DModel",
    "Image2DModel",
    "Image3DModel",
    "PolygonsModel",
    "ShapesModel",
    "PointsModel",
    "TableModel",
    "get_schema",
]


def _parse_transformations(
    element: SpatialElement, transformations: Optional[MappingToCoordinateSystem_t] = None
) -> None:
    _validate_mapping_to_coordinate_system_type(transformations)
    transformations_in_element = _get_transformations(element)
    if (
        transformations_in_element is not None
        and len(transformations_in_element) > 0
        and transformations is not None
        and len(transformations) > 0
    ):
        raise ValueError(
            "Transformations are both specified for the element and also passed as an argument to the parser. Please "
            "specify the transformations only once."
        )
    elif transformations_in_element is not None and len(transformations_in_element) > 0:
        parsed_transformations = transformations_in_element
    elif transformations is not None and len(transformations) > 0:
        parsed_transformations = transformations
    else:
        parsed_transformations = {DEFAULT_COORDINATE_SYSTEM: Identity()}
    _set_transformations(element, parsed_transformations)


class RasterSchema(DataArraySchema):
    """Base schema for raster data."""

    @classmethod
    def parse(
        cls,
        data: Union[ArrayLike, DataArray, DaskArray],
        dims: Optional[Sequence[str]] = None,
        transformations: Optional[MappingToCoordinateSystem_t] = None,
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
        dims
            Dimensions of the data.
        transformations
            Transformations to apply to the data.
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
        # check if dims is specified and if it has correct values

        # if dims is specified inside the data, get the value of dims from the data
        if isinstance(data, DataArray) or isinstance(data, SpatialImage):
            if not isinstance(data.data, DaskArray):  # numpy -> dask
                data.data = from_array(data.data)
            if dims is not None:
                if dims != data.dims:
                    raise ValueError(
                        f"`dims`: {dims} does not match `data.dims`: {data.dims}, please specify the dims only once."
                    )
                else:
                    logger.info("`dims` is specified redundantly: found also inside `data`")
            else:
                dims = data.dims  # type: ignore[assignment]
            _reindex = lambda d: d
        elif isinstance(data, np.ndarray) or isinstance(data, DaskArray):
            if not isinstance(data, DaskArray):  # numpy -> dask
                data = from_array(data)
            if dims is None:
                dims = cls.dims.dims
                logger.info(f"no axes information specified in the object, setting `dims` to: {dims}")
            else:
                if len(set(dims).symmetric_difference(cls.dims.dims)) > 0:
                    raise ValueError(f"Wrong `dims`: {dims}. Expected {cls.dims.dims}.")
            _reindex = lambda d: dims.index(d)  # type: ignore[union-attr]
        else:
            raise ValueError(f"Unsupported data type: {type(data)}.")

        # transpose if possible
        if dims != cls.dims.dims:
            try:
                assert isinstance(data, DaskArray) or isinstance(data, DataArray)
                # mypy complains that data has no .transpose but I have asserted right above that data is a DaskArray...
                data = data.transpose(*[_reindex(d) for d in cls.dims.dims])  # type: ignore[attr-defined]
                logger.info(f"Transposing `data` of type: {type(data)} to {cls.dims.dims}.")
            except ValueError:
                raise ValueError(f"Cannot transpose arrays to match `dims`: {dims}. Try to reshape `data` or `dims`.")

        data = to_spatial_image(array_like=data, dims=cls.dims.dims, **kwargs)
        assert isinstance(data, SpatialImage)
        _parse_transformations(data, transformations)
        if multiscale_factors is not None:
            # check that the image pyramid doesn't contain axes that get collapsed and eventually truncates the list
            # of downscaling factors to avoid this
            adjusted_multiscale_factors: list[int] = []
            assert isinstance(data, DataArray)
            current_shape: ArrayLike = np.array(data.shape, dtype=float)
            # multiscale_factors could be a dict, we don't support this case here (in the future this code and the
            # more general case will be handled by multiscale-spatial-image)
            assert isinstance(multiscale_factors, list)
            for factor in multiscale_factors:
                scale_vector = np.array([1.0 if ax == "c" else factor for ax in data.dims])
                current_shape /= scale_vector
                if current_shape.min() < 1:
                    logger.warning(
                        f"Detected a multiscale factor that would collapse an axis: truncating list of factors from {multiscale_factors} to {adjusted_multiscale_factors}"
                    )
                    break
                adjusted_multiscale_factors.append(factor)
            parsed_transform = _get_transformations(data)
            del data.attrs["transform"]
            data = to_multiscale(
                data,
                scale_factors=adjusted_multiscale_factors,
                method=method,
                chunks=chunks,
            )
            _parse_transformations(data, parsed_transform)
            assert isinstance(data, MultiscaleSpatialImage)
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
            # suppressing the check of .attrs['transform']; see https://github.com/scverse/spatialdata/issues/115
            # attrs=self.attrs,
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
            # suppressing the check of .attrs['transform']; see https://github.com/scverse/spatialdata/issues/115
            # attrs=self.attrs,
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
            # suppressing the check of .attrs['transform']; see https://github.com/scverse/spatialdata/issues/115
            # attrs=self.attrs,
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
            # suppressing the check of .attrs['transform']; see https://github.com/scverse/spatialdata/issues/115
            # attrs=self.attrs,
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
        raise NotImplementedError()

    @parse.register(np.ndarray)
    @classmethod
    def _(
        cls,
        data: np.ndarray,  # type: ignore[type-arg]
        offsets: tuple[np.ndarray, ...],  # type: ignore[type-arg]
        geometry: Literal[3, 6],  # [GeometryType.POLYGON, GeometryType.MULTIPOLYGON]
        transformations: Optional[MappingToCoordinateSystem_t] = None,
        **kwargs: Any,
    ) -> GeoDataFrame:

        geometry = GeometryType(geometry)
        data = from_ragged_array(geometry, data, offsets)
        geo_df = GeoDataFrame({"geometry": data})
        _parse_transformations(geo_df, transformations)
        cls.validate(geo_df)
        return geo_df

    @parse.register(str)
    @parse.register(Path)
    @classmethod
    def _(
        cls,
        data: str,
        transformations: Optional[Any] = None,
        **kwargs: Any,
    ) -> GeoDataFrame:
        data = Path(data) if isinstance(data, str) else data  # type: ignore[assignment]
        if TYPE_CHECKING:
            assert isinstance(data, Path)

        gc: GeometryCollection = from_geojson(data.read_bytes())
        if not isinstance(gc, GeometryCollection):
            raise ValueError(f"`{data}` does not contain a `GeometryCollection`.")
        geo_df = GeoDataFrame({"geometry": gc.geoms})
        _parse_transformations(geo_df, transformations)
        cls.validate(geo_df)
        return geo_df

    @parse.register(GeoDataFrame)
    @classmethod
    def _(
        cls,
        data: GeoDataFrame,
        transformations: Optional[MappingToCoordinateSystem_t] = None,
        **kwargs: Any,
    ) -> GeoDataFrame:

        _parse_transformations(data, transformations)
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
        index: Optional[Sequence[Union[str, float]]] = None,
        transformations: Optional[MappingToCoordinateSystem_t] = None,
        **kwargs: Any,
    ) -> AnnData:
        """
        Parse shape data into SpatialData.

        Parameters
        ----------
        coords
            Coordinates of shapes.
        ids
            Unique ids of shapes.
        shape_type
            Type of shape.
        shape_size
            Size of shape.
        index
            Index names of the shapes.
        transformations
            Transformations for the shape element.
        kwargs
            Additional arguments for shapes.

        Returns
        -------
        :class:`anndata.AnnData` formatted for shapes elements.
        """
        if shape_type is None:
            shape_type = "Circle"
        assert shape_type in ["Circle", "Square"]
        if isinstance(shape_size, list):
            if len(shape_size) != len(coords):
                raise ValueError("Length of `shape_size` must match length of `coords`.")
        shape_size_ = np.repeat(shape_size, len(coords)) if isinstance(shape_size, float) else shape_size
        if index is None:
            index_ = map(str, np.arange(coords.shape[0]))
            logger.info("No index provided, using default index `np.arange(coords.shape[0])`.")
        else:
            index_ = index  # type: ignore[assignment]
        adata = AnnData(
            None,
            obs=pd.DataFrame({cls.SIZE_KEY: shape_size_}, index=index_),
            **kwargs,
        )
        adata.obsm[cls.COORDS_KEY] = coords

        _parse_transformations(adata, transformations)
        adata.uns[cls.ATTRS_KEY] = {cls.TYPE_KEY: shape_type}
        return adata


class PointsModel:
    ATTRS_KEY = "spatialdata_attrs"
    NAME_KEY = "name"
    TRANSFORM_KEY = "transform"

    @classmethod
    def validate(cls, data: pa.Table) -> None:
        for ax in [X, Y, Z]:
            if ax in data.column_names:
                assert data.schema.field(ax).type in [pa.float32(), pa.float64(), pa.int64()]
        try:
            assert data.schema.metadata is not None
            _ = _get_transformations(data)
        except Exception as e:  # noqa: B902
            logger.error("cannot parse the transformation from the pyarrow.Table object")
            raise e

    @classmethod
    def parse(
        cls,
        coords: np.ndarray,  # type: ignore[type-arg]
        annotations: Optional[pa.Table] = None,
        transformations: Optional[MappingToCoordinateSystem_t] = None,
        supress_categorical_check: bool = False,
        **kwargs: Any,
    ) -> pa.Table:
        if annotations is not None:
            if not supress_categorical_check:
                assert isinstance(annotations, pa.Table)
                # TODO: restore this warning and fix, or TODO(giovp): consider casting to categorical
                # for column in annotations.column_names:
                #     c = annotations[column]
                #     if not isinstance(c.dictionary_encode().type, pa.DictionaryType):
                #         logger.warning(
                #             "detected columns that are not categorical, consider converting it to categorical with "
                #             ".dictionary_encode() to achieve faster performance. Call parse() with "
                #             "suppress_categorical_check=True to hide this warning"
                #         )
        assert len(coords.shape) == 2
        ndim = coords.shape[1]
        axes = [X, Y, Z][:ndim]
        table = pa.Table.from_arrays([pa.array(coords[:, i].flatten()) for i in range(coords.shape[1])], names=axes)
        if annotations is not None:
            for column in annotations.column_names:
                if column in [X, Y, Z]:
                    raise ValueError(f"Column name {column} is reserved for coordinates.")
                table = table.append_column(column, annotations[column])
        _parse_transformations(table, transformations)
        cls.validate(table)
        return table
        # new_table = _parse_transformations(table, transform)
        # cls.validate(new_table)
        # return new_table


class TableModel:
    ATTRS_KEY = "spatialdata_attrs"
    REGION_KEY = "region"
    REGION_KEY_KEY = "region_key"
    INSTANCE_KEY = "instance_key"

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
        region: Optional[Union[str, list[str]]] = None,
        region_key: Optional[str] = None,
        instance_key: Optional[str] = None,
    ) -> AnnData:
        # either all live in adata.uns or all be passed in as argument
        n_args = sum([region is not None, region_key is not None, instance_key is not None])
        if n_args > 0:
            if cls.ATTRS_KEY in adata.uns:
                raise ValueError(
                    f"Either pass `{cls.REGION_KEY}`, `{cls.REGION_KEY_KEY}` and `{cls.INSTANCE_KEY}` as arguments or have them in `adata.uns[{cls.ATTRS_KEY!r}]`."
                )
        elif cls.ATTRS_KEY in adata.uns:
            attr = adata.uns[cls.ATTRS_KEY]
            region = attr[cls.REGION_KEY]
            region_key = attr[cls.REGION_KEY_KEY]
            instance_key = attr[cls.INSTANCE_KEY]

        if isinstance(region, str):
            if region_key is not None:
                raise ValueError(
                    f"If `{cls.REGION_KEY}` is of type `str`, `{cls.REGION_KEY_KEY}` must be `None` as it is redundant."
                )
            if instance_key is None:
                raise ValueError("`instance_key` must be provided if `region` is of type `List`.")
        elif isinstance(region, list):
            if region_key is None:
                raise ValueError(f"`{cls.REGION_KEY_KEY}` must be provided if `{cls.REGION_KEY}` is of type `List`.")
            if not adata.obs[region_key].isin(region).all():
                raise ValueError(f"`adata.obs[{region_key}]` values do not match with `{cls.REGION_KEY}` values.")
            if not is_categorical_dtype(adata.obs[region_key]):
                logger.warning(f"Converting `{cls.REGION_KEY_KEY}: {region_key}` to categorical dtype.")
                adata.obs[region_key] = pd.Categorical(adata.obs[region_key])
            if instance_key is None:
                raise ValueError("`instance_key` must be provided if `region` is of type `List`.")
        else:
            if region is not None:
                raise ValueError(f"`{cls.REGION_KEY}` must be of type `str` or `List`.")

        # TODO: check for `instance_key` values?
        attr = {"region": region, "region_key": region_key, "instance_key": instance_key}
        adata.uns[cls.ATTRS_KEY] = attr
        return adata


# TODO: consider removing if we settle with geodataframe
def _sparse_matrix_from_assignment(
    n_obs: int, var_names: Union[list[str], ArrayLike], assignment: pd.Series
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
        # tqdm not needed as a dependency if this function is removed
        for cat in tqdm(assignment.cat.categories, desc="creating sparse matrix"):
            value = var_names.index(cat)
            col[assignment == cat] = value
        assert np.sum(np.isnan(col)) == 0
    else:
        raise TypeError(f"var_names must be either np.array or List, but got {type(var_names)}")
    sparse = csr_matrix((data, (row, col)), shape=(n_obs, len(var_names)))
    return sparse


Schema_t = Union[
    type[Image2DModel],
    type[Image3DModel],
    type[Labels2DModel],
    type[Labels3DModel],
    type[PointsModel],
    type[PolygonsModel],
    type[ShapesModel],
    type[TableModel],
]


def get_schema(
    e: SpatialElement,
) -> Schema_t:
    def _validate_and_return(
        schema: Schema_t,
        e: Union[SpatialElement],
    ) -> Schema_t:
        schema().validate(e)
        return schema

    if isinstance(e, SpatialImage) or isinstance(e, MultiscaleSpatialImage):
        axes = get_dims(e)
        if "c" in axes:
            if "z" in axes:
                return _validate_and_return(Image3DModel, e)
            else:
                return _validate_and_return(Image2DModel, e)
        else:
            if "z" in axes:
                return _validate_and_return(Labels3DModel, e)
            else:
                return _validate_and_return(Labels2DModel, e)
    elif isinstance(e, GeoDataFrame):
        return _validate_and_return(PolygonsModel, e)
    elif isinstance(e, pa.Table):
        return _validate_and_return(PointsModel, e)
    elif isinstance(e, AnnData):
        if "spatial" in e.obsm:
            return _validate_and_return(ShapesModel, e)
        else:
            return _validate_and_return(TableModel, e)
    else:
        raise TypeError(f"Unsupported type {type(e)}")
