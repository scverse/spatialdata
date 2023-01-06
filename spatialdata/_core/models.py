"""This file contains models and schema for SpatialData"""
import copy
import json
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

from spatialdata._core.coordinate_system import CoordinateSystem
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
from spatialdata._core.transformations import (
    Affine,
    BaseTransformation,
    ByDimension,
    Identity,
    MapAxis,
)
from spatialdata._core.transformations import Sequence as SequenceTransformation
from spatialdata._logging import logger

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
]


def _parse_transform(element: SpatialElement, transform: Optional[BaseTransformation] = None) -> SpatialElement:
    # if input and output coordinate systems are not specified by the user, we try to infer them. If it's logically
    # not possible to infer them, an exception is raised.
    t: BaseTransformation
    if transform is None:
        t = Identity()
    else:
        t = transform
    if t.input_coordinate_system is None:
        dims = get_dims(element)
        t.input_coordinate_system = get_default_coordinate_system(dims)
    if t.output_coordinate_system is None:
        t.output_coordinate_system = SequenceTransformation._inferring_cs_infer_output_coordinate_system(t)

    # this function is to comply with mypy since we could call .axes_names on the wrong type
    def _get_axes_names(cs: Optional[Union[str, CoordinateSystem]]) -> tuple[str, ...]:
        assert isinstance(cs, CoordinateSystem)
        return cs.axes_names

    # determine if we are in the 2d case or 3d case and determine the coordinate system we want to map to (basically
    # we want both the spatial dimensions and c). If the output coordinate system of the transformation t is not
    # matching, compose the transformation with an appropriate transformation to map to the correct coordinate system
    if Z in _get_axes_names(t.output_coordinate_system):
        mapper_output_coordinate_system = get_default_coordinate_system((C, Z, Y, X))
    else:
        # if we are in the 3d case but the element does not contain the Z dimension, it's up to the user to specify
        # the correct coordinate transformation and output coordinate system
        mapper_output_coordinate_system = get_default_coordinate_system((C, Y, X))
    combined: BaseTransformation
    assert isinstance(t.output_coordinate_system, CoordinateSystem)
    assert isinstance(mapper_output_coordinate_system, CoordinateSystem)

    # patch to be removed when this function is refactored to address https://github.com/scverse/spatialdata/issues/39
    cs1 = copy.deepcopy(t.output_coordinate_system)
    cs2 = copy.deepcopy(mapper_output_coordinate_system)
    for ax1, ax2 in zip(cs1._axes, cs2._axes):
        ax1.unit = None
        ax2.unit = None

    if cs1._axes != cs2._axes:
        mapper_input_coordinate_system = t.output_coordinate_system
        assert C not in _get_axes_names(mapper_input_coordinate_system)
        any_axis_cs = get_default_coordinate_system((_get_axes_names(t.input_coordinate_system)[0],))
        c_cs = get_default_coordinate_system((C,))
        mapper = ByDimension(
            transformations=[
                MapAxis(
                    {ax: ax for ax in _get_axes_names(t.input_coordinate_system)},
                    input_coordinate_system=t.input_coordinate_system,
                    output_coordinate_system=t.input_coordinate_system,
                ),
                Affine(
                    np.array([[0, 0], [0, 1]]),
                    input_coordinate_system=any_axis_cs,
                    output_coordinate_system=c_cs,
                ),
            ],
            input_coordinate_system=mapper_input_coordinate_system,
            output_coordinate_system=mapper_output_coordinate_system,
        )
        combined = SequenceTransformation(
            [t, mapper],
            input_coordinate_system=t.input_coordinate_system,
            output_coordinate_system=mapper_output_coordinate_system,
        )
    else:
        combined = t
    # test that all il good by checking that we can compute an affine matrix from this
    try:
        _ = combined.to_affine().affine
    except Exception as e:  # noqa: B902
        # debug
        logger.debug("Error while trying to compute affine matrix from transformation: ")
        from pprint import pprint

        pprint(combined.to_dict())
        raise e

    # finalize
    new_element = set_transform(element, combined)
    return new_element


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
        dims
            Dimensions of the data.
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
                dims = data.dims
            _reindex = lambda d: d

        elif isinstance(data, np.ndarray) or isinstance(data, DaskArray):
            if not isinstance(data, DaskArray):  # numpy -> dask
                data = from_array(data)
            if dims is None:
                dims = cls.dims.dims
                logger.info(f"`dims` is set to: {dims}")
            else:
                if len(set(dims).symmetric_difference(cls.dims.dims)) > 0:
                    raise ValueError(f"Wrong `dims`: {dims}. Expected {cls.dims.dims}.")
            _reindex = lambda d: dims.index(d)  # type: ignore[union-attr]
        else:
            raise ValueError(f"Unsupported data type: {type(data)}.")

        # transpose if possible
        if dims != cls.dims.dims:
            try:
                data = data.transpose(*[_reindex(d) for d in cls.dims.dims])
                logger.info(f"Transposing `data` of type: {type(data)} to {cls.dims.dims}.")
            except ValueError:
                raise ValueError(f"Cannot transpose arrays to match `dims`: {dims}. Try to reshape `data` or `dims`.")

        data = to_spatial_image(array_like=data, dims=cls.dims.dims, **kwargs)
        if TYPE_CHECKING:
            assert isinstance(data, SpatialImage)
        # TODO(giovp): drop coordinates for now until solution with IO.
        data = data.drop(data.coords.keys())
        _parse_transform(data, transform)
        if multiscale_factors is not None:
            data = to_multiscale(
                data,
                scale_factors=multiscale_factors,
                method=method,
                chunks=chunks,
            )
            _parse_transform(data, transform)
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
        raise NotImplementedError()

    @parse.register(np.ndarray)
    @classmethod
    def _(
        cls,
        data: np.ndarray,  # type: ignore[type-arg]
        offsets: tuple[np.ndarray, ...],  # type: ignore[type-arg]
        geometry: Literal[3, 6],  # [GeometryType.POLYGON, GeometryType.MULTIPOLYGON]
        transform: Optional[Any] = None,
        **kwargs: Any,
    ) -> GeoDataFrame:

        geometry = GeometryType(geometry)
        data = from_ragged_array(geometry, data, offsets)
        geo_df = GeoDataFrame({"geometry": data})
        _parse_transform(geo_df, transform)
        cls.validate(geo_df)
        return geo_df

    @parse.register(str)
    @parse.register(Path)
    @classmethod
    def _(
        cls,
        data: str,
        transform: Optional[Any] = None,
        **kwargs: Any,
    ) -> GeoDataFrame:
        data = Path(data) if isinstance(data, str) else data  # type: ignore[assignment]
        if TYPE_CHECKING:
            assert isinstance(data, Path)

        gc: GeometryCollection = from_geojson(data.read_bytes())
        if not isinstance(gc, GeometryCollection):
            raise ValueError(f"`{data}` does not contain a `GeometryCollection`.")
        geo_df = GeoDataFrame({"geometry": gc.geoms})
        _parse_transform(geo_df, transform)
        cls.validate(geo_df)
        return geo_df

    @parse.register(GeoDataFrame)
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
    GEOMETRY_KEY = "geometry"
    ATTRS_KEY = "spatialdata_attrs"
    GEOS_KEY = "geos"
    TYPE_KEY = "type"
    NAME_KEY = "name"
    TRANSFORM_KEY = "transform"

    @classmethod
    def validate(cls, data: pa.Table) -> None:
        for ax in [X, Y, Z]:
            if ax in data.column_names:
                assert data.schema.field(ax).type in [pa.float32(), pa.float64(), pa.int64()]
        try:
            assert data.schema.metadata is not None
            t_bytes = data.schema.metadata[TRANSFORM_KEY.encode("utf-8")]
            BaseTransformation.from_dict(json.loads(t_bytes.decode("utf-8")))
        except Exception as e:  # noqa: B902
            logger.error("cannot parse the transformation from the pyarrow.Table object")
            raise e

    @classmethod
    def parse(
        cls,
        coords: np.ndarray,  # type: ignore[type-arg]
        annotations: Optional[pa.Table] = None,
        transform: Optional[Any] = None,
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
        new_table = _parse_transform(table, transform)
        cls.validate(new_table)
        return new_table


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
        region_values: Optional[Union[str, Sequence[str]]] = None,
        instance_values: Optional[Sequence[Any]] = None,
    ) -> AnnData:
        # either all live in adata.uns or all be passed in as argument
        n_args = sum([region is not None, region_key is not None, instance_key is not None])
        if n_args > 0:
            if cls.ATTRS_KEY in adata.uns:
                raise ValueError(
                    f"Either pass `{cls.REGION_KEY}`, `{cls.REGION_KEY_KEY}` and `{cls.INSTANCE_KEY}` as arguments or have them in `adata.uns['{cls.ATTRS_KEY}']`."
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
            if region_values is not None:
                raise ValueError(
                    f"If `{cls.REGION_KEY}` is of type `str`, `region_values` must be `None` as it is redundant."
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

        # TODO(giovp): do we really need to support this?
        if region_values is not None:
            if region_key in adata.obs:
                raise ValueError(
                    f"this annotation table already contains the {region_key} ({cls.REGION_KEY_KEY}) column"
                )
            assert isinstance(region_values, str) or len(adata) == len(region_values)
            adata.obs[region_key] = region_values
        if instance_values is not None:
            if instance_key in adata.obs:
                raise ValueError(
                    f"this annotation table already contains the {instance_key} ({cls.INSTANCE_KEY}) column"
                )
            assert len(adata) == len(instance_values)
            adata.obs[instance_key] = instance_values
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
