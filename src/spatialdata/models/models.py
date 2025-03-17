"""Models and schema for SpatialData."""

import warnings
from collections.abc import Mapping, Sequence
from functools import singledispatchmethod
from pathlib import Path
from typing import Any, Literal, TypeAlias

import dask.dataframe as dd
import numpy as np
import pandas as pd
from anndata import AnnData
from dask.array import Array as DaskArray
from dask.array.core import from_array
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame, GeoSeries
from multiscale_spatial_image import to_multiscale
from multiscale_spatial_image.to_multiscale.to_multiscale import Methods
from pandas import CategoricalDtype
from shapely._geometry import GeometryType
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.collection import GeometryCollection
from shapely.io import from_geojson, from_ragged_array
from spatial_image import to_spatial_image
from xarray import DataArray, DataTree
from xarray_schema.components import (
    ArrayTypeSchema,
    AttrSchema,
    AttrsSchema,
    DimsSchema,
)
from xarray_schema.dataarray import DataArraySchema

from spatialdata._core.validation import validate_table_attr_keys
from spatialdata._logging import logger
from spatialdata._types import ArrayLike
from spatialdata._utils import _check_match_length_channels_c_dim
from spatialdata.config import LARGE_CHUNK_THRESHOLD_BYTES
from spatialdata.models import C, X, Y, Z, get_axes_names
from spatialdata.models._utils import (
    DEFAULT_COORDINATE_SYSTEM,
    TRANSFORM_KEY,
    MappingToCoordinateSystem_t,
    SpatialElement,
    _validate_mapping_to_coordinate_system_type,
    convert_region_column_to_categorical,
)
from spatialdata.transformations._utils import (
    _get_transformations,
    _set_transformations,
    compute_coordinates,
)
from spatialdata.transformations.transformations import BaseTransformation, Identity

# Types
Chunks_t: TypeAlias = int | tuple[int, ...] | tuple[tuple[int, ...], ...] | Mapping[Any, None | int | tuple[int, ...]]
ScaleFactors_t = Sequence[dict[str, int] | int]

Transform_s = AttrSchema(BaseTransformation, None)
ATTRS_KEY = "spatialdata_attrs"


def _parse_transformations(element: SpatialElement, transformations: MappingToCoordinateSystem_t | None = None) -> None:
    _validate_mapping_to_coordinate_system_type(transformations)
    transformations_in_element = _get_transformations(element)
    if (
        transformations_in_element is not None
        and len(transformations_in_element) > 0
        and transformations is not None
        and len(transformations) > 0
    ):
        # we can relax this and overwrite the transformations using the one passed as argument
        raise ValueError(
            "Transformations are both specified for the element and also passed as an argument to the parser. Please "
            "specify the transformations only once."
        )
    if transformations is not None and len(transformations) > 0:
        parsed_transformations = transformations
    elif transformations_in_element is not None and len(transformations_in_element) > 0:
        parsed_transformations = transformations_in_element
    else:
        parsed_transformations = {DEFAULT_COORDINATE_SYSTEM: Identity()}
    _set_transformations(element, parsed_transformations)


class RasterSchema(DataArraySchema):
    """Base schema for raster data."""

    # TODO add DataTree validation, validate has scale0... etc and each scale contains 1 image in .variables.
    ATTRS_KEY = ATTRS_KEY

    @classmethod
    def parse(
        cls,
        data: ArrayLike | DataArray | DaskArray,
        dims: Sequence[str] | None = None,
        c_coords: str | list[str] | None = None,
        transformations: MappingToCoordinateSystem_t | None = None,
        scale_factors: ScaleFactors_t | None = None,
        method: Methods | None = None,
        chunks: Chunks_t | None = None,
        **kwargs: Any,
    ) -> DataArray | DataTree:
        r"""
        Validate (or parse) raster data.

        Parameters
        ----------
        data
            Data to validate (or parse). The shape of the data should be c(z)yx for 2D (3D) images and (z)yx for 2D (
            3D) labels. If you have a 2D image with shape yx, you can use :func:`numpy.expand_dims` (or an equivalent
            function) to add a channel dimension.
        dims
            Dimensions of the data (e.g. ['c', 'y', 'x'] for 2D image data). If the data is a :class:`xarray.DataArray`,
            the dimensions can also be inferred from the data. If the dimensions are not in the order (c)(z)yx, the data
            will be transposed to match the order.
        c_coords : str | list[str] | None
            Channel names of image data. Must be equal to the length of dimension 'c'. Only supported for `Image`
            models.
        transformations
            Dictionary of transformations to apply to the data. The key is the name of the target coordinate system,
            the value is the transformation to apply. By default, a single `Identity` transformation mapping to the
            `"global"` coordinate system is applied.
        scale_factors
            Scale factors to apply to construct a multiscale image (:class:`datatree.DataTree`).
            If `None`, a :class:`xarray.DataArray` is returned instead.
            Importantly, each scale factor is relative to the previous scale factor. For example, if the scale factors
            are `[2, 2, 2]`, the returned multiscale image will have 4 scales. The original image and then the 2x, 4x
            and 8x downsampled images.
        method
            Method to use for multiscale downsampling (default is `'nearest'`). Please refer to
            :class:`multiscale_spatial_image.to_multiscale` for details.\n
            Note (advanced): the default choice (`'nearest'`) will keep the original scale lazy and compute each
            downscaled version. On the other hand `'xarray_coarsen'` will compute each scale lazily (this implies that
            each scale will be recomputed each time it is accessed unless `.persist()` is manually called to cache the
            intermediate results). Please refer direclty to the source code of `to_multiscale()` in the
            `multiscale-spatial-image` for precise information on how this is handled.
        chunks
            Chunks to use for dask array.
        kwargs
            Additional arguments for :func:`to_spatial_image`. In particular the `c_coords` kwargs argument (an
            iterable) can be used to set the channel coordinates for image data. `c_coords` is not available for labels
            data as labels do not have channels.

        Returns
        -------
        :class:`xarray.DataArray` or :class:`datatree.DataTree`

        Notes
        -----
        **RGB images**

        If you have an image with 3 or 4 channels and you want to interpret it as an RGB or RGB(A) image, you can use
        the `c_coords` argument to specify the channel coordinates as `["r", "g", "b"]` or `["r", "g", "b", "a"]`.

        You can also pass the `rgb` argument to `kwargs` to automatically set the `c_coords` to `["r", "g", "b"]`.
        Please refer to :func:`to_spatial_image` for more information. Note: if you set `rgb=None` in `kwargs`, 3-4
        channel images will be interpreted automatically as RGB(A) images.
        """
        if transformations:
            transformations = transformations.copy()
        if "name" in kwargs:
            raise ValueError("The `name` argument is not (yet) supported for raster data.")
        # if dims is specified inside the data, get the value of dims from the data
        if isinstance(data, DataArray):
            if not isinstance(data.data, DaskArray):  # numpy -> dask
                data.data = from_array(data.data)
            if dims is not None:
                if set(dims).symmetric_difference(data.dims):
                    raise ValueError(
                        f"`dims`: {dims} does not match `data.dims`: {data.dims}, please specify the dims only once."
                    )
                logger.info("`dims` is specified redundantly: found also inside `data`.")
            else:
                dims = data.dims
            # but if dims don't match the model's dims, throw error
            if set(dims).symmetric_difference(cls.dims.dims):
                raise ValueError(f"Wrong `dims`: {dims}. Expected {cls.dims.dims}.")
            _reindex = lambda d: d
        # if there are no dims in the data, use the model's dims or provided dims
        elif isinstance(data, np.ndarray | DaskArray):
            if not isinstance(data, DaskArray):  # numpy -> dask
                data = from_array(data)
            if dims is None:
                dims = cls.dims.dims
                logger.info(f"no axes information specified in the object, setting `dims` to: {dims}")
            else:
                if len(set(dims).symmetric_difference(cls.dims.dims)) > 0:
                    raise ValueError(f"Wrong `dims`: {dims}. Expected {cls.dims.dims}.")
            _reindex = lambda d: dims.index(d)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}.")

        # transpose if possible
        if tuple(dims) != cls.dims.dims:
            try:
                if isinstance(data, DataArray):
                    data = data.transpose(*list(cls.dims.dims))
                elif isinstance(data, DaskArray):
                    data = data.transpose(*[_reindex(d) for d in cls.dims.dims])
                else:
                    raise ValueError(f"Unsupported data type: {type(data)}.")
                logger.info(f"Transposing `data` of type: {type(data)} to {cls.dims.dims}.")
            except ValueError as e:
                raise ValueError(
                    f"Cannot transpose arrays to match `dims`: {dims}.",
                    "Try to reshape `data` or `dims`.",
                ) from e

        # finally convert to spatial image
        if c_coords is not None:
            c_coords = _check_match_length_channels_c_dim(data, c_coords, cls.dims.dims)

        if c_coords is not None and len(c_coords) != data.shape[cls.dims.dims.index("c")]:
            raise ValueError(
                f"The number of channel names `{len(c_coords)}` does not match the length of dimension 'c'"
                f" with length {data.shape[cls.dims.dims.index('c')]}."
            )

        data = to_spatial_image(array_like=data, dims=cls.dims.dims, c_coords=c_coords, **kwargs)
        # parse transformations
        _parse_transformations(data, transformations)
        # convert to multiscale if needed
        if scale_factors is not None:
            parsed_transform = _get_transformations(data)
            # delete transforms
            del data.attrs["transform"]
            data = to_multiscale(
                data,
                scale_factors=scale_factors,
                method=method,
                chunks=chunks,
            )
            _parse_transformations(data, parsed_transform)
        cls()._check_chunk_size_not_too_large(data)
        # recompute coordinates for (multiscale) spatial image
        return compute_coordinates(data)

    @singledispatchmethod
    def validate(self, data: Any) -> None:
        """
        Validate data.

        Parameters
        ----------
        data
            Data to validate.

        Raises
        ------
        ValueError
            If data is not valid.
        """
        raise ValueError(
            f"Unsupported data type: {type(data)}. Please use .parse() from Image2DModel, Image3DModel, Labels2DModel "
            "or Labels3DModel to construct data that is guaranteed to be valid."
        )

    @validate.register(DataArray)
    def _(self, data: DataArray) -> None:
        super().validate(data)
        self._check_chunk_size_not_too_large(data)

    @validate.register(DataTree)
    def _(self, data: DataTree) -> None:
        for j, k in zip(data.keys(), [f"scale{i}" for i in np.arange(len(data.keys()))], strict=True):
            if j != k:
                raise ValueError(f"Wrong key for multiscale data, found: `{j}`, expected: `{k}`.")
        name = {list(data[i].data_vars.keys())[0] for i in data}
        if len(name) != 1:
            raise ValueError(f"Expected exactly one data variable for the datatree: found `{name}`.")
        name = list(name)[0]
        for d in data:
            super().validate(data[d][name])
        self._check_chunk_size_not_too_large(data)

    def _check_chunk_size_not_too_large(self, data: DataArray | DataTree) -> None:
        if isinstance(data, DataArray):
            try:
                max_per_dimension: dict[int, int] = {}
                if isinstance(data.chunks, list | tuple):
                    for i, sizes in enumerate(data.chunks):
                        max_per_dimension[i] = max(sizes)
                else:
                    assert isinstance(data.chunks, dict)
                    for i, sizes in enumerate(data.chunks.values()):
                        max_per_dimension[i] = max(sizes)
            except ValueError:
                warnings.warn(
                    f"Unable to estimate the maximum chunk size for the data: {sizes}. Please report this bug.",
                    UserWarning,
                    stacklevel=2,
                )
                return
            n_elems = np.array(list(max_per_dimension.values())).prod().item()
            usage = n_elems * data.dtype.itemsize
            if usage > LARGE_CHUNK_THRESHOLD_BYTES:
                warnings.warn(
                    f"Detected chunks larger than: {usage} > {LARGE_CHUNK_THRESHOLD_BYTES} bytes. "
                    "This can lead to low "
                    "performance and memory issues downstream, and sometimes cause compression errors when writing "
                    "(https://github.com/scverse/spatialdata/issues/812#issuecomment-2575983527). Please consider using"
                    " 1) smaller chunks and/or 2) using a multiscale representation for the raster data.\n"
                    "1) Smaller chunks can be achieved by using the `chunks` argument in the `parse()` function or by "
                    "calling the `chunk()` method on `DataArray`/`DataTree` objects.\n"
                    "2) Multiscale representations can be achieved by using the `scale_factors` argument in the "
                    "`parse()` function.\n"
                    "You can suppress this warning by increasing the value of "
                    "`spatialdata.config.LARGE_CHUNK_THRESHOLD_BYTES`.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            assert isinstance(data, DataTree)
            name = {list(data[i].data_vars.keys())[0] for i in data}
            assert len(name) == 1
            name = list(name)[0]
            for d in data:
                super().validate(data[d][name])
            self._check_chunk_size_not_too_large(data[d][name])


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

    @classmethod
    def parse(  # noqa: D102
        self,
        *args: Any,
        **kwargs: Any,
    ) -> DataArray | DataTree:
        if kwargs.get("c_coords") is not None:
            raise ValueError("`c_coords` is not supported for labels")
        if kwargs.get("scale_factors") is not None and kwargs.get("method") is None:
            # Override default scaling method to preserve labels
            kwargs["method"] = Methods.DASK_IMAGE_NEAREST
        return super().parse(*args, **kwargs)


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

    @classmethod
    def parse(self, *args: Any, **kwargs: Any) -> DataArray | DataTree:  # noqa: D102
        if kwargs.get("c_coords") is not None:
            raise ValueError("`c_coords` is not supported for labels")
        if kwargs.get("scale_factors") is not None and kwargs.get("method") is None:
            # Override default scaling method to preserve labels
            kwargs["method"] = Methods.DASK_IMAGE_NEAREST
        return super().parse(*args, **kwargs)


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


class ShapesModel:
    GEOMETRY_KEY = "geometry"
    GEOS_KEY = "geos"
    TYPE_KEY = "type"
    NAME_KEY = "name"
    RADIUS_KEY = "radius"
    TRANSFORM_KEY = "transform"
    ATTRS_KEY = ATTRS_KEY

    @classmethod
    def validate(cls, data: GeoDataFrame) -> None:
        """
        Validate data.

        Parameters
        ----------
        data
            :class:`geopandas.GeoDataFrame` to validate.

        Returns
        -------
        None
        """
        SUGGESTION = " Please use ShapesModel.parse() to construct data that is guaranteed to be valid."
        if cls.GEOMETRY_KEY not in data:
            raise KeyError(f"GeoDataFrame must have a column named `{cls.GEOMETRY_KEY}`." + SUGGESTION)
        if not isinstance(data[cls.GEOMETRY_KEY], GeoSeries):
            raise ValueError(f"Column `{cls.GEOMETRY_KEY}` must be a GeoSeries." + SUGGESTION)
        if len(data[cls.GEOMETRY_KEY]) == 0:
            raise ValueError(f"Column `{cls.GEOMETRY_KEY}` is empty." + SUGGESTION)
        geom_ = data[cls.GEOMETRY_KEY].values[0]
        if not isinstance(geom_, Polygon | MultiPolygon | Point):
            raise ValueError(
                f"Column `{cls.GEOMETRY_KEY}` can only contain `Point`, `Polygon` or `MultiPolygon` shapes,"
                f"but it contains {type(geom_)}." + SUGGESTION
            )
        if isinstance(geom_, Point):
            if cls.RADIUS_KEY not in data.columns:
                raise ValueError(f"Column `{cls.RADIUS_KEY}` not found." + SUGGESTION)
            radii = data[cls.RADIUS_KEY].values
            if np.any(radii <= 0):
                raise ValueError("Radii of circles must be positive.")
            if np.any(np.isnan(radii)) or np.any(np.isinf(radii)):
                # using logger.warning instead of warnings.warn to avoid the warning to being silenced in some cases
                # (e.g. PyCharm console)
                logger.warning(
                    "Radii of circles must not be nan or inf (this warning will be turned into a ValueError in the "
                    "next code release). If you are seeing this warning after reading previously saved Xenium data, "
                    "please see https://github.com/scverse/spatialdata/discussions/657 for a solution. Otherwise, "
                    "please correct the radii of the circles before calling the parser function.",
                )
        if cls.TRANSFORM_KEY not in data.attrs:
            raise ValueError(f":class:`geopandas.GeoDataFrame` does not contain `{TRANSFORM_KEY}`." + SUGGESTION)
        if len(data) > 0:
            n = data.geometry.iloc[0]._ndim
            if n != 2:
                warnings.warn(
                    f"The geometry column of the GeoDataFrame has {n} dimensions, while 2 is expected. Please consider "
                    "discarding the third dimension as it could led to unexpected behaviors. To achieve so, you can use"
                    " `.force_2d()` if you are using `geopandas > 0.14.3, otherwise you can use `force_2d()` from "
                    "`spatialdata.models`.",
                    UserWarning,
                    stacklevel=2,
                )

    @classmethod
    def validate_shapes_not_mixed_types(cls, gdf: GeoDataFrame) -> None:
        """
        Check that the Shapes element is either composed of Point or Polygon/MultiPolygon.

        Parameters
        ----------
        gdf
            The Shapes element.

        Raises
        ------
        ValueError
            When the geometry column composing the object does not satisfy the type requirements.

        Notes
        -----
        This function is not called by ShapesModel.validate() because computing the unique types by default could be
        expensive.
        """
        values_geotypes = list(gdf.geom_type.unique())
        if values_geotypes == ["Point"]:
            return
        if set(values_geotypes).issubset(["Polygon", "MultiPolygon"]):
            return
        raise ValueError(
            "The geometry column of a Shapes element should either be composed of Point, either of "
            f"Polygon/MultyPolygon. Found: {values_geotypes}"
        )

    @singledispatchmethod
    @classmethod
    def parse(cls, data: Any, **kwargs: Any) -> GeoDataFrame:
        """
        Parse shapes data.

        Parameters
        ----------
        data
            Data to parse:

                - If :class:`numpy.ndarray`, it assumes the shapes are parsed as
                  ragged arrays, in case of :class:`shapely.Polygon` or :class:`shapely.MultiPolygon`.
                  Therefore additional arguments `offsets` and `geometry` must be provided
                - if `Path` or `str`, it's read as a GeoJSON file.
                - If :class:`geopandas.GeoDataFrame`, it's validated. The object needs to
                  have a column called `geometry` which is a :class:`geopandas.GeoSeries`
                  or `shapely` objects. Valid options are combinations of :class:`shapely.Polygon`
                  or :class:`shapely.MultiPolygon` or :class:`shapely.Point`.
                  If the geometries are `Point`, there must be another column called `radius`.

        geometry
            Geometry type of the shapes. The following geometries are supported:

                - 0: `Circles`
                - 3: `Polygon`
                - 6: `MultiPolygon`

        offsets
            In the case of :class:`shapely.Polygon` or :class:`shapely.MultiPolygon` shapes,
            in order to initialize the shapes from their ragged array representation,
            the offsets of the polygons must be provided.
            Alternatively you can call the parser as `ShapesModel.parse(data)`, where data is a
            `GeoDataFrame` object and ignore the `offset` parameter (recommended).
        radius
            Size of the `Circles`. It must be provided if the shapes are `Circles`.
        index
            Index of the shapes, must be of type `str`. If None, it's generated automatically.
        transformations
            Transformations of shapes.
        kwargs
            Additional arguments for GeoJSON reader.

        Returns
        -------
        :class:`geopandas.GeoDataFrame`
        """
        raise TypeError(f"ShapesModel.parse() does not support the type {type(data)}")

    @parse.register(np.ndarray)
    @classmethod
    def _(
        cls,
        data: np.ndarray,  # type: ignore[type-arg]
        geometry: Literal[0, 3, 6],  # [GeometryType.POINT, GeometryType.POLYGON, GeometryType.MULTIPOLYGON]
        offsets: tuple[ArrayLike, ...] | None = None,
        radius: float | ArrayLike | None = None,
        index: ArrayLike | None = None,
        transformations: MappingToCoordinateSystem_t | None = None,
    ) -> GeoDataFrame:
        geometry = GeometryType(geometry)
        data = from_ragged_array(geometry_type=geometry, coords=data, offsets=offsets)
        geo_df = GeoDataFrame({"geometry": data})
        if GeometryType(geometry).name == "POINT":
            if radius is None:
                raise ValueError("If `geometry` is `Circles`, `radius` must be provided.")
            geo_df[cls.RADIUS_KEY] = radius
        if index is not None:
            geo_df.index = index
        _parse_transformations(geo_df, transformations)
        cls.validate(geo_df)
        return geo_df

    @parse.register(str)
    @parse.register(Path)
    @classmethod
    def _(
        cls,
        data: str | Path,
        radius: float | ArrayLike | None = None,
        index: ArrayLike | None = None,
        transformations: Any | None = None,
        **kwargs: Any,
    ) -> GeoDataFrame:
        data = Path(data) if isinstance(data, str) else data

        gc: GeometryCollection = from_geojson(data.read_bytes(), **kwargs)
        if not isinstance(gc, GeometryCollection):
            raise ValueError(f"`{data}` does not contain a `GeometryCollection`.")
        geo_df = GeoDataFrame({"geometry": gc.geoms})
        if isinstance(geo_df["geometry"].iloc[0], Point):
            if radius is None:
                raise ValueError("If `geometry` is `Circles`, `radius` must be provided.")
            geo_df[cls.RADIUS_KEY] = radius
        if index is not None:
            geo_df.index = index
        _parse_transformations(geo_df, transformations)
        cls.validate(geo_df)
        return geo_df

    @parse.register(GeoDataFrame)
    @classmethod
    def _(
        cls,
        data: GeoDataFrame,
        transformations: MappingToCoordinateSystem_t | None = None,
    ) -> GeoDataFrame:
        if "geometry" not in data.columns:
            raise ValueError("`geometry` column not found in `GeoDataFrame`.")
        if isinstance(data["geometry"].iloc[0], Point) and cls.RADIUS_KEY not in data.columns:
            raise ValueError(f"Column `{cls.RADIUS_KEY}` not found.")
        _parse_transformations(data, transformations)
        cls.validate(data)
        return data


class PointsModel:
    INSTANCE_KEY = "instance_key"
    FEATURE_KEY = "feature_key"
    TRANSFORM_KEY = "transform"
    ATTRS_KEY = ATTRS_KEY
    NPARTITIONS = 1

    @classmethod
    def validate(cls, data: DaskDataFrame) -> None:
        """
        Validate data.

        Parameters
        ----------
        data
            :class:`dask.dataframe.core.DataFrame` to validate.

        Returns
        -------
        None
        """
        SUGGESTION = " Please use PointsModel.parse() to construct data that is guaranteed to be valid."
        for ax in [X, Y, Z]:
            # TODO: check why this can return int32 on windows.
            if ax in data.columns and data[ax].dtype not in [
                np.int32,
                np.float32,
                np.float64,
                np.int64,
            ]:
                raise ValueError(f"Column `{ax}` must be of type `int` or `float`.")
        if cls.TRANSFORM_KEY not in data.attrs:
            raise ValueError(
                f":attr:`dask.dataframe.core.DataFrame.attrs` does not contain `{cls.TRANSFORM_KEY}`." + SUGGESTION
            )
        if ATTRS_KEY in data.attrs and "feature_key" in data.attrs[ATTRS_KEY]:
            feature_key = data.attrs[ATTRS_KEY][cls.FEATURE_KEY]
            if not isinstance(data[feature_key].dtype, CategoricalDtype):
                logger.info(f"Feature key `{feature_key}`could be of type `pd.Categorical`. Consider casting it.")

    @singledispatchmethod
    @classmethod
    def parse(cls, data: Any, **kwargs: Any) -> DaskDataFrame:
        """
        Validate (or parse) points data.

        Parameters
        ----------
        data
            Data to parse:

                - If :class:`numpy.ndarray`, an `annotation` :class:`pandas.DataFrame`
                  can be provided, as well as a `feature_key` column in the `annotation` dataframe. Furthermore,
                  :class:`numpy.ndarray` is assumed to have shape `(n_points, axes)`, with `axes` being
                  "x", "y" and optionally "z".
                - If :class:`pandas.DataFrame`, a `coordinates` mapping can be provided
                  with key as *valid axes* ('x', 'y', 'z') and value as column names in dataframe. If the dataframe
                  already has columns named 'x', 'y' and 'z', the mapping can be omitted.

        annotation
            Annotation dataframe. Only if `data` is :class:`numpy.ndarray`. If data is an array, the index of the
            annotations will be used as the index of the parsed points.
        coordinates
            Mapping of axes names (keys) to column names (valus) in `data`. Only if `data` is
            :class:`pandas.DataFrame`. Example: {'x': 'my_x_column', 'y': 'my_y_column'}.
            If not provided and `data` is :class:`pandas.DataFrame`, and `x`, `y` and optionally `z` are column names,
            then they will be used as coordinates.
        feature_key
            Optional, feature key in `annotation` or `data`. Example use case: gene id categorical column describing the
            gene identity of each point.
        instance_key
            Optional, instance key in `annotation` or `data`. Example use case: cell id column, describing which cell
            a point belongs to. This argument is likely going to be deprecated:
            https://github.com/scverse/spatialdata/issues/503.
        transformations
            Transformations of points.
        kwargs
            Additional arguments for :func:`dask.dataframe.from_array`.

        Returns
        -------
        :class:`dask.dataframe.core.DataFrame`

        Notes
        -----
        The order of the columns of the dataframe returned by the parser is not guaranteed to be the same as the order
        of the columns in the dataframe passed as an argument.
        """
        raise TypeError(f"PointsModel.parse() does not support the type {type(data)}")

    @parse.register(np.ndarray)
    @classmethod
    def _(
        cls,
        data: np.ndarray,  # type: ignore[type-arg]
        annotation: pd.DataFrame | None = None,
        feature_key: str | None = None,
        instance_key: str | None = None,
        transformations: MappingToCoordinateSystem_t | None = None,
        **kwargs: Any,
    ) -> DaskDataFrame:
        if "npartitions" not in kwargs and "chunksize" not in kwargs:
            kwargs["npartitions"] = cls.NPARTITIONS
        assert len(data.shape) == 2
        ndim = data.shape[1]
        axes = [X, Y, Z][:ndim]
        index = annotation.index if annotation is not None else None
        df_dict = {ax: data[:, i] for i, ax in enumerate(axes)}
        df_kwargs = {"data": df_dict, "index": index}

        if annotation is not None:
            if feature_key is not None:
                df_dict[feature_key] = annotation[feature_key].astype(str).astype("category")
            if instance_key is not None:
                df_dict[instance_key] = annotation[instance_key]
            if Z not in axes and Z in annotation.columns:
                logger.info(f"Column `{Z}` in `annotation` will be ignored since the data is 2D.")
            for c in set(annotation.columns) - {feature_key, instance_key, X, Y, Z}:
                df_dict[c] = annotation[c]

        table: DaskDataFrame = dd.from_pandas(pd.DataFrame(**df_kwargs), **kwargs)
        return cls._add_metadata_and_validate(
            table,
            feature_key=feature_key,
            instance_key=instance_key,
            transformations=transformations,
        )

    @parse.register(pd.DataFrame)
    @parse.register(DaskDataFrame)
    @classmethod
    def _(
        cls,
        data: pd.DataFrame,
        coordinates: Mapping[str, str] | None = None,
        feature_key: str | None = None,
        instance_key: str | None = None,
        transformations: MappingToCoordinateSystem_t | None = None,
        **kwargs: Any,
    ) -> DaskDataFrame:
        if "npartitions" not in kwargs and "chunksize" not in kwargs:
            kwargs["npartitions"] = cls.NPARTITIONS
        if coordinates is None:
            if X in data.columns and Y in data.columns:
                coordinates = {X: X, Y: Y, Z: Z} if Z in data.columns else {X: X, Y: Y}
            else:
                raise ValueError(
                    f"Coordinates must be provided as a mapping of axes names (keys) to column names (values) in "
                    f"dataframe. Example: `{'x': 'my_x_column', 'y': 'my_y_column'}`."
                )
        ndim = len(coordinates)
        axes = [X, Y, Z][:ndim]
        if "sort" not in kwargs:
            index_monotonically_increasing = data.index.is_monotonic_increasing
            if not isinstance(index_monotonically_increasing, bool):
                index_monotonically_increasing = index_monotonically_increasing.compute()
            sort = index_monotonically_increasing
        else:
            sort = kwargs["sort"]
        if not sort:
            warnings.warn(
                "The index of the dataframe is not monotonic increasing. It is recommended to sort the data to "
                "adjust the order of the index before calling .parse() (or call `parse(sort=True)`) to avoid possible "
                "problems due to unknown divisions.",
                UserWarning,
                stacklevel=2,
            )
        if isinstance(data, pd.DataFrame):
            table: DaskDataFrame = dd.from_pandas(
                pd.DataFrame(
                    data[[coordinates[ax] for ax in axes]].to_numpy(),
                    columns=axes,
                    index=data.index,
                ),
                # we need to pass sort=True also when the index is sorted to ensure that the divisions are computed
                sort=sort,
                **kwargs,
            )
            # we cannot compute the divisions whne the index is not monotonically increasing and npartitions > 1
            if not table.known_divisions and (sort or table.npartitions == 1):
                table.divisions = table.compute_current_divisions()
            if feature_key is not None:
                feature_categ = dd.from_pandas(
                    data[feature_key].astype(str).astype("category"),
                    sort=sort,
                    **kwargs,
                )
                table[feature_key] = feature_categ
        elif isinstance(data, dd.DataFrame):
            table = data[[coordinates[ax] for ax in axes]]
            table.columns = axes
            if feature_key is not None:
                if data[feature_key].dtype.name == "category":
                    table[feature_key] = data[feature_key]
                else:
                    table[feature_key] = data[feature_key].astype(str).astype("category")
        if instance_key is not None:
            table[instance_key] = data[instance_key]
        for c in [X, Y, Z]:
            if c in coordinates and c != coordinates[c] and c in data.columns:
                logger.info(
                    f'The column "{coordinates[c]}" has now been renamed to "{c}"; the column "{c}" was already '
                    f"present in the dataframe, and will be dropped."
                )
        if Z not in axes and Z in data.columns:
            logger.info(f"Column `{Z}` in `data` will be ignored since the data is 2D.")
        for c in set(data.columns) - {
            feature_key,
            instance_key,
            *coordinates.values(),
            X,
            Y,
            Z,
        }:
            table[c] = data[c]

        validated = cls._add_metadata_and_validate(
            table,
            feature_key=feature_key,
            instance_key=instance_key,
            transformations=transformations,
        )

        # when `coordinates` is None, and no columns have been added or removed, preserves the original order
        old_columns = list(data.columns)
        new_columns = list(validated.columns)
        if set(new_columns) == set(old_columns) and new_columns != old_columns:
            col_order = [col for col in old_columns if col in new_columns]
            validated = validated[col_order]
        return validated

    @classmethod
    def _add_metadata_and_validate(
        cls,
        data: DaskDataFrame,
        feature_key: str | None = None,
        instance_key: str | None = None,
        transformations: MappingToCoordinateSystem_t | None = None,
    ) -> DaskDataFrame:
        assert isinstance(data, dd.DataFrame)
        if feature_key is not None or instance_key is not None:
            data.attrs[ATTRS_KEY] = {}
        if feature_key is not None:
            assert feature_key in data.columns
            data.attrs[ATTRS_KEY][cls.FEATURE_KEY] = feature_key
        if instance_key is not None:
            assert instance_key in data.columns
            data.attrs[ATTRS_KEY][cls.INSTANCE_KEY] = instance_key

        for c in data.columns:
            #  Here we are explicitly importing the categories
            #  but it is a convenient way to ensure that the categories are known.
            # It also just changes the state of the series, so it is not a big deal.
            if isinstance(data[c].dtype, CategoricalDtype) and not data[c].cat.known:
                try:
                    data[c] = data[c].cat.set_categories(data[c].head(1).cat.categories)
                except ValueError:
                    logger.info(f"Column `{c}` contains unknown categories. Consider casting it.")

        _parse_transformations(data, transformations)
        cls.validate(data)
        # false positive with the PyCharm mypy plugin
        return data


class TableModel:
    REGION_KEY = "region"
    REGION_KEY_KEY = "region_key"
    INSTANCE_KEY = "instance_key"
    ATTRS_KEY = ATTRS_KEY

    def _validate_set_region_key(self, data: AnnData, region_key: str | None = None) -> None:
        """
        Validate the region key in table.uns or set a new region key as the region key column.

        Parameters
        ----------
        data
            The AnnData table.
        region_key
            The region key to be validated and set in table.uns.


        Raises
        ------
        ValueError
            If no region_key is found in table.uns and no region_key is provided as an argument.
        ValueError
            If the specified region_key in table.uns is not present as a column in table.obs.
        ValueError
            If the specified region key column is not present in table.obs.
        """
        attrs = data.uns.get(ATTRS_KEY)
        if attrs is None:
            data.uns[ATTRS_KEY] = attrs = {}
        table_region_key = attrs.get(self.REGION_KEY_KEY)
        if not region_key:
            if not table_region_key:
                raise ValueError(
                    "No region_key in table.uns and no region_key provided as argument. Please specify 'region_key'."
                )
            if data.obs.get(attrs[TableModel.REGION_KEY_KEY]) is None:
                raise ValueError(
                    f"Specified region_key in table.uns '{table_region_key}' is not "
                    f"present as column in table.obs. Please specify region_key."
                )
        else:
            if region_key not in data.obs:
                raise ValueError(f"'{region_key}' column not present in table.obs")
            attrs[self.REGION_KEY_KEY] = region_key

    def _validate_set_instance_key(self, data: AnnData, instance_key: str | None = None) -> None:
        """
        Validate the instance_key in table.uns or set a new instance_key as the instance_key column.

        If no instance_key is provided as argument, the presence of instance_key in table.uns is checked and validated.
        If instance_key is provided, presence in table.obs will be validated and if present it will be set as the new
        instance_key in table.uns.

        Parameters
        ----------
        data
            The AnnData table.

        instance_key
            The instance_key to be validated and set in table.uns.

        Raises
        ------
        ValueError
            If no instance_key is provided as argument and no instance_key is found in the `uns` attribute of table.
        ValueError
            If no instance_key is provided and the instance_key in table.uns does not match any column in table.obs.
        ValueError
            If provided instance_key is not present as table.obs column.
        """
        attrs = data.uns.get(ATTRS_KEY)
        if attrs is None:
            data.uns[ATTRS_KEY] = {}

        if not instance_key:
            if not attrs.get(TableModel.INSTANCE_KEY):
                raise ValueError(
                    "No instance_key in table.uns and no instance_key provided as argument. Please "
                    "specify instance_key."
                )
            if data.obs.get(attrs[self.INSTANCE_KEY]) is None:
                raise ValueError(
                    f"Specified instance_key in table.uns '{attrs.get(self.INSTANCE_KEY)}' is not present"
                    f" as column in table.obs. Please specify instance_key."
                )
        if instance_key:
            if instance_key in data.obs:
                attrs[self.INSTANCE_KEY] = instance_key
            else:
                raise ValueError(f"Instance key column '{instance_key}' not found in table.obs.")

    def _validate_table_annotation_metadata(self, data: AnnData) -> None:
        """
        Validate annotation metadata.

        Parameters
        ----------
        data
            The AnnData object containing the table annotation data.

        Raises
        ------
        ValueError
            If any of the required metadata keys are not found in the `adata.uns` dictionary or the `adata.obs`
            dataframe.

            - If "region" is not found in `adata.uns['ATTRS_KEY']`.
            - If "region_key" is not found in `adata.uns['ATTRS_KEY']`.
            - If "instance_key" is not found in `adata.uns['ATTRS_KEY']`.
            - If `attr[self.REGION_KEY_KEY]` is not found in `adata.obs`, with attr = adata.uns['ATTRS_KEY']
            - If `attr[self.INSTANCE_KEY]` is not found in `adata.obs`.
            - If the regions in `adata.uns['ATTRS_KEY']['self.REGION_KEY']` and the unique values of
                `attr[self.REGION_KEY_KEY]` do not match.

        Notes
        -----
        This does not check whether the annotation target of the table is present in a given SpatialData object. Rather
        it is an internal validation of the annotation metadata of the table.

        """
        SUGGESTION = " Please use TableModel.parse() to construct data that is guaranteed to be valid."
        attr = data.uns[ATTRS_KEY]

        if "region" not in attr:
            raise ValueError(f"`region` not found in `adata.uns['{ATTRS_KEY}']`." + SUGGESTION)
        if "region_key" not in attr:
            raise ValueError(f"`region_key` not found in `adata.uns['{ATTRS_KEY}']`." + SUGGESTION)
        if "instance_key" not in attr:
            raise ValueError(f"`instance_key` not found in `adata.uns['{ATTRS_KEY}']`." + SUGGESTION)

        if attr[self.REGION_KEY_KEY] not in data.obs:
            raise ValueError(f"`{attr[self.REGION_KEY_KEY]}` not found in `adata.obs`. Please create the column.")
        if attr[self.INSTANCE_KEY] not in data.obs:
            raise ValueError(f"`{attr[self.INSTANCE_KEY]}` not found in `adata.obs`. Please create the column.")
        if (dtype := data.obs[attr[self.INSTANCE_KEY]].dtype) not in [
            int,
            np.int16,
            np.uint16,
            np.int32,
            np.uint32,
            np.int64,
            np.uint64,
            "O",
        ] or (dtype == "O" and (val_dtype := type(data.obs[attr[self.INSTANCE_KEY]].iloc[0])) is not str):
            dtype = dtype if dtype != "O" else val_dtype
            raise TypeError(
                f"Only int, np.int16, np.int32, np.int64, uint equivalents or string allowed as dtype for "
                f"instance_key column in obs. Dtype found to be {dtype}"
            )
        expected_regions = attr[self.REGION_KEY] if isinstance(attr[self.REGION_KEY], list) else [attr[self.REGION_KEY]]
        found_regions = data.obs[attr[self.REGION_KEY_KEY]].unique().tolist()
        if len(set(expected_regions).symmetric_difference(set(found_regions))) > 0:
            raise ValueError(f"Regions in the AnnData object and `{attr[self.REGION_KEY_KEY]}` do not match.")

    def validate(
        self,
        data: AnnData,
    ) -> AnnData:
        """
        Validate the data.

        Parameters
        ----------
        data
            The data to validate.

        Returns
        -------
        The validated data.
        """
        validate_table_attr_keys(data)
        if ATTRS_KEY not in data.uns:
            return data

        self._validate_table_annotation_metadata(data)

        return data

    @classmethod
    def parse(
        cls,
        adata: AnnData,
        region: str | list[str] | None = None,
        region_key: str | None = None,
        instance_key: str | None = None,
        overwrite_metadata: bool = False,
    ) -> AnnData:
        """
        Parse the :class:`anndata.AnnData` to be compatible with the model.

        Parameters
        ----------
        adata
            The AnnData object.
        region
            Region(s) to be used.
        region_key
            Key in `adata.obs` that specifies the region.
        instance_key
            Key in `adata.obs` that specifies the instance.
        overwrite_metadata
            If `True`, the `region`, `region_key` and `instance_key` metadata will be overwritten.

        Returns
        -------
        The parsed data.
        """
        validate_table_attr_keys(adata)
        # either all live in adata.uns or all be passed in as argument
        n_args = sum([region is not None, region_key is not None, instance_key is not None])
        if n_args == 0:
            if cls.ATTRS_KEY not in adata.uns:
                # table not annotating any element
                return adata
            attr = adata.uns[cls.ATTRS_KEY]
            region = attr[cls.REGION_KEY]
            region_key = attr[cls.REGION_KEY_KEY]
            instance_key = attr[cls.INSTANCE_KEY]
        elif n_args > 0 and not overwrite_metadata and cls.ATTRS_KEY in adata.uns:
            raise ValueError(
                f"`{cls.REGION_KEY}`, `{cls.REGION_KEY_KEY}` and / or `{cls.INSTANCE_KEY}` is/has been passed as"
                f" argument(s). However, `adata.uns[{cls.ATTRS_KEY!r}]` has already been set."
            )

        if cls.ATTRS_KEY not in adata.uns:
            adata.uns[cls.ATTRS_KEY] = {}

        if region is None:
            raise ValueError(f"`{cls.REGION_KEY}` must be provided.")
        if region_key is None:
            raise ValueError(f"`{cls.REGION_KEY_KEY}` must be provided.")
        if instance_key is None:
            raise ValueError("`instance_key` must be provided.")

        if isinstance(region, np.ndarray):
            region = region.tolist()
        region_: list[str] = region if isinstance(region, list) else [region]
        if not adata.obs[region_key].isin(region_).all():
            raise ValueError(f"`adata.obs[{region_key}]` values do not match with `{cls.REGION_KEY}` values.")

        adata.uns[cls.ATTRS_KEY][cls.REGION_KEY] = region
        adata.uns[cls.ATTRS_KEY][cls.REGION_KEY_KEY] = region_key
        adata.uns[cls.ATTRS_KEY][cls.INSTANCE_KEY] = instance_key

        # note! this is an expensive check and therefore we skip it during validation
        # https://github.com/scverse/spatialdata/issues/715
        grouped = adata.obs.groupby(region_key, observed=True)
        grouped_size = grouped.size()
        grouped_nunique = grouped.nunique()
        not_unique = grouped_size[grouped_size != grouped_nunique[instance_key]].index.tolist()
        if not_unique:
            raise ValueError(
                f"Instance key column for region(s) `{', '.join(not_unique)}` does not contain only unique values"
            )

        attr = {
            "region": region,
            "region_key": region_key,
            "instance_key": instance_key,
        }
        adata.uns[cls.ATTRS_KEY] = attr
        cls().validate(adata)
        return convert_region_column_to_categorical(adata)


Schema_t: TypeAlias = (
    type[Image2DModel]
    | type[Image3DModel]
    | type[Labels2DModel]
    | type[Labels3DModel]
    | type[PointsModel]
    | type[ShapesModel]
    | type[TableModel]
)


def get_model(
    e: SpatialElement,
) -> Schema_t:
    """
    Get the model for the given element.

    Parameters
    ----------
    e
        The element.

    Returns
    -------
    The SpatialData model.
    """

    def _validate_and_return(
        schema: Schema_t,
        e: SpatialElement,
    ) -> Schema_t:
        schema().validate(e)
        return schema

    if isinstance(e, DataArray | DataTree):
        axes = get_axes_names(e)
        if "c" in axes:
            if "z" in axes:
                return _validate_and_return(Image3DModel, e)
            return _validate_and_return(Image2DModel, e)
        if "z" in axes:
            return _validate_and_return(Labels3DModel, e)
        return _validate_and_return(Labels2DModel, e)
    if isinstance(e, GeoDataFrame):
        return _validate_and_return(ShapesModel, e)
    if isinstance(e, DaskDataFrame):
        return _validate_and_return(PointsModel, e)
    if isinstance(e, AnnData):
        return _validate_and_return(TableModel, e)
    raise TypeError(f"Unsupported type {type(e)}")


def get_table_keys(table: AnnData) -> tuple[str | list[str], str, str]:
    """
    Get the table keys giving information about what spatial element is annotated.

    The first element returned gives information regarding which spatial elements are annotated by the table, the second
    element gives information which column in table.obs contains the information which spatial element is annotated
    by each row in the table and the instance key indicates the column in obs giving information of the id of each row.

    Parameters
    ----------
    table:
        AnnData table for which to retrieve the spatialdata_attrs keys.

    Returns
    -------
    The keys in table.uns['spatialdata_attrs']
    """
    if table.uns.get(TableModel.ATTRS_KEY):
        attrs = table.uns[TableModel.ATTRS_KEY]
        return (
            attrs[TableModel.REGION_KEY],
            attrs[TableModel.REGION_KEY_KEY],
            attrs[TableModel.INSTANCE_KEY],
        )

    raise ValueError(
        "No spatialdata_attrs key found in table.uns, therefore, no table keys found. Please parse the table."
    )


def _get_region_metadata_from_region_key_column(table: AnnData) -> list[str]:
    _, region_key, instance_key = get_table_keys(table)
    region_key_column = table.obs[region_key]
    if not isinstance(region_key_column.dtype, CategoricalDtype):
        warnings.warn(
            f"The region key column `{region_key}` is not of type `pd.Categorical`. Consider casting it to "
            f"improve performance.",
            UserWarning,
            stacklevel=2,
        )
        annotated_regions = region_key_column.unique().tolist()
    else:
        annotated_regions = table.obs[region_key].cat.remove_unused_categories().cat.categories.unique().tolist()
    assert isinstance(annotated_regions, list)
    return annotated_regions
