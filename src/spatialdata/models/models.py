"""Models and schema for SpatialData."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from functools import singledispatchmethod
from pathlib import Path
from typing import Any, Literal, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
from anndata import AnnData
from dask.array.core import Array as DaskArray
from dask.array.core import from_array
from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame, GeoSeries
from multiscale_spatial_image import to_multiscale
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from multiscale_spatial_image.to_multiscale.to_multiscale import Methods
from pandas import CategoricalDtype
from shapely._geometry import GeometryType
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.collection import GeometryCollection
from shapely.io import from_geojson, from_ragged_array
from spatial_image import SpatialImage, to_spatial_image
from xarray import DataArray
from xarray_schema.components import (
    ArrayTypeSchema,
    AttrSchema,
    AttrsSchema,
    DimsSchema,
)
from xarray_schema.dataarray import DataArraySchema

from spatialdata._logging import logger
from spatialdata._types import ArrayLike
from spatialdata.models import C, X, Y, Z, get_axes_names
from spatialdata.models._utils import (
    DEFAULT_COORDINATE_SYSTEM,
    TRANSFORM_KEY,
    MappingToCoordinateSystem_t,
    SpatialElement,
    _validate_mapping_to_coordinate_system_type,
)
from spatialdata.transformations._utils import (
    _get_transformations,
    _set_transformations,
    compute_coordinates,
)
from spatialdata.transformations.transformations import BaseTransformation, Identity

# Types
Chunks_t = Union[
    int,
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
    Mapping[Any, Union[None, int, tuple[int, ...]]],
]
ScaleFactors_t = Sequence[Union[dict[str, int], int]]

Transform_s = AttrSchema(BaseTransformation, None)


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

    @classmethod
    def parse(
        cls,
        data: ArrayLike | DataArray | DaskArray,
        dims: Sequence[str] | None = None,
        transformations: MappingToCoordinateSystem_t | None = None,
        scale_factors: ScaleFactors_t | None = None,
        method: Methods | None = None,
        chunks: Chunks_t | None = None,
        **kwargs: Any,
    ) -> SpatialImage | MultiscaleSpatialImage:
        """
        Validate (or parse) raster data.

        Parameters
        ----------
        data
            Data to validate (or parse). The shape of the data should be c(z)yx for 2D (3D) images and (z)yx for 2D (
            3D) labels. If you have a 2D image with shape yx, you can use :func:`numpy.expand_dims` (or an equivalent
            function) to add a channel dimension.
        dims
            Dimensions of the data.
        transformations
            Transformations to apply to the data.
        scale_factors
            Scale factors to apply for multiscale.
            If not None, a :class:`multiscale_spatial_image.MultiscaleSpatialImage` is returned.
        method
            Method to use for multiscale.
        chunks
            Chunks to use for dask array.

        Returns
        -------
        :class:`spatial_image.SpatialImage` or
        :class:`multiscale_spatial_image.MultiscaleSpatialImage`.
        """
        if transformations:
            transformations = transformations.copy()
        if "name" in kwargs:
            raise ValueError("The `name` argument is not (yet) supported for raster data.")
        # if dims is specified inside the data, get the value of dims from the data
        if isinstance(data, (DataArray, SpatialImage)):
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
        elif isinstance(data, (np.ndarray, DaskArray)):
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
        data = to_spatial_image(array_like=data, dims=cls.dims.dims, **kwargs)
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
        raise ValueError(f"Unsupported data type: {type(data)}.")

    @validate.register(SpatialImage)
    def _(self, data: SpatialImage) -> None:
        super().validate(data)

    @validate.register(MultiscaleSpatialImage)
    def _(self, data: MultiscaleSpatialImage) -> None:
        for j, k in zip(data.keys(), [f"scale{i}" for i in np.arange(len(data.keys()))]):
            if j != k:
                raise ValueError(f"Wrong key for multiscale data, found: `{j}`, expected: `{k}`.")
        name = {list(data[i].data_vars.keys())[0] for i in data}
        if len(name) > 1:
            raise ValueError(f"Wrong name for datatree: `{name}`.")
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


class ShapesModel:
    GEOMETRY_KEY = "geometry"
    ATTRS_KEY = "spatialdata_attrs"
    GEOS_KEY = "geos"
    TYPE_KEY = "type"
    NAME_KEY = "name"
    RADIUS_KEY = "radius"
    TRANSFORM_KEY = "transform"

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
        if cls.GEOMETRY_KEY not in data:
            raise KeyError(f"GeoDataFrame must have a column named `{cls.GEOMETRY_KEY}`.")
        if not isinstance(data[cls.GEOMETRY_KEY], GeoSeries):
            raise ValueError(f"Column `{cls.GEOMETRY_KEY}` must be a GeoSeries.")
        if len(data[cls.GEOMETRY_KEY]) == 0:
            raise ValueError(f"Column `{cls.GEOMETRY_KEY}` is empty.")
        geom_ = data[cls.GEOMETRY_KEY].values[0]
        if not isinstance(geom_, (Polygon, MultiPolygon, Point)):
            raise ValueError(
                f"Column `{cls.GEOMETRY_KEY}` can only contain `Point`, `Polygon` or `MultiPolygon` shapes,"
                f"but it contains {type(geom_)}."
            )
        if isinstance(geom_, Point):
            if cls.RADIUS_KEY not in data.columns:
                raise ValueError(f"Column `{cls.RADIUS_KEY}` not found.")
            radii = data[cls.RADIUS_KEY].values
            if np.any(radii <= 0):
                raise ValueError("Radii of circles must be positive.")
        if cls.TRANSFORM_KEY not in data.attrs:
            raise ValueError(f":class:`geopandas.GeoDataFrame` does not contain `{TRANSFORM_KEY}`.")
        if len(data) > 0:
            n = data.geometry.iloc[0]._ndim
            if n != 2:
                warnings.warn(
                    f"The geometry column of the GeoDataFrame has {n} dimensions, while 2 is expected. Please consider "
                    "discarding the third dimension as it could led to unexpected behaviors.",
                    UserWarning,
                    stacklevel=2,
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
                  ragged arrays, in case of (Multi)`Polygons`.
                  Therefore additional arguments `offsets` and `geometry` must be provided
                - if `Path` or `str`, it's read as a GeoJSON file.
                - If :class:`geopandas.GeoDataFrame`, it's validated. The object needs to
                  have a column called `geometry` which is a :class:`geopandas.GeoSeries`
                  or `shapely` objects. Valid options are combinations of `Polygon`/
                  `MultiPolygon` or `Point`. If the geometries are `Point`, there must be
                  another column called `radius`.

        geometry
            Geometry type of the shapes. The following geometries are supported:

                - 0: `Circles`
                - 3: `Polygon`
                - 6: `MultiPolygon`

        offsets
            In the case of (Multi)`Polygons` shapes, in order to initialize the shapes from their
            ragged array representation, the offsets of the polygons must be provided.
            Alternatively you can call the parser as `ShapesModel.parse(data)`, where data is a
            `GeoDataFrame` object and ignore the `offset` paramter (recommended).
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
        raise NotImplementedError()

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
    ATTRS_KEY = "spatialdata_attrs"
    INSTANCE_KEY = "instance_key"
    FEATURE_KEY = "feature_key"
    TRANSFORM_KEY = "transform"
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
        for ax in [X, Y, Z]:
            if ax in data.columns:
                # TODO: check why this can return int32 on windows.
                assert data[ax].dtype in [np.int32, np.float32, np.float64, np.int64]
        if cls.TRANSFORM_KEY not in data.attrs:
            raise ValueError(f":attr:`dask.dataframe.core.DataFrame.attrs` does not contain `{cls.TRANSFORM_KEY}`.")
        if cls.ATTRS_KEY in data.attrs and "feature_key" in data.attrs[cls.ATTRS_KEY]:
            feature_key = data.attrs[cls.ATTRS_KEY][cls.FEATURE_KEY]
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
                  must be provided, as well as the `feature_key` in the `annotation`. Furthermore,
                  :class:`numpy.ndarray` is assumed to have shape `(n_points, axes)`, with `axes` being
                  "x", "y" and optionally "z".
                - If :class:`pandas.DataFrame`, a `coordinates` mapping must be provided
                  with key as *valid axes* and value as column names in dataframe.

        annotation
            Annotation dataframe. Only if `data` is :class:`numpy.ndarray`. If data is an array, the index of the
            annotations will be used as the index of the parsed points.
        coordinates
            Mapping of axes names (keys) to column names (valus) in `data`. Only if `data` is
            :class:`pandas.DataFrame`. Example: {'x': 'my_x_column', 'y': 'my_y_column'}.
            If not provided and `data` is :class:`pandas.DataFrame`, and `x`, `y` and optinally `z` are column names,
            then they will be used as coordinates.
        feature_key
            Feature key in `annotation` or `data`.
        instance_key
            Instance key in `annotation` or `data`.
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
        raise NotImplementedError()

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
        table: DaskDataFrame = dd.from_pandas(pd.DataFrame(data, columns=axes, index=index), **kwargs)  # type: ignore[attr-defined]
        if annotation is not None:
            if feature_key is not None:
                feature_categ = dd.from_pandas(  # type: ignore[attr-defined]
                    annotation[feature_key].astype(str).astype("category"), **kwargs
                )
                table[feature_key] = feature_categ
            if instance_key is not None:
                table[instance_key] = annotation[instance_key]
            if Z not in axes and Z in annotation.columns:
                logger.info(f"Column `{Z}` in `annotation` will be ignored since the data is 2D.")
            for c in set(annotation.columns) - {feature_key, instance_key, X, Y, Z}:
                table[c] = dd.from_pandas(annotation[c], **kwargs)  # type: ignore[attr-defined]
            return cls._add_metadata_and_validate(
                table, feature_key=feature_key, instance_key=instance_key, transformations=transformations
            )
        return cls._add_metadata_and_validate(table, transformations=transformations)

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
        if isinstance(data, pd.DataFrame):
            table: DaskDataFrame = dd.from_pandas(  # type: ignore[attr-defined]
                pd.DataFrame(data[[coordinates[ax] for ax in axes]].to_numpy(), columns=axes, index=data.index),
                **kwargs,
            )
            if feature_key is not None:
                feature_categ = dd.from_pandas(
                    data[feature_key].astype(str).astype("category"),
                    **kwargs,
                )  # type: ignore[attr-defined]
                table[feature_key] = feature_categ
        elif isinstance(data, dd.DataFrame):  # type: ignore[attr-defined]
            table = data[[coordinates[ax] for ax in axes]]
            table.columns = axes
            if feature_key is not None and data[feature_key].dtype.name != "category":
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
        for c in set(data.columns) - {feature_key, instance_key, *coordinates.values(), X, Y, Z}:
            table[c] = data[c]

        # when `coordinates` is None, and no columns have been added or removed, preserves the original order
        # here I tried to fix https://github.com/scverse/spatialdata/issues/486, didn't work
        # old_columns = list(data.columns)
        # new_columns = list(table.columns)
        # if new_columns == set(old_columns) and new_columns != old_columns:
        #     col_order = [col for col in old_columns if col in new_columns]
        #     table = table[col_order]

        return cls._add_metadata_and_validate(
            table, feature_key=feature_key, instance_key=instance_key, transformations=transformations
        )

    @classmethod
    def _add_metadata_and_validate(
        cls,
        data: DaskDataFrame,
        feature_key: str | None = None,
        instance_key: str | None = None,
        transformations: MappingToCoordinateSystem_t | None = None,
    ) -> DaskDataFrame:
        assert isinstance(data, dd.DataFrame)  # type: ignore[attr-defined]
        if feature_key is not None or instance_key is not None:
            data.attrs[cls.ATTRS_KEY] = {}
        if feature_key is not None:
            assert feature_key in data.columns
            data.attrs[cls.ATTRS_KEY][cls.FEATURE_KEY] = feature_key
        if instance_key is not None:
            assert instance_key in data.columns
            data.attrs[cls.ATTRS_KEY][cls.INSTANCE_KEY] = instance_key

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
        return data  # type: ignore[no-any-return]


class TableModel:
    ATTRS_KEY = "spatialdata_attrs"
    REGION_KEY = "region"
    REGION_KEY_KEY = "region_key"
    INSTANCE_KEY = "instance_key"

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
        attrs = data.uns.get(self.ATTRS_KEY)
        if attrs is None:
            data.uns[self.ATTRS_KEY] = attrs = {}
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
        attrs = data.uns.get(self.ATTRS_KEY)
        if attrs is None:
            data.uns[self.ATTRS_KEY] = {}

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
        attr = data.uns[self.ATTRS_KEY]

        if "region" not in attr:
            raise ValueError(f"`region` not found in `adata.uns['{self.ATTRS_KEY}']`.")
        if "region_key" not in attr:
            raise ValueError(f"`region_key` not found in `adata.uns['{self.ATTRS_KEY}']`.")
        if "instance_key" not in attr:
            raise ValueError(f"`instance_key` not found in `adata.uns['{self.ATTRS_KEY}']`.")

        if attr[self.REGION_KEY_KEY] not in data.obs:
            raise ValueError(f"`{attr[self.REGION_KEY_KEY]}` not found in `adata.obs`.")
        if attr[self.INSTANCE_KEY] not in data.obs:
            raise ValueError(f"`{attr[self.INSTANCE_KEY]}` not found in `adata.obs`.")
        if (dtype := data.obs[attr[self.INSTANCE_KEY]].dtype) not in [int, np.int16, np.int32, np.int64, "O"] or (
            dtype == "O" and (val_dtype := type(data.obs[attr[self.INSTANCE_KEY]].iloc[0])) != str
        ):
            dtype = dtype if dtype != "O" else val_dtype
            raise TypeError(
                f"Only int, np.int16, np.int32, np.int64 or string allowed as dtype for "
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
        if self.ATTRS_KEY not in data.uns:
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

        Returns
        -------
        The parsed data.
        """
        # either all live in adata.uns or all be passed in as argument
        n_args = sum([region is not None, region_key is not None, instance_key is not None])
        if n_args == 0:
            return adata
        if n_args > 0:
            if cls.ATTRS_KEY in adata.uns:
                raise ValueError(
                    f"`{cls.REGION_KEY}`, `{cls.REGION_KEY_KEY}` and / or `{cls.INSTANCE_KEY}` is/has been passed as"
                    f"as argument(s). However, `adata.uns[{cls.ATTRS_KEY!r}]` has already been set."
                )
        elif cls.ATTRS_KEY in adata.uns:
            attr = adata.uns[cls.ATTRS_KEY]
            region = attr[cls.REGION_KEY]
            region_key = attr[cls.REGION_KEY_KEY]
            instance_key = attr[cls.INSTANCE_KEY]

        if region_key is None:
            raise ValueError(f"`{cls.REGION_KEY_KEY}` must be provided.")
        if isinstance(region, np.ndarray):
            region = region.tolist()
        if region is None:
            raise ValueError(f"`{cls.REGION_KEY}` must be provided.")
        region_: list[str] = region if isinstance(region, list) else [region]
        if not adata.obs[region_key].isin(region_).all():
            raise ValueError(f"`adata.obs[{region_key}]` values do not match with `{cls.REGION_KEY}` values.")
        if not isinstance(adata.obs[region_key].dtype, CategoricalDtype):
            warnings.warn(
                f"Converting `{cls.REGION_KEY_KEY}: {region_key}` to categorical dtype.", UserWarning, stacklevel=2
            )
            adata.obs[region_key] = pd.Categorical(adata.obs[region_key])
        if instance_key is None:
            raise ValueError("`instance_key` must be provided.")

        grouped = adata.obs.groupby(region_key, observed=True)
        grouped_size = grouped.size()
        grouped_nunique = grouped.nunique()
        not_unique = grouped_size[grouped_size != grouped_nunique[instance_key]].index.tolist()
        if not_unique:
            raise ValueError(
                f"Instance key column for region(s) `{', '.join(not_unique)}` does not contain only unique integers"
            )

        attr = {"region": region, "region_key": region_key, "instance_key": instance_key}
        adata.uns[cls.ATTRS_KEY] = attr
        cls().validate(adata)
        return adata


Schema_t = Union[
    type[Image2DModel],
    type[Image3DModel],
    type[Labels2DModel],
    type[Labels3DModel],
    type[PointsModel],
    type[ShapesModel],
    type[TableModel],
]


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

    if isinstance(e, (SpatialImage, MultiscaleSpatialImage)):
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
        return attrs[TableModel.REGION_KEY], attrs[TableModel.REGION_KEY_KEY], attrs[TableModel.INSTANCE_KEY]

    raise ValueError(
        "No spatialdata_attrs key found in table.uns, therefore, no table keys found. Please parse the table."
    )


def check_target_region_column_symmetry(table: AnnData, region_key: str, target: str | pd.Series) -> None:
    """
    Check region and region_key column symmetry.

    This checks whether the specified targets are also present in the region key column in obs and raises an error
    if this is not the case.

    Parameters
    ----------
    table
        Table annotating specific SpatialElements
    region_key
        The column in obs containing for each row which SpatialElement is annotated by that row.
    target
         Name of target(s) SpatialElement(s)

    Raises
    ------
    ValueError
        If there is a mismatch between specified target regions and regions in the region key column of table.obs.

    Example
    -------
    Assuming we have a table with region column in obs given by `region_key` called 'region' for which we want to check
    whether it contains the specified annotation targets in the `target` variable as `pd.Series['region1', 'region2']`:

    ```python
    check_target_region_column_symmetry(table, region_key=region_key, target=target)
    ```

    This returns None if both specified targets are present in the region_key obs column. In this case the annotation
    targets can be safely set. If not then a ValueError is raised stating the elements that are not shared between
    the region_key column in obs and the specified targets.
    """
    found_regions = set(table.obs[region_key].unique().tolist())
    target_element_set = [target] if isinstance(target, str) else target
    symmetric_difference = found_regions.symmetric_difference(target_element_set)
    if symmetric_difference:
        raise ValueError(
            f"Mismatch(es) found between regions in region column in obs and target element: "
            f"{', '.join(diff for diff in symmetric_difference)}"
        )
