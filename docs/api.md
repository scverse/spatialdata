# API

```{eval-rst}
.. module:: spatialdata
```

## SpatialData

The `SpatialData` class.

```{eval-rst}
.. autosummary::
    :toctree: generated

    SpatialData
```

## Operations

Operations on `SpatialData` objects.

```{eval-rst}
.. autosummary::
    :toctree: generated

    bounding_box_query
    polygon_query
    concatenate
    rasterize
    transform
    aggregate
```

### Utilities

```{eval-rst}
.. autosummary::
    :toctree: generated

    unpad_raster
```

## Models

The elements (building-blocks) that consitute `SpatialData`.

```{eval-rst}
.. currentmodule:: spatialdata.models
.. autosummary::
    :toctree: generated

    Image2DModel
    Image3DModel
    Labels2DModel
    Labels3DModel
    ShapesModel
    PointsModel
    TableModel
```

### Utilities

```{eval-rst}
.. autosummary::
    :toctree: generated

    get_model
    SpatialElement
    get_axes_names
    get_spatial_axes
    points_geopandas_to_dask_dataframe
    points_dask_dataframe_to_geopandas
```

## Transformations

The transformations that can be defined between elements and coordinate systems in `SpatialData`.

```{eval-rst}
.. currentmodule:: spatialdata.transformations

.. autosummary::
    :toctree: generated

    BaseTransformation
    Identity
    MapAxis
    Translation
    Scale
    Affine
    Sequence
```

### Utilities

```{eval-rst}
.. autosummary::
    :toctree: generated

    get_transformation
    set_transformation
    remove_transformation
    get_transformation_between_coordinate_systems
    get_transformation_between_landmarks
    align_elements_using_landmarks
```

## DataLoader

```{eval-rst}
.. currentmodule:: spatialdata.dataloader

.. autosummary::
    :toctree: generated

    ImageTilesDataset
```

## Input/output

```{eval-rst}
.. currentmodule:: spatialdata

.. autosummary::
    :toctree: generated

    read_zarr
    save_transformations
```
