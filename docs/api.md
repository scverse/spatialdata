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
    concatenate
    rasterize
    transform
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
    get_axis_names
    get_spatial_axes
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
```

## DataLoader

```{eval-rst}
.. currentmodule:: spatialdata.dataloader

.. autosummary::
    :toctree: generated

    ImageTilesDataset
```

## Readers

```{eval-rst}
.. currentmodule:: spatialdata

.. autosummary::
    :toctree: generated

    read_zarr
```
