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

### Operations

Operations in `SpatialData`.

```{eval-rst}
.. autosummary::
    :toctree: generated

    bounding_box_query
    transform
    rasterize
    concatenate
```

### Models

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

### Transformations

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

### Misc

```{eval-rst}
.. currentmodule:: spatialdata.dataloader

.. autosummary::
    :toctree: generated

    datasets.ImageTilesDataset
    utils.SpatialDataToDataDict
```
