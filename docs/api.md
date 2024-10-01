# API

```{eval-rst}
.. module:: spatialdata
```

## SpatialData

The `SpatialData` class (follow the link to explore its methods).

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
    get_values
    get_element_instances
    get_extent
    get_centroids
    join_spatialelement_table
    match_element_to_table
    get_centroids
    match_table_to_element
    concatenate
    transform
    rasterize
    rasterize_bins
    to_circles
    to_polygons
    aggregate
    map_raster
```

### Operations Utilities

```{eval-rst}
.. autosummary::
    :toctree: generated

    unpad_raster
    are_extents_equal
    deepcopy
    get_pyramid_levels
```

## Models

The elements (building-blocks) that constitute `SpatialData`.

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

### Models Utilities

```{eval-rst}
.. currentmodule:: spatialdata.models

.. autosummary::
    :toctree: generated

    get_model
    SpatialElement
    get_axes_names
    get_spatial_axes
    points_geopandas_to_dask_dataframe
    points_dask_dataframe_to_geopandas
    get_channels
    force_2d
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

### Transformations Utilities

```{eval-rst}
.. currentmodule:: spatialdata.transformations

.. autosummary::
    :toctree: generated

    get_transformation
    set_transformation
    remove_transformation
    get_transformation_between_coordinate_systems
    get_transformation_between_landmarks
    align_elements_using_landmarks
    remove_transformations_to_coordinate_system
```

## DataLoader

```{eval-rst}
.. currentmodule:: spatialdata.dataloader

.. autosummary::
    :toctree: generated

    ImageTilesDataset
```

## Input/Output

```{eval-rst}
.. currentmodule:: spatialdata

.. autosummary::
    :toctree: generated

    read_zarr
    save_transformations
    get_dask_backing_files
```

## Testing utilities

```{eval-rst}
.. currentmodule:: spatialdata.testing

.. autosummary::
    :toctree: generated

    assert_spatial_data_objects_are_identical
    assert_elements_are_identical
    assert_elements_dict_are_identical
```

## Datasets

Convenience small datasets

```{eval-rst}
.. currentmodule:: spatialdata.datasets

.. autosummary::
    :toctree: generated

    blobs
    blobs_annotating_element
    raccoon

```

## Data format (advanced topic)

The SpatialData format is defined as a set of versioned subclasses of :class:`spatialdata._io.format.SpatialDataFormat`, one per type of element.
These classes are useful to ensure backward compatibility whenever a major version change is introduced. We also provide pointers to the latest format.

### Raster format

```{eval-rst}
.. currentmodule:: spatialdata._io.format

.. autosummary::
    :toctree: generated

    CurrentRasterFormat
    RasterFormatV01
```

### Shapes format

```{eval-rst}
.. autosummary::
    :toctree: generated

    CurrentShapesFormat
    ShapesFormatV01
    ShapesFormatV02
```

### Points format

```{eval-rst}
.. autosummary::
    :toctree: generated

    CurrentPointsFormat
    PointsFormatV01
```

### Tables format

```{eval-rst}
.. autosummary::
    :toctree: generated

    CurrentTablesFormat
    TablesFormatV01
```

