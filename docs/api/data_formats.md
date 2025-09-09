# Data formats (advanced)

The SpatialData format is defined as a set of versioned subclasses of `ome_zarr.format.Format`, one per type of element. The `spatialdata.SpatialDataFormatType` is a union type encompassing the possible valid formats.
These format subclasses are useful to ensure backward compatibility whenever a major version change is introduced. We also provide pointers to the latest format.

```{eval-rst}
.. currentmodule:: spatialdata._io.format

.. autoclass:: CurrentRasterFormat
.. autoclass:: RasterFormatV01
.. autoclass:: CurrentShapesFormat
.. autoclass:: ShapesFormatV01
.. autoclass:: ShapesFormatV02
.. autoclass:: CurrentPointsFormat
.. autoclass:: PointsFormatV01
.. autoclass:: CurrentTablesFormat
.. autoclass:: TablesFormatV01
```
