# Data formats (advanced)

The SpatialData format is defined as a set of versioned subclasses of `spatialdata._io.format.SpatialDataFormat`, one per type of element.
These classes are useful to ensure backward compatibility whenever a major version change is introduced. We also provide pointers to the latest format.

```{eval-rst}
.. currentmodule:: spatialdata._io.format

.. autoclass:: CurrentRasterFormat
   :members:
   :undoc-members:
.. autoclass:: RasterFormatV01
   :members:
   :undoc-members:
.. autoclass:: CurrentShapesFormat
   :members:
   :undoc-members:
.. autoclass:: ShapesFormatV01
   :members:
   :undoc-members:
.. autoclass:: ShapesFormatV02
   :members:
   :undoc-members:
.. autoclass:: CurrentPointsFormat
   :members:
   :undoc-members:
.. autoclass:: PointsFormatV01
   :members:
   :undoc-members:
.. autoclass:: CurrentTablesFormat
   :members:
   :undoc-members:
.. autoclass:: TablesFormatV01
   :members:
   :undoc-members:
```
