# Glossary

### Spatial Element

_Spatial Elements_ are the building blocks of _SpatialData_ datasets, and are split into multiple categories: `images`, `labels`, `shapes`, `points`, and `tables`. SpatialData elements are not special classes, but are instead standard scientific Python classes (e.g., `xarray.DataArray`, `AnnData`) with specified metadata. The metadata provides the necessary information for displaying and integrating the different elements (e.g., coordinate systems and coordinate transformations).

### Raster / Rasterization

Raster data represents images using a grid of pixels, where each pixel contains a specific value representing information such as color, intensity, or another attribute.

Rasterization is the process of converting data from a vector format (see definition below) into a raster format of pixels (i.e., an image).

### Vector / Vectorization

Vector data represents spatial information using geometric shapes such as points, circles, or polygons, each defined by mathematical coordinates.

Vectorization is the process of converting raster data (i.e. pixel-based images) into vector format. For instance, a "raster" cell boundary can be "vectorized" into a polygon represented by the coordinates of the vertices of the polygon.

### ROI

An ROI (_Region of Interest_) is a specific subset of a dataset that highlights an area of particular relevance, such as a niche, cell, or tissue location. For example, an ROI may be defined as a polygon outlining a lymphoid structure.

### IO

IO means input/output. For instance, `spatialdata-io` is about reading and/or writing data.

### Spatial query

A spatial query subsets the data based on the location of the spatial elements. For example, subsetting data with a bounding box query selects elements within a defined rectangular region, while a polygon query selects data within a specified shape.

### NGFF

[NGFF (Next-Generation File Format)](https://ngff.openmicroscopy.org/latest/) is a specification for storing multi-dimensional, large-scale imaging and spatial data efficiently. Developed by the OME (Open Microscopy Environment), it supports formats like multi-resolution images, enabling flexible data access, high performance, and compatibility with cloud-based storage. NGFF is designed to handle modern microscopy, biomedical imaging, and other high-dimensional datasets in a scalable, standardized way, often using formats like Zarr for storage.

### Zarr storage

Zarr is a format for storing multi-dimensional arrays, designed for efficient, scalable access to large datasets. It supports chunking (splitting data into smaller, manageable pieces) and compression, enabling fast reading and writing of data, even for very large arrays. Zarr is particularly useful for distributed and parallel computing, allowing access to subsets of data without loading the entire dataset into memory. A `SpatialData` object can be stored as a Zarr store (`.zarr` directory).
