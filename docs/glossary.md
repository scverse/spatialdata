# Glossary

## IO

IO means input/output. For instance, `spatialdata-io` is about reading and/or writing data.

## NGFF

[NGFF (Next-Generation File Format)](https://ngff.openmicroscopy.org/latest/) is a specification for storing multi-dimensional, large-scale imaging and spatial data efficiently. Developed by the OME (Open Microscopy Environment), it supports formats like multi-resolution images, enabling flexible data access, high performance, and compatibility with cloud-based storage. NGFF is designed to handle modern microscopy, biomedical imaging, and other high-dimensional datasets in a scalable, standardized way, often using formats like Zarr for storage.

## OME

OME stands for Open Microscopy Environment and is a consortium of universities, research labs, industry and developers producing open-source software and format standards for microscopy data. It developed, among others, the OME-NGFF specification.

## OME-NGFF

See NGFF.

## OME-Zarr

An implementation of the OME-NGFF specification using the Zarr format. The SpatialData Zarr format (see below) is an extnesion of the OME-Zarr format.

## Raster / Rasterization

Raster data represents images using a grid of pixels, where each pixel contains a specific value representing information such as color, intensity, or another attribute.

Rasterization is the process of converting data from a vector format (see definition below) into a raster format of pixels (i.e., an image).

## ROI

An ROI (_Region of Interest_) is a specific subset of a dataset that highlights an area of particular relevance, such as a niche, cell, or tissue location. For example, an ROI may be defined as a polygon outlining a lymphoid structure.

## Spatial query

A spatial query subsets the data based on the location of the spatial elements. For example, subsetting data with a bounding box query selects elements, and the corresponding tabular annotations, within a defined rectangular region, while a polygon query selects data within a specified shape.

## SpatialData Zarr format

An extension of the OME-Zarr format used to represent SpatialData objects on disk. Our aim is to converge the SpatialData Zarr format with the OME-Zarr format, by adapting to future versions of the NGFF specification and by contributing to the development of new features in NGFF. The OME-Zarr format was developed for bioimaging data and therefore and extension of the OME-Zarr format is necessary to accommodate spatial omics data.

## SpatialElements

_SpatialElements_ are the building blocks of _SpatialData_ datasets, and are split into multiple categories: `images`, `labels`, `shapes` and `points`. SpatialData SpatialElements are not special classes, but are instead standard scientific Python classes (e.g., `xarray.DataArray`, `geopandas.GeoDataFrame`) with specified metadata. The metadata provides the necessary information for displaying and integrating the different elements (e.g., coordinate systems and coordinate transformations). SpatialElements can be annotated by `tables`, which are represented as `anndata.AnnData` objects. Tables are a building block of SpatialData, but are not considered SpatialElements since they do not contain spatial information.

## Tables

A building block of SpatialData, represented as `anndata.AnnData` objeets. Tables are used to store tabular data, such as gene expression values, cell metadata, or other annotations. Tables can be associated with SpatialElements to provide additional information about the spatial data, such as cell type annotations or gene expression levels.

## Vector / Vectorization

Vector data represents spatial information using geometric shapes such as points, circles, or polygons, each defined by mathematical coordinates.

Vectorization is the process of converting raster data (i.e. pixel-based images) into vector format. For instance, a "raster" cell boundary can be "vectorized" into a polygon represented by the coordinates of the vertices of the polygon.

## Zarr storage

Zarr is a format for storing multi-dimensional arrays, designed for efficient, scalable access to large datasets. It supports chunking (splitting data into smaller, manageable pieces) and compression, enabling fast reading and writing of data, even for very large arrays. Zarr is particularly useful for distributed and parallel computing, allowing access to subsets of data without loading the entire dataset into memory. A `SpatialData` object can be stored as a Zarr store (`.zarr` directory).
