# Design document for `SpatialData`

This document describes the goals and the design of the `spatialdata` Python library, part of SpatialData: an open and interoperable framework for storage and processing of multi-modal spatial omics data. This is meant to be a living document that can be updated as the project evolves.

:::{note}
The SpatialData data model and storage format are defined by the [SpatialData specifications][spec], not by this document. Implementations in other languages should follow the specifications.
:::

## Motivation and Scope

Recent advances in molecular profiling technologies allow to measure abundance of RNA and proteins in tissue, at high throughput, multiplexing and resolution. The variety of experimental techniques poses unique challenges in data handling and processing, in particular around data types and size. _SpatialData_ aims at implementing a performant in-memory representation in Python and an on-disk representation based on the Zarr and Parquet data formats and following, when applicable, the OME-NGFF specification. By maximing interoperability, performant implementations and efficient (cloud-based) IO operations, _SpatialData_ aims at laying the foundations for new methods and pipelines for the analysis of spatial omics data.

### Goals

The goals define _what_ SpatialData will be able to do (as opposed to _how_). Goals can have the following priority levels:

- P0: highest priority, required for successful implementation (i.e., must have)
- P1: high priority, but not required (i.e., nice to have)
- P2: nice to have, but not a priority

**1. Load data from modern spatial multiomics experiments**

- P0. Data can be loaded from the OME-NGFF and saved to OME-NGFF.
    - [x] multiscale images and labels, 2d and 3d
    - [x] point clouds
    - [x] polygon-shaped regions of interest
    - [x] circle-shaped regions of interest
    - [x] tables
    - [x] graphs
- P0. Data can be loaded lazily.
    - [x] Images
    - [x] Points
    - [ ] (P1) Shapes https://github.com/scverse/spatialdata/issues/359
- P1.
    - [x] Loaded data can be iterated over to generate tiles for multiprocessing and deep learning.

**2. Align different datasets via affine transformations**

- [x] P0. Transformations can be loaded from and written to OME-NGFF.
- [x] P0. Identity transformation
- [x] P0. Affine transformations.
    - [x] scale
    - [x] translation
    - [x] rotation
- [x] P0. Support definition of common coordinate systems across datasets (i.e., extrinsic coordinate systems).
- [x] P0. Sequence of transformation.
- Utils
    - [x] P0 permute axis
- [ ] P2. non-linear
    - [ ] coordinates and displacements

**3. Performant spatial query of multimodal spatial datasets**

- [x] P0. Support querying a multimodal dataset for all data in a specified region (at the cost of creating spatial index every time).
    - [x] Arbitrary bounding boxes
    - [x] Polygons or regions of interest (ball, shape)

**4. Aggregate observations by regions of interest**

- [x] P0. Support aggregation functions with standard summary statistics
    - [x] mean
    - [x] sum
    - [x] count
- [x] P1. User-defined aggregation function

### Non-goals

- _SpatialData_ is not an analysis library. Instead the aim is to provide an infrastructure to analysis libraries for IO and spatial queries.
- _SpatialData_ is not a format converter. We should not support converting to/from too many formats and instead use OME-NGFF as the interchange format. Nevertheless,[spatialdata-io][] offers a place for some common data conversions (external contributions are highly encouraged).
- _SpatialData_ is based on standard on-disk storage formats (Zarr and Parquet) and on existing specifications (NGFF, AnnData) and uses existing solutions when possible. The resulting storage objects which bring together these technologies form the _SpatialData storage format_, which is specified in the [SpatialData storage format specification][spec-storage-format].

## Satellite projects

We strongly encourage collaborations and community supports in all of these projects.

- [x] P0. _Visualization_: we are developing a napari plugin for interactive visualization of _SpatialData_ objects @ [napari-spatialdata][].
- [x] P0. _Raw data IO_: we are implementing readers for raw data of common spatial omics technologies @ [spatialdata-io][].
- [x] P1. _Static plotting_: a static plotting library for _SpatialData_ @ [spatialdata-plot][].
- [ ] P2. _Image analysis_: Library to perform image analysis, wrapping common analysis library in python such as skimage.
      Once ready, we will deprecate such functionalities in [squidpy][].
- [ ] P2. _Spatial and graph analysis_: [squidpy][] will be refactor to accept SpatialData objects as input.
- [ ] P2. _Database_: Some form of update on released datasets with updated specs as development progresses. A temporary sandbox where we store downloader and converter scripts for representative datasets is available @ [spatialdata-sandbox][].

<!-- Links -->

[napari-spatialdata]: https://github.com/scverse/napari-spatialdata
[spatialdata-io]: https://github.com/scverse/spatialdata-io
[spatialdata-plot]: https://github.com/scverse/spatialdata-plot
[squidpy]: https://github.com/scverse/squidpy
[spatialdata-sandbox]: https://github.com/giovp/spatialdata-sandbox

## Detailed description

### Terminology

_SpatialData_ is both the name of the Python library as well as the name of the framework (including [spatialdata-io][], [napari-spatialdata][], [spatialdata-plot][]) and the name of the in-memory Python object `SpatialData`. To distinguish between the three, we will use the _italics_ formatting for the _SpatialData_ library and the _SpatialData_ framework (the distinction between them will be clear from the context), and we will use the `code` formatting for the `SpatialData` object.

**_SpatialData_**

The _SpatialData_ library implements the SpatialData data model and storage format in Python. It provides in-memory representations for spatial omics datasets, with the goal of unifying spatial omics pipelines and making raw and processed datasets interoperable with browser-based viewers, and basic operations to query and manipulate such data. The data model and the storage format are defined, independently of any implementation, by the [SpatialData specifications][spec]; the storage format builds on OME-NGFF for raster types (images and labels) and for several types of metadata.

**Elements**

_Elements_ are the building blocks of _SpatialData_ datasets. Each element represents a particular datatype (e.g., raster image, label image, expression table). SpatialData elements are not special classes, but are instead standard scientific Python classes (e.g., `xarray.DataArray`, `AnnData`) with specified metadata. The metadata provides the necessary information for displaying and integrating the different elements (e.g., coordinate systems and coordinate transformations). Elements can either be initialized with valid metadata from disk or from in memory objects (e.g., `numpy` arrays) via _SpatialData_ parser functions.

**`SpatialData`**

The `SpatialData` object contains a set of Elements to be used for analysis. Elements contained within a `SpatialData` object can be annotated by one or multiple Table elements. All Elements within a `SpatialData` object can be queried, selected, and saved via the `SpatialData` APIs.

### Elements

:::{important}
The element types, their constraints, the naming rules and the on-disk layout are defined by the [SpatialData data model][spec-data-model] and the [SpatialData storage format][spec-storage-format] specifications. Where this document and the specifications disagree, the specifications are authoritative. This section only describes how the Python library represents the elements in memory and where it is more restrictive than the specifications.
:::

A spatial dataset is a composition of distinct elements: _Images_, _Labels_, _Points_, _Shapes_ and _Tables_. There is no explicit link between spatial elements; they are related spatially through [coordinate systems](#transforms-and-coordinate-systems) and by annotation through tables. By decomposing the data model into building blocks we support the storage of any arbitrary combinations of elements, which can be added and modified independently at any moment.

| Element | Specification                                              | In-memory representation in `spatialdata`                       |
| ------- | ---------------------------------------------------------- | --------------------------------------------------------------- |
| Images  | [data model][spec-images], [storage][storage-images]       | `xarray.DataArray`, or `xarray.DataTree` for multiscale images  |
| Labels  | [data model][spec-labels], [storage][storage-labels]       | `xarray.DataArray`, or `xarray.DataTree` for multiscale labels  |
| Points  | [data model][spec-points], [storage][storage-points]       | `dask.dataframe.DataFrame`                                      |
| Shapes  | [data model][spec-shapes], [storage][storage-shapes]       | `geopandas.GeoDataFrame`                                        |
| Tables  | [data model][spec-tables], [storage][storage-tables]       | `anndata.AnnData`                                               |

Element names, and the column and key names inside tables, follow the [naming rules of the storage format][spec-naming].

The coordinate transformations of a spatial element are stored in its `.attrs["transform"]`; the additional metadata defined by the specifications (e.g. `feature_key` and `instance_key` for points) is stored in `.attrs["spatialdata_attrs"]`, and for tables in `table.uns["spatialdata_attrs"]`. The models in `spatialdata.models` validate and parse these objects; see the [API documentation](api/models.md).

#### Implementation notes

- **Images and labels.** Chunked storage and lazy loading are implemented via [xarray][xarray library] and [dask][dask library]. The library currently supports only the axes `cyx` and `czyx` for images and `yx` and `zyx` for labels, in this order; time axes are not yet supported. The xarray coordinates are not saved to disk: raster elements are always defined in their pixel space and are mapped to other coordinate systems via transformations ([see this issue](https://github.com/scverse/spatialdata/issues/308)).
- **Shapes.** Lazy loading of shapes is not yet supported ([see this issue](https://github.com/scverse/spatialdata/issues/359)); we are considering the [dask-geopandas library](https://dask-geopandas.readthedocs.io/en/stable/) ([discussion here](https://github.com/scverse/spatialdata/issues/122)).
- **Points.** The representation of points is still under discussion ([here](https://github.com/scverse/spatialdata/issues/233) and [here](https://github.com/scverse/spatialdata/issues/46)).
- **Tables.** Tables do not have coordinate systems. Spatial coordinates stored in a table (e.g. in `obsm["spatial"]`) are currently not processed by the library; to be recognized by the framework they need to be placed in a spatial element.
- **Default column names.** [spatialdata-io][] uses `"region"` as the name of the `region_key` column and `"instance_id"` as the name of the `instance_key` column. These are conventions, not requirements.

<!-- Links -->

[dask library]: https://www.dask.org/
[xarray library]: https://docs.xarray.dev/en/stable/

### Transforms and coordinate systems

Coordinate systems and coordinate transformations, and how they are stored, are defined by the [data model][spec-coordinate-systems] and the [storage format][storage-coordinate-systems] specifications, which build on the coordinate transformations proposed for OME-NGFF ([discussion](https://github.com/ome/ngff/pull/138)).

The library currently has some restrictions compared to the specifications ([see this issue](https://github.com/scverse/spatialdata/issues/308)):

- the intrinsic coordinate system of an element is not stored explicitly but inferred from the element schema;
- a transformation can only be defined between the intrinsic coordinate system of an element and an extrinsic coordinate system;
- each element is mapped to at least one extrinsic coordinate system. When no transformation is specified, the parsers add an `Identity` transformation to the `"global"` coordinate system.

#### In-memory representation

We define classes that follow the NGFF specifications to represent the coordinate systems (class `NgffCoordinateSystem`) and coordinate transformations (classes inheriting from `NgffBaseTransformations`). Anyway, these classes are used only during input and output. For operations we define new classes (inheriting from `BaseTransformation`).

Classes inheriting from `NgffBaseTransformation` are: `NgffIdentity`, `NgffMapAxis`, `NgffTranslation`, `NgffScale`, `NgffAffine`, `NgffRotation`, `NgffSequence`, `NgffByDimension`. The following are not supported: `NgffMapIndex`, `NgffDisplacements`, `NgffCoordinates`, `NgffInverseOf`, `NgffBijection`. In the future these classes could be moved outside _SpatialData_, for instance in [ome-zarr-py](https://github.com/ome/ome-zarr-py).

Classes inheriting from `BaseTransformation` are: `Identity`, `MapAxis`, `Translation`, `Scale`, `Affine`, `Sequence`.

The conversion between the two transformations is still not 100% supported; it will be finalized when the NGFF specifications are approved; [this issue](https://github.com/scverse/spatialdata/issues/114) keeps track of this.

#### Reasons for having two sets of classes

The `NgffBaseTransformations` require full specification of the input and output coordinate system for each transformation. A transformation MUST be compatible with the input coordinate system and output coordinate system (full description in the NGFF specification) and two transformations can be chained together only if the output coordinate system of the first coincides with the input coordinate system of the second.

On the contrary, each `BaseTransformation` is self-defined and does not require the information on coordinate systems. Almost (see below) any transformation can be applied unambiguously to any element and almost any pair of transformations can be chained together. The result is either uniquely defined, either an exception is raised when there is ambiguity.

Precisely, this is performed by "passing through" (keeping unaltered) those axis that are present in an element but not in a transformation, and by ignoring axes that are present in a transformation but not in an element.

For example one can apply a `Scale([2, 3, 4], axes=('x', 'y', 'z'))` to a `cyx` image (the axes `c` is passed through unaltered, and the scaling on `z` is ignored since there is no `z` axis.)

An example of transformation that cannot be applied is an `Affine xy -> xyz` to `xyz` data, since `z` can't be passed through as it is also the output of the transformation.

To know more about the separation between the two set of classes see [this closed issue](https://github.com/scverse/spatialdata/issues/39), [this other closed issue](https://github.com/scverse/spatialdata/issues/47) and [this merged pr](https://github.com/scverse/spatialdata/pull/100).

After the [NGFF transformations specification](http://api.csswg.org/bikeshed/?url=https://raw.githubusercontent.com/bogovicj/ngff/coord-transforms/latest/index.bs) is released, we will work on reaching 100% compliance to it. Until then there may be some small difference between the proposed NGFF transformations storage and the SpatialData on-disk storage.
If you need to read or write SpatialData outside of Python, refer to the [SpatialData storage format specification][spec-storage-format].

### Examples

#### Transformations

See [this notebook](https://github.com/scverse/spatialdata-notebooks/blob/main/notebooks/transformations.ipynb) for extensive examples on the transformations.

### Roadmap

This section describes a more detailed timeline of future developments, including also more technical tasks like code refactoring of existing functionalities for improving stability/performance. Compared to the "goal" section above, here we provide an concrete timeline.

#### 2024

- [x] Simplify data models
    - [x] Use `xarray.DataArray` instead of the subclass `SpatialImage` and `xarray.DataTree` instad of the subclass `MultiscaleSpatialImage`
- [x] More performant disk storage
    - [x] Use `geoparquet` for shapes and points
- [x] Start working on multiple tables
- [x] Start working on the transformations refactoring
- [x] Finalize multiple tables support

#### 2025

- [ ] Move transformation code [to `ome-zarr-models-py`](https://github.com/ome-zarr-models/ome-zarr-models-py/issues/54)
- [ ] Refactor `spatialdata` to use `ome-zarr-models-py` as a dependency.
- [ ] Expose public APIs for [modular `read()` operations](https://github.com/scverse/spatialdata/issues/843).
- [ ] Support Zarr v3 (sharding).
- [ ] Remove Dask constraints ([latest `dask-expr` versions are not compatible](https://github.com/dask/dask/issues/11146)).

---

### Examples

Real world examples will be available as notebooks [in this repository](https://github.com/scverse/spatialdata-notebooks), furthermore, some draft [implementations are available here](https://github.com/giovp/spatialdata-sandbox).

#### Related notes/issues/PRs

- [Issue discussing SpatialData layout](https://github.com/scverse/spatialdata/issues/12)
- [Notes from Basel Hackathon](https://hackmd.io/MPeMr2mbSRmeIzOCgwKbxw)

<!-- Links to the SpatialData specifications -->

[spec]: https://spatialdata-spec.scverse.org/
[spec-data-model]: https://spatialdata-spec.scverse.org/data-model.html
[spec-storage-format]: https://spatialdata-spec.scverse.org/storage-format.html
[spec-images]: https://spatialdata-spec.scverse.org/data-model.html#spec-images
[spec-labels]: https://spatialdata-spec.scverse.org/data-model.html#spec-labels
[spec-points]: https://spatialdata-spec.scverse.org/data-model.html#spec-points
[spec-shapes]: https://spatialdata-spec.scverse.org/data-model.html#spec-shapes
[spec-tables]: https://spatialdata-spec.scverse.org/data-model.html#spec-tables
[spec-coordinate-systems]: https://spatialdata-spec.scverse.org/data-model.html#spec-coordinate-systems
[spec-naming]: https://spatialdata-spec.scverse.org/storage-format.html#storage-datasets-element-groups-and-names
[storage-images]: https://spatialdata-spec.scverse.org/storage-format.html#storage-images
[storage-labels]: https://spatialdata-spec.scverse.org/storage-format.html#storage-labels
[storage-points]: https://spatialdata-spec.scverse.org/storage-format.html#storage-points
[storage-shapes]: https://spatialdata-spec.scverse.org/storage-format.html#storage-shapes
[storage-tables]: https://spatialdata-spec.scverse.org/storage-format.html#storage-tables
[storage-coordinate-systems]: https://spatialdata-spec.scverse.org/storage-format.html#storage-coordinate-systems
