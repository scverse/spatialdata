# Design document for `SpatialData`

This documents defines the specifications and design of SpatialData: an open and interoperable framework for storage and processing of multi-modal spatial omics data. This is meant to be a living document that can be updated as the project evolves.

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
- _SpatialData_ is based on standard on-disk storage formats (Zarr and Parquet) and on existing specifications (NGFF, AnnData) and uses existing solutions when possible. The resulting storage objects which brings together these technologies defines the _SpatialData on-disk format_, which is described in this document and finely characterized in [this online resource](https://github.com/scverse/spatialdata-notebooks/tree/main/notebooks/developers_resources/storage_format).

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

The _SpatialData_ library provides a set of specifications and in-memory representations for spatial omics datasets with the goal of unifying spatial omics pipelines and making raw and processed datasets interoperable with browser-based viewers. _SpatialData_ also provides basic operations to query and manipulate such data. The _SpatialData_ specs inherit the OME-NGFF specification for the storage of raster types (images and labels) and for storing several types of metadata. Additional storage requirements not covered by OME-NGFF are described in this document and in [this online resource](https://github.com/scverse/spatialdata-notebooks/tree/main/notebooks/developers_resources/storage_format). _SpatialData_ also implements Python objects to load, save, and interact with spatial data.

**Elements**

_Elements_ are the building blocks of _SpatialData_ datasets. Each element represents a particular datatype (e.g., raster image, label image, expression table). SpatialData elements are not special classes, but are instead standard scientific Python classes (e.g., `xarray.DataArray`, `AnnData`) with specified metadata. The metadata provides the necessary information for displaying and integrating the different elements (e.g., coordinate systems and coordinate transformations). Elements can either be initialized with valid metadata from disk or from in memory objects (e.g., `numpy` arrays) via _SpatialData_ parser functions. See the Elements section below for a detailed description of the different types of elements.

**`SpatialData`**

The `SpatialData` object contains a set of Elements to be used for analysis. Elements contained within a `SpatialData` object can be annotated by one or multiple Table elements. All Elements within a `SpatialData` object can be queried, selected, and saved via the `SpatialData` APIs.

### Elements

We model a spatial dataset as a composition of distinct elements, of any type. The elements correspond to:

- Pixel-based _Images_, 2D or 3D
- Regions of interest
    - _Shapes_ (circles, polygons, multipolygons), 2D
    - Pixel masks (such as segmentation masks), aka _Labels_, 2D, or 3D
- Points (such as transcript locations, point clouds, ...), 2D or 3D
- _Tables_ of annotations

Each of these elements should be useful by itself, and in combination with other relevant elements. All elements are stored in the Zarr container in hierarchy store that MAY be flat; currently Zarr hierarchies are not supported, [see here](https://github.com/scverse/spatialdata/issues/340)).

There is no explicit link between elements (e.g. we don't save information equivalent to "this Labels element refers to this Image element"), and one is encouranged to use coordinate systems to semantically group elements together, based on spatial overlap. Coordinate systems are explained later in this document.

By decomposing the data model into building blocks (i.e. Elements) we support the storage of any arbitrary combinations of elements, which can be added and modified independently at any moment.

#### Assumptions

_SpatialData_ follows the OME-NGFF specifications whenever possible and therefore much of its assumptions are inherited from it. Extra assumptions will be discussed with the OME-NGFF community and adapted to the community-agreed design. The key assumptions are the following:

- `Images`, `Labels`, `Points` and `Shapes` MUST have one or more _coordinate systems_ and _coordinate transformations_.
- `Tables` CAN NOT have a _coordinate system_ or _coordinate transforms_. Tables should not contain spatial coordinate: the user can decided to store them there, but they will not be processed by the library and needs to placed in a element and a coordinate system to be recognized by the framework.
- `Labels` and `Shapes` are both instances of `Regions`, `Regions` are `Elements`.
- Any `Element` MAY be annotated by `Tables`; also `Shapes` and `Points` MAY contain annotations within themselves as additional dataframe columns (e.g. intensity of point spread function of a each point, or gene id).
- `Tables` CAN NOT be annotated by other `Tables`.

#### Naming

Names of SpatialData elements must fulfill certain restrictions to ensure robust storage and compatibility:

- MUST NOT be the empty string ``.
- MUST only contain alphanumeric characters or hyphens `-`, dots `.`, underscores `_`. Alphanumeric includes letters from different alphabets and number-like symbols, but excludes whitespace, slashes and other symbols.
- MUST NOT be the full strings `.` or `..`, which would have special meanings as file paths.
- MUST NOT start with double underscores `__`.
- MUST NOT only differ in character case, to avoid name collisions on case-insensitive systems.

In tables, the above restrictions apply to the column names of `obs` and `var`, and to the key names of the `obsm`,
`obsp`, `varm`, `varp`, `uns`, `layers` slots (for example `adata.obs['has space']` and `adata.layers['.']` are not
allowed).
Additionally, dataframes in tables MUST NOT have a column named `_index`, which is reserved.

#### Images

Images of a sample. Should conform to the [OME-NGFF concept of an image](https://ngff.openmicroscopy.org/latest/#image-layout).
Images are n-dimensional arrays where each element of an array is a pixel of an image. These arrays have labelled dimensions which correspond to:

- Spatial dimensions (height and width).
- Imaging or feature channels.
- Z-stacks.

We require the following axes (in the following order):

- 2D images: cyx
- 3D images: czyx

Other ordering or axes neames are currently not supported.

- [ ] P2 We will support also time-point axes in the future. Furthermore, thanks to NGFF specs v0.5, such axes will not have name constraints (although they do for first iteration due to NGFF specs v0.4).

The image object itself builds on prior art in image analysis, in particular the [xarray library][].

Images have labeled dimensions, and coordinate transformations. These transformations are used to map between pixel space and physical space, and between different physical spaces.

For computational efficiency the images can use lazy loading, chunked storage, and can have a multiscale (aka pyramidal) format. Chunked storage and lazy loading is implemented via the [xarray library][] and [dask library][], multiscale representation uses [xarray datatree library][]. The coordinate system and transformations are stored in `xarray.DataArray.attrs`.

More precisely, we are using the [spatial-image library][] and [multiscale-spatial-image libary][] to have richer objects representations for the above-mentioned libraries.

The coordinate systems and transforms are stored in `spatial_image.SpatialImage.attrs` or in `multiscale_spatial_image.MultiscaleSpatialImage.attrs`.

The xarray coordinates are not saved in the NGFF storage. APIs to take into account for the xarray coordinates, such as converting back and forth between NGFF transformations and xarray coordinates, will be implemented ([see this issue](https://github.com/scverse/spatialdata/issues/308)). In particular, the xarray coordinates will be converted to NGFF transformations before saving the images to disk, and will be reconstructed after reading the data from disk. Supporing the representation of xarray coordinates will allow raster data types to be assigned a coordinate systems; otherwise (as of now) they must be defined in the "pixel space" (this is done implicitly).

<!-- Links -->

[dask library]: https://www.dask.org/
[xarray library]: https://docs.xarray.dev/en/stable/
[xarray datatree library]: https://github.com/xarray-contrib/datatree
[spatial-image library]: https://github.com/spatial-image/spatial-image
[multiscale-spatial-image libary]: https://github.com/spatial-image/multiscale-spatial-image

#### Regions of interest

Regions of interest define distinct regions of space that can be used to select and aggregate observations. For instance, regions can correspond to

- Tissues
- Tissue structures
- Clinical annotations
- Multi-cellular communities
- Cells
- Subcellular structures
- Physical structures from the assay (e.g. Visium "spots")
- Synthetic regions created by analysts (e.g. output of algorithms)

As an example, regions can be used for:

- subsetting observations (e.g., get all observations in a given region)
- aggregating observations (e.g., count all observations in an region)

Regions can be defined in multiple ways.

##### Labels (pixel mask)

Labels are a pixel mask representation of regions. This is an array of integers where each integer value corresponds to a region of space. This is commonly used along side pixel-based imaging techniques, where the label array will share dimensionality with the image array. These may also be hierarchical.
Should conform to the [OME-NGFF definition](https://ngff.openmicroscopy.org/latest/#image-layout).

The Python data structures used for Labels are the same one that we discussed for Images; same holds for the discussion around coordinate systems and xarray coordinates.

We require the following axes (in the following order):

- 2D labels: yx
- 3D labels: zyx

##### Shapes

A set of (multi-)polygons or points (circles) associated with a set of observations. Each set of polygons is associated with a coordinate system. Shapes can be used to represent a variety of regions of interests, such as clinical annotations and user-defined regions of interest. Shapes can also be used to represent most of array-based spatial omics technologies such as 10x Genomics Visium, BGI Stereo-seq and DBiT-seq.

The Shapes object is implemented as a geopandas dataframe with its associated [geopandas data structure](https://geopandas.org/en/stable/docs/user_guide/data_structures.html). The coordinate systems and transforms are stored in `geopandas.DataFrame.attrs`.
We are considering using the [dask-geopandas library](https://dask-geopandas.readthedocs.io/en/stable/), [discussion here](https://github.com/scverse/spatialdata/issues/122).

#### Points

_This representation is still under discussion and it might change.
What is described here is the current implementation. See discussion [here](https://github.com/scverse/spatialdata/issues/233) and [here](https://github.com/scverse/spatialdata/issues/46)._

Coordinates of points for single molecule data. Each observation is a point, and might have additional information (intensity etc.).
Current implementation represent points as a Parquet file and a [`dask.dataframe.DataFrame`](https://docs.dask.org/en/stable/dataframe.html) in memory.
The requirements are the following:

- The dataframe MUST contains axis name to represent the axes.
    - If it's 2D, the axes should be `["x","y"]`.
    - If it's 3D, the axes should be `["x","y","z"]`.
- It MUST also contain coordinate transformations in `dask.dataframe.DataFrame().attrs["transform"]`. This information will be saved on-disk in JSON under a `"coordinateTransformations"` key in a `.zattrs` file (Zarr v2), or `zarr.json` file (Zarr v3), in a Zarr Group containing the Element. The specific way this information is stored in disk is described by the NGFF specification (the implementaiton of the latest version of the specs is ongoing). Precisely, the NGFF specification dscribes coordinate transformations for images, but we will reuse it also for Points and Shapes.

Additional information is stored in `dask.dataframe.DataFrame().attrs["spatialdata_attrs"]` (or on-disk in `.attrs["spatialdata_attrs"]` for the Zarr Group containing the Element).

- It MAY also contains `"feature_key"`, that is, the column name of the dataframe that refers to the features. This `Series` MAY be of type `pandas.Categorical`.
- It MAY contains additional information in `dask.dataframe.DataFrame().attrs["spatialdata_attrs"]`, specifically:
    - `"instance_key"`: the column name of the dataframe where unique instance ids that this point refers to are stored, if available.

#### Table (table of annotations for regions)

Annotations of regions of interest. Each row in this table corresponds to a single region on the coordinate space. This is represented as an `AnnData` object to allow for complex annotations on the data. This includes:

- multivariate feature support, e.g. a matrix of dimensions regions x variables;
- annotations on top of the features or of the observations. E.g. calculated statistic, prior knowledge based annotations, cell types etc.
- graphs of observations or variables. These can be spatial graphs, nearest neighbor networks based on feature similarity, etc.

One region table can refer to multiple sets of Regions. But each row can map to only one region in its Regions element. For example, one region table can store annotation for multiple slides, though each slide would have its own label element.

    * `region: str | list[str]`: Regions or list of regions this table refers to.
    * `region_key: str`: Key in obs which says which Regions container
           this obs exists in (e.g. "library_id").
    * `instance_key: str`: Key in obs that says which instance the obs
           represents (e.g. "cell_id").

If any of `region`, `region_key` and `instance_key` are defined, they all MUST be defined. A table not defining them is still allowed, but it will not be mapped to any spatial element.

In `spatialdata-io` we use a consistent naming scheme for the `region_key` and `instance_key` column, which is suggested (but not required):

- we use the name `'region'` as the default name for the `region_key` column;
- we use the name `'instance_id'` as the default name for the `instance_key` column.

### Summary

- Image `type: Image`
- Regions `type: Labels | Shapes`
    - Labels `type: Labels`
    - Shapes `type: Shapes`
- Points `type: Points`
- Tables `type: Table`

#### Open discussions

- Points vs Circles [discussion](https://github.com/scverse/spatialdata/issues/46)

### Transforms and coordinate systems

In the following we refer to the NGFF proposal for transformations and coordinate systems.
You can find the [current transformations and coordinate systems specs proposal here](http://api.csswg.org/bikeshed/?url=https://raw.githubusercontent.com/bogovicj/ngff/coord-transforms/latest/index.bs), **# TODO update reference once proposal accepted**; [discussion on the proposal is here](https://github.com/ome/ngff/pull/138)).

The NGFF specifications introduces the concepts of coordinate systems and axes. Coordinate sytems are sets of axes that have a name, and where each axis is an object that has a name, a type and eventually a unit information. The set of operations required to transform elements between coordinate systems are stored as coordinate transformations. A table MUST not have a coordinate system since it annotates Region Elements (which already have one or more coordinate systems).

#### NGFF approach

There are two types of coordinate systems: intrinsic (called also implicit) and extrinsic (called also explicit). Intrinsic coordinate systems are tied to the data structure of the element and decribe it (for NGFF, an image without an intrinsic coordinate system would have no information on the axes). The intrinsic coordinate system of an image is the set of axes of the array containing the image. Extrinsic coordinate systems are not anchored to a specific element.

The NGFF specification only operates with images and labels, so it specifies rules for the coordinate systems only for these two types of elements. The main points are the following:

- each image/labels MUST have one and only one intrinsic coordinate system;
- each image/labels MAY have a transformation mapping them to one (at last one MUST be present) or more extrinsic coordinate systems;
- a transformation MAY be defined between any two coordinate systems, including intrinsic and extrinsic coordinate systems.

Furthermore, acoording to NGFF, a coordinate system:

- MUST have a name;
- MUST specify all the axes.

#### SpatialData approach

In SpatialData we extend the concept of coordinate systems also for the other types of spatial elements (Points, Shapes, Polygons).
Since elements are allowed to have only (a subset of the) c, x, y, z axes and must follow a specific schema, we can relax some restrictions of the NGFF coordinate systems and provide less verbose APIs. The framework still reads and writes to valid NGFF; converting to the SpatialData coordinate system if generally possible, and when not possible we raise an exception.

In details:

- we don't need to specify the intrinsic coordinate systems, these are inferred from the element schema
- each element MAY have a transformation mapping them to one or more extrinsic coordinate systems

Each extrinsic coordinate system

- MUST have a name
- MAY specify its axes

We also have a constraint (that we will relax in the future, [see here](https://github.com/scverse/spatialdata/issues/308)):

- a transformation MAY be defined only between an intrinsic coordinate system and an extrinsic coordinate system
- each element MUST be mapped at least to an extrinsic coordinate system. When no mapping is specified, we define a mapping to the "global" coordinate system via an "Identity" transformation.

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
If you need to implement a method not in Python please refer to [this online resource](https://github.com/scverse/spatialdata-notebooks/tree/main/notebooks/developers_resources/storage_format) for precise on-disk storage information.

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
