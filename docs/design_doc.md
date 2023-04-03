# Design document for `SpatialData`

This documents defines the specifications and design of SpatialData: a FAIR framework for storage and processing of multi-modal spatial omics data. It also describes the initial implementation plan. This is meant to be a living document that can be updated as the project evolves.

## Motivation and Scope

Recent advances in molecular profiling technologies allow to measure abundance of RNA and proteins in tissue, at high throughput, multiplexing and resolution. The variety of experimental techniques poses unique challenges in data handling and processing, in particular around data types and size. _SpatialData_ aims at implementing a performant in-memory representation in Python and an on-disk representation based on the Zarr data format and following the OME-NGFF specifications. By maximing interoperability, performant implementations and efficient (cloud-based) IO, _SpatialData_ aims at laying the foundations for new methods and pipelines for the analysis of spatial omics data.

### Goals

The goals define _what_ SpatialData will be able to do (as opposed to _how_). Goals can have the following priority levels:

-   P0: highest priority, required for successful implementation (i.e., must have)
-   P1: high priority, but not required (i.e., nice to have)
-   P2: nice to have, but not a priority

**1. Load data from modern spatial multiomics experiments**

-   P0. Data can be loaded from the OME-NGFF and saved to OME-NGFF.
    -   [x] multiscale images and labels, 2d and 3d
    -   [x] point clouds
    -   [x] polygon-shaped regions of interest
    -   [x] circle-shaped regions of interest
    -   [x] tables
    -   [x] graphs
-   P0. Data can be loaded lazily.
    -   [x] Images
    -   [x] Points
    -   [ ] Shapes
-   P1. [ ] Meshes can be loaded and saved in OME-NGFF
-   P1. [x] Loaded data can be iterated over to generate tiles for multiprocessing and deep learning.

**2. Align different datasets via affine transformations**

-   [x] P0. Transformations can be loaded from and written to OME-NGFF.
-   [x] P0. Identity transformation
-   [x] P0. Affine transformations.
    -   [x] scale
    -   [x] translation
    -   [x] rotation
-   [x] P0. Support definition of common coordinate systems across datasets (i.e., extrinsic coordinate systems).
-   [x] P0. Sequence of transformation.
-   Utils
    -   [x] P0 permute axis
    -   [ ] P1 inverse of
-   [ ] P1. non-linear
    -   [ ] inverseof
    -   [ ] coordinates and displacements
    -   [ ] bijections (manually specified forward and inverse)

**3. Performant spatial query of multimodal spatial datasets**

-   [x] P0. Support querying a multimodal dataset for all data in a specified region (at the cost of creating spatial index every time).
    -   [x] Arbitrary bounding boxes
    -   [x] Polygons or regions of interest (ball, shape)
-   [ ] P0. Subsetting and slicing objet with indexing
-   [ ] P1. Sparse data (e.g., point clouds) can be queried without loading all of the data.
-   [ ] P2. Any spatial index use can be read/written to/from disk. In general spatial index will have very rough implementation from the start.

**4. Aggregate observations by regions of interest**

-   [ ] P0. Support aggregation functions with standard summary statistics
    -   [ ] min
    -   [ ] median
    -   [ ] max
    -   [ ] mean
    -   [ ] standard deviation
-   [ ] P1. User-defined aggregation function
-   [ ] P1. Aggregation is parallelized

**5. Contribute on-disk specification for key data types back to OME-NGFF**

-   [x] P0. Table specification
-   [ ] P1. Points specification
-   [ ] P1. Polygons specification

### Non-goals

-   _SpatialData_ is not an analysis library. Instead, analysis libraries should depend on SpatialData for IO and query.
-   _SpatialData_ is not a format converter. We should not support converting to/from too many formats and instead use OME-NGFF as the interchange format.
-   _SpatialData_ is not a new format. Instead, _SpatialData_ builds upon the OME-NGFF. It is anyway inevitable that until the OME-NGFF format reviews all the new capabilities (e.g. transformations, tables, ...), we need to make further assumption on the data, that we will gradually relax to align in full to the NGFF specs.

## Satellite projects

We strongly encourage collaborations and community supports in all of these projects.

-   [ ] P0. _Visualization_: we are developing a napari plugin for interactive visualization of _SpatialData_ objects @ [napari-spatialdata][].
-   [x] P0. _Raw data IO_: we are implementing readers for raw data of common spatial omics technologies @ [spatialdata-io][].
-   [ ] P1. _Static plotting_: a static plotting library for _SpatialData_ @ [spatialdata-plot][].
-   [ ] P1. _Image analysis_: Library to perform image analysis, wrapping common analysis library in python such as skimage.
        Once ready, we will deprecate such functionalities in [squidpy][].
-   [ ] P1. _Spatial and graph analysis_: [squidpy][] will be refactor to accept SpatialData objects as input.
-   [ ] P2. _Database_: Some form of update on released datasets with updated specs as development progresses. A temporary sandbox where we store downloader and converter scripts for representative datasets is available @ [spatialdata-sandbox][].

<!-- Links -->

[napari-spatialdata]: https://github.com/scverse/napari-spatialdata
[spatialdata-io]: https://github.com/scverse/spatialdata-io
[spatialdata-plot]: https://github.com/scverse/spatialdata-plot
[squidpy]: https://github.com/scverse/squidpy
[spatialdata-sandbox]: https://github.com/giovp/spatialdata-sandbox

## Detailed description

### Terminology

_SpatialData_ is both the name of the Python library as well as of the in-memory python object `SpatialData`. To distinguish between the two, we use the _italics_ formatting for the _SpatialData_ library and the `code` formatting for the `SpatialData` object.

**_SpatialData_**

The _SpatialData_ library provides a set of specifications and in-memory representations for spatial omics datasets with the goal of unifying spatial omics pipelines and making raw and processed datasets interoperable with browser-based viewers. _SpatialData_ also provides basic operations to query and manipulate such data. The _SpatialData_ specs inherit the OME-NGFF specs in full, yet for the start it defines additional specifications in particular around data types and metadata. _SpatialData_ also implements Python objects to load, save, and interact with spatial data.

**Elements**

Elements are the building blocks of _SpatialData_ datasets. Each element represents a particular datatype (e.g., raster image, label image, expression table). SpatialData elements are not special classes, but are instead standard scientific Python classes (e.g., `xarray.DataArray`, `AnnData`) with specified metadata. The metadata provides the necessary information for displaying and integrating the different elements (e.g., coordinate systems and coordinate transformations). Elements can either be initialized with valid metadata from disk or from in memory objects (e.g., `numpy` arrays) via _SpatialData_ parser functions. See the Elements section below for a detailed description of the different types of elements.

**`SpatialData`**

The `SpatialData` object contains a set of Elements to be used for analysis. Elements contained within a `SpatialData` object can be annotated by a single Region Table. Future work may extend `SpatialData` to allow multiple tables (see discussion [here](https://github.com/scverse/spatialdata/issues/43)). All Elements within a `SpatialData` object can be queried, selected, and saved via the `SpatialData` objects.

-   [ ] P2. **`NGFFStore`**

    The `NGFFStore` is an object representing the on-disk layout of a dataset. The `NGFFStore` parses the files to determine what data are available to be loaded.
    Initial implementations will target a single Zarr file on disk, but future implementations may support reading from a collection of files. A `SpatialData` object can be instantiated from a `NGFFStore`.
    The implementation of the `NGFFStore` has low priority since we can get around it by loading and saving our data in multiple `.zarr` files and combine them in memory, but it will be important for reading arbitrary `.zarr` files when the NGFF specifications will be updated to support the various elements.

### Elements

We model a spatial dataset as a composition of distinct element types. The elements correspond to:

-   Pixel-based images, 2D or 3D
-   Regions of interest
    -   Shapes (circles, squares), 2D or 3D
    -   Polygons, 2D
    -   Pixel masks (such as segmentation masks), aka Labels, 2D, or 3D
-   Points (such as transcript locations, point clouds, ...), 2D or 3D
-   Tables of annotations (initially, these are annotations on the regions of interest, not annotations for points)

Each of these elements should be useful by itself, and in combination with other relevant elements. All elements are stored in the Zarr container in hierarchy store that MAY be flat. When the `NGFFStore` is implemented, `SpatialData` will nonetheless be able to read arbitrary hierarchies and make sense of them based on Zarr groups metadata and coordinate systems.

By decomposing the data model into building blocks (i.e. Elements) we support the storage of any arbitrary combinations of elements, which can be added and modified independently at any moment.

#### Assumptions

_SpatialData_ closely follows the OME-NGFF specifications and therefore much of its assumptions are inherited from it. Extra assumptions will be discussed with the OME-NGFF community and adapted to the community-agreed design. The key assumptions are the following:

-   `Images`, `Labels`, `Points` and `Shapes` MUST have one or more _coordinate systems_ and _coordinate transformations_.
-   `Tables` CAN NOT have a _coordinate system_ or _coordinate transforms_.
-   `Labels` and `Shapes` are both instances of `Regions`.
-   `Regions` are `Elements` and they MAY be annotated by `Tables`.
-   `Points` CAN NOT be annotated with `Tables`, but they can contain annotations within themselves (e.g. intensity of point spread function of a each point, gene id).
-   `Tables` CAN NOT be annotated by other `Tables`.

#### Images

Images of a sample. Should conform to the [OME-NGFF concept of an image](https://ngff.openmicroscopy.org/latest/#image-layout).
Images are n-dimensional arrays where each element of an array is a pixel of an image. These arrays have labelled dimensions which correspond to:

-   Spatial dimensions (height and width).
-   Imaging or feature channels.
-   Z-stacks.

-   [ ] P2 We will support also time-point axes in the future. Furthermore, due to NGFF specs v0.5, such axes will not have name constraints (although they do for first iteration due to NGFF specs v0.4).

The image object itself builds on prior art in image analysis, in particular the [xarray library][].

Images have labeled dimensions, and coordinate transformations. These transformations are used to map between pixel space and physical space, and between different physical spaces.

For computational efficiency the images can use lazy loading, chunked storage, and can have a pyramidal or multiscale format. Chunked storage and lazy loading is implemented via the [xarray library][] and [dask library][], multiscale representation uses [xarray datatree library][]. The coordinate system and transformations are stored in `xarray.DataArray.attrs`.

More precisely, we are using the [spatial-image library][] and [multiscale-spatial-image libary][] to have richer objects representations for the above-mentioned libraries.

The coordinate systems and transforms are stored in `spatial_image.SpatialImage.attrs` or in `multiscale_spatial_image.MultiscaleSpatialImage.attrs`.

<!-- Links -->

[dask library]: https://www.dask.org/
[xarray library]: https://docs.xarray.dev/en/stable/
[xarray datatree library]: https://github.com/xarray-contrib/datatree
[spatial-image library]: https://github.com/spatial-image/spatial-image
[multiscale-spatial-image libary]: https://github.com/spatial-image/multiscale-spatial-image

#### Regions of interest

Regions of interest define distinct regions of space that can be used to select and aggregate observations. For instance, regions can correspond to

-   Tissues
-   Tissue structures
-   Clinical annotations
-   Multi-cellular communities
-   Cells
-   Subcellular structures
-   Physical structures from the assay (e.g. Visium "spots")
-   Synthetic regions created by analysts (e.g. output of algorithms)

As an example, regions can be used for:

-   subsetting observations (e.g., get all observations in a given region)
-   aggregating observations (e.g., count all observations in an region)

Regions can be defined in multiple ways.

##### Labels (pixel mask)

Labels are a pixel mask representation of regions. This is an array of integers where each integer value corresponds to a region of space. This is commonly used along side pixel based imaging techniques, where the label array will share dimensionality with the image array. These may also be hierarchichal.
Should conform to the [OME-NGFF definition](https://ngff.openmicroscopy.org/latest/#image-layout).

The Python data structures used for Labels are the same one that we discussed for Images.

The coordinate systems and transforms are stored in `spatial_image.SpatialImage.attrs` or in `multiscale_spatial_image.MultiscaleSpatialImage.attrs`.

##### Shapes

A set of (multi-)polygons or points (circles) associated with a set of observations. Each set of polygons is associated with a coordinate system. Shapes can be used to represent a variety of regions of interests, such as clinical annotations and user-defined regions of interest. Shapes can also be used to represent most of array-based spatial omics technologies such as 10X Genomics Visium, BGI Stereo-seq and DBiT-seq.

The Shapes object is implemented as a geopandas dataframe with its associated [geopandas data structure](https://geopandas.org/en/stable/docs/user_guide/data_structures.html). The coordinate systems and transforms are stored in `geopandas.DataFrame.attrs`.
We are considering using the [dask-geopandas library](https://dask-geopandas.readthedocs.io/en/stable/), [discussion here](https://github.com/scverse/spatialdata/issues/122).

**Warning**

In the case where both a `Labels` image and its centroids coordinates are present, the centroids are stored as type of annotation.
Therefore, there is no `Shapes` element and the centroids coordinates can still be stored in `obsm["spatial"]`
of slot of the `Table`, yet no coordinates system is defined for them.
The assumption is that the coordinate system of the centroids corresponds to the implicit coordinates system of the `Labels` image. If the image gets transfomed (e.g. cropped), the coordinates could get out of sync, [see this issue](https://github.com/scverse/spatialdata/issues/123).

Example:

```{code-block} python

SpatialData
  - Labels: ["Label1", "Label2", ...]
  - Table: AnnData
    - obsm: "spatial" # no coordinate system, assumed to be default of implicit
                      # of `Labels`

```

#### Points

```{note}
This representation is still under discussion and it might change. What is described here is the current implementation.
```

Coordinates of points for single molecule data. Each observation is a point, and might have additional information (intensity etc.).
Current implementation represent points as a Parquet file and a [`dask.dataframe.DataFrame`](https://docs.dask.org/en/stable/dataframe.html) in memory.
The requirements are the following:

-   The table MUST contains axis name to represent the axes.
    -   If it's 2D, the axes should be `["x","y"]`.
    -   If it's 3D, the axes should be `["x","y","z"]`.
-   It MUST also contains coordinates transformations in `dask.dataframe.DataFrame().attrs["transform"]`.

Additional information is stored in `dask.dataframe.DataFrame().attrs["spatialdata_attrs"]`

-   It MAY also contains `"feature_key"`, that is, the column name of the table that refers to the features. This `Series` MAY be of type `pandas.Categorical`.
-   It MAY contains additional information in `dask.dataframe.DataFrame().attrs["spatialdata_attrs"]`, specifically:
    -   `"instance_key"`: the column name of the table where unique instance ids that this point refers to are stored, if available.

The points representation is anyway still being under [discussion](https://github.com/scverse/spatialdata/issues/46).

#### Region Table (table of annotations for regions)

Annotations of regions of interest. Each row in this table corresponds to a single region on the coordinate space. This is represented as an `AnnData` object to allow for complex annotations on the data. This includes:

-   multivariate feature support, e.g. a matrix of dimensions regions x variables;
-   annotations on top of the features or of the observations. E.g. calculated statistic, prior knowledge based annotations, cell types etc.
-   graphs of observations or variables. These can be spatial graphs, nearest neighbor networks based on feature similarity, etc.

One region table can refer to multiple sets of Regions. But each row can map to only one region in its Regions element. For example, one region table can store annotation for multiple slides, though each slide would have it's own label element.

    * 'type': str: `ngff:regions_table`
    * `region: str | list[str]`: Regions or list of regions this table refers to
    * `region_key: str`: Key in obs which says which Regions container
           this obs exists in ("library_id")..
    * `instance_key: str`: Key in obs that says which instance the obs
           represents.

If any of `region`, region_key`and`instance_key` are defined, they all MUST be defined.

#### Graphs (representation to be refined)

Graphs are stored in the annotating table for a Regions element. Graphs represent relationships between observations. Coordinates MAY be stored redundantly in the `obsm` slot of the annotating table, and are assumed to be in the intrinsic coordinate system of the label image.
Features on edges would just be separate obsp.
Features or annotation on nodes coincide with the information stored in the annotating table (either `X` or `adata.obs`).

-   Graphs on Points and Shapes could be stored in AnnData `obsp` directly, but we likely will not be using `AnnData` for Points and Shapes anymore.
-   Graphs in general (on Points, Labels, Shapes, Polygons) are stored in `obsp` of the associated label table. Coordinates are stored in `obsm` and are assumed to be in the intrinsic coordinate system of the label image. After a transformation the coordinates could get out of sync, [see this issue](https://github.com/scverse/spatialdata/issues/123).

### Summary

-   Image `type: Image`
-   Regions `type: Union[Labels, Shapes]`
    -   Labels `type: Labels`
    -   Shapes `type: Shapes`
-   Points `type: Points`
-   Tables `type: Table`

#### Open discussions

-   Multiple tables [discussion](https://github.com/scverse/spatialdata/issues/43)
-   Feature annotations and spatial coordinates in the same table [discussion](https://github.com/scverse/spatialdata/issues/45)
-   Points vs Circles [discussion](https://github.com/scverse/spatialdata/issues/46)

### Transforms and coordinate systems

In the following we refer to the NGFF proposal for transformations and coordinate systems.
You can find the [current transformations and coordinate systems specs proposal here](http://api.csswg.org/bikeshed/?url=https://raw.githubusercontent.com/bogovicj/ngff/coord-transforms/latest/index.bs), **# TODO update reference once proposal accepted**; [discussion on the proposal is here](https://github.com/ome/ngff/pull/138)).

The NGFF specifications introduces the concepts of coordiante systems and axes. Coordinate sytems are sets of axes that have a name, and where each axis is an object that has a name, a type and eventually a unit information. The set of operations required to transform elements between coordinate systems are stored as coordinate transformations. A table MUST not have a coordinate system since it annotates Region Elements (which already have one or more coordinate systems).

#### NGFF approach

There are two types of coordinate systems: intrinsic (called also implicit) and extrinsic (called also explicit). Intrinsic coordinate systems are tied to the data structure of the element and decribe it (for NGFF, an image without an intrinsic coordinate system would have no information on the axes). The intrinsic coordinate system of an image is the set of axes of the array containing the image. Extrinsic coordinate systems are not anchored to a specific element.

The NGFF specification only operates with images and labels, so it specifies rules for the coordinate systems only for these two types of elements. The main points are the following:

-   each image/labels MUST have one and only one intrinsic coordinate system;
-   each image/labels MAY have a transformation mapping them to one or more extrinsic coordinate systems;
-   a transformation MAY be defined between any two coordinate systems, including intrinsic and extrinsic coordinate systems.

Furthermore, acoording to NGFF, a coordinate system:

-   MUST have a name;
-   MUST specify all the axes.

#### SpatialData approach

In SpatialData we extend the concept of coordiante systems also for the other types of spatial elements (Points, Shapes, Polygons).
Since elements are allowed to have only (a subset of the) c, x, y, z axes and must follow a specific schema, we can relax some restrictions of the NGFF coordinate systems and provide less verbose APIs. The framework still reads and writes to valid NGFF, converting to the SpatialData coordinate system if possible or raising an error.

In details:

-   we don't need to specify the intrinsic coordinate systems, these are inferred from the element schema
-   each element MAY have a transformation mapping them to one or more extrinsic coordinate systems

Each coordinate system

-   MUST have a name
-   MAY specify its axes

We also have a constraint (that we may relax in the future):

-   a transformation MAY be defined only between an intrinsic coordinate system and an extrinsic coordinate system
-   each element MUST be mapped at least to an extrinsic coordinate system. When no mapping is specified, we define a mapping to the "global" coordinate system via an "Identity" transformation.

#### In-memory representation

We define classes that follow the NGFF specifications to represent the coordinate systems (class `NgffCoordinateSystem`) and coordinate transformations (classes inheriting from `NgffBaseTransformations`). Anyway, these classes are used only during input and ouput. For operations we define new classes (inheriting from `BaseTransformation`).

Classes inheriting from `NgffBaseTransformation` are: `NgffIdentity`, `NgffMapAxis`, `NgffTranslation`, `NgffScale`, `NgffAffine`, `NgffRotation`, `NgffSequence`, `NgffByDimension`. The following are not supported: `NgffMapIndex`, `NgffDisplacements`, `NgffCoordinates`, `NgffInverseOf`, `NgffBijection`. In the future these classes could be moved outside _SpatialData_, for instance in [ome-zarr-py](https://github.com/ome/ome-zarr-py).

Classes inheriting from `BaseTransformation` are: `Identity`, `MapAxis`, `Translation`, `Scale`, `Affine`, `Sequence`.

The conversion between the two transformation is still not 100% supported; it will be finalized when the NGFF specifications are approaved; [this issue](https://github.com/scverse/spatialdata/issues/114) keeps track of this.

#### Reasons for having two sets of classes

The `NgffBaseTransformations` require full specification of the input and output coordinate system for each transformation. A transformation MUST be compatible with the input coordinate system and output coordinate system (full description in the NGFF specification) and two transformations can be chained together only if the output coordinate system of the first coincides with the input coordinate system of the second.

On the contrary, each `BaseTransformation` is self-defined and does not require the information on coordinate systems. Almost (see below) any transformation can be applied unambiguously to any element and almost any pair of transformations can be chained together. The result is either uniqueuly defined, either an exception is raised when there is ambiguity.

Precisely, this is performed by passing through unaltered those axis that are present in an element but not in a transformation, and by ignoring axes that are present in a transformation but not in an element.

For example one can apply a `Scale([2, 3, 4], axes=('x', 'y', 'z'))` to a `cyx` image (the axes `c` is passed through unaltered, and the scaling on `z` is ignored since there is no `z` axis.)

An example of transformation that cannot be applied is an `Affine xy -> xyz` to `xyz` data, since `z` can't be passed through as it is also the output of the transformation.

To know more about the separation between the two set of classes see [this issue](https://github.com/scverse/spatialdata/issues/39), [this other issue](https://github.com/scverse/spatialdata/issues/47) and [this pr](https://github.com/scverse/spatialdata/pull/100).

This document will be udpated with the precise description of each transformation, for the moment please refer to `transformations.py` to see the exact implementation.

### Examples

#### Transformations

See [this notebook](https://github.com/scverse/spatialdata-notebooks/blob/main/notebooks/transformations.ipynb) for extensive examples on the transformations.

#### Legacy examples

_The text down below may not reflect the latest version of the code and will be eventually replaced by notebooks_

Here is a short list of examples of the elements used to represent some spatial omics datasets. Real world examples will be available as notebooks [in this repository](https://github.com/scverse/spatialdata-notebooks), furthermore, some draft [implementations are available here](https://github.com/giovp/spatialdata-sandbox).

API

```python
import spatialdata as sd
from spatialdata import SpatialData

sdata = SpatialData(...)
points = sd.transform(sdata.points["image1"], tgt="tgt_space")
sdata = sd.transform(sdata, tgt="tgt_space")
```

The transfromation object should not have a method to apply itself to an element.
`SpatialData` can have a `transform` method, that can be applied to either a `SpatialData` object or an element.

##### Layout of a SpatialData object

The layout of some common datasets.

**Layout of [MERFISH example](https://github.com/giovp/spatialdata-sandbox/tree/main/merfish)**

-   points (coordinates of spots);
-   each point has features (e.g., gene, size, cell assignment);
-   segmented cell locations are saved as labels (missing in this example) or approximated as circles of variable diameter;
-   gene expression for cells, obtained by counting the points inside each cell;
-   large anatomical regions saved as polygons;
-   rasterized version of the single molecule points (to mimic the original hires image, missing in this example).

**Layout of [Visium example](https://github.com/giovp/spatialdata-sandbox/tree/main/visium)**

-   The datasets include multiple slides from the same individual, or slides from multiple samples;
-   "Visium spots" (circular regions) where sequences are captured;
-   each spot has RNA expression;
-   H&E image (multiscale 2D);
-   (optional) large microscopy (e.g. 40x magnification, 50K x 50K pixels) images may be available, which would need to be aligned to the rest of spatial elements;
-   (optional) cell segmentation labels can be derived from the H&E images;
-   (optional) the cell segmentation can be annotated with image-derived features (image features/statistics).

##### Code/pseudo-code workflows

**Workflows to show**

-   [x] loading multiple samples visium data from disk (SpaceRanger), concatenating and saving them to .zarr
-   [x] loading a generic NGFF dataset
-   [ ] calling the SpatialData constructor with some transformations on it
-   [x] accumulation with multiple types of elements
-   [x] subsetting/querying by coordinate system, bounding box, spatial region, table rows

##### Loading multiple Visium samples from the SpaceRanger output and saving them to NGFF using the SpatialData APIs

```python
import spatialdata as sd
from spatialdata_io import read_visium

samples = ["152806", "152807", "152810", "152811"]
sdatas = []

for sample in samples:
    sdata = read_visium(path=sample, coordinate_system_name=sample)
    sdatas.append(sdata)

sdata = sd.SpatialData.concatenate(sdatas, merge_tables=True)
sdata.write("data.zarr")
```

##### Loading multiple Visium samples from a generic NGFF storage with arbitrary folder structure (i.e. a NGFF file that was not created with the SpatialData APIs).

This is the multislide Visium use case.

```python console
>>> # This dataset comprises multiple Visium slides which have been stored in a unique OME-NGFF store
... ngff_store = open_container(uri)
... ngff_store
data.zarr
├── sample_0
│   ├── circles
│   ├── hne_image
│   └── table
├── sample_1
│   ├── circles
│   ├── hne_image
│   └── table
├── sample_2
│   ├── circles
│   ├── hne_image
│   └── table
└── sample_3
    ├── circles
    ├── hne_image
    └── table
>>> # Read in each Visium slide as a separate SpatialData object. Each table has each row associated to a Circles element, which belongs to the same coordinate system of the corresponding H&E image. For this reason specifying a table is enough to identify and extract a SpatialData object.
... slides = {}
... for sample_name in ["sample_1", "sample_2", "sample_3", "sample_4"]:
...     slides[sample_name] = ngff_store.get_spatial_data(f"{sample_name}_table")
... slides["sample_1"]
SpatialData object with:
├── Images
│     ├── 'sample_1': DataArray (2000, 1969, 3)
├── Regions
│     ├── 'sample_1': Circles (2987)
└── Table
      └── 'AnnData object with n_obs × n_vars = 2987 × 31053
    obs: "in_tissue", "array_row", "array_col", "library_id", "visium_spot_id"'

>>> # Combine these to do a joint analysis over a collection of slides
... joint_dataset = spatialdata.concatenate(slides)
... joint_dataset
SpatialData object with:
├── Images
│     ├── 'sample_1': DataArray (2000, 1969, 3)
│     ├── 'sample_2': DataArray (2000, 1969, 3)
│     ├── 'sample_3': DataArray (2000, 1968, 3)
│     ├── 'sample_4': DataArray (2000, 1963, 3)
├── Regions
│     ├── 'sample_1': Circles (2987)
│     ├── 'sample_2': Circles (3499)
│     ├── 'sample_3': Circles (3497)
│     ├── 'sample_4': Circles (2409)
└── Table
      └── 'AnnData object with n_obs × n_vars = 12392 × 31053
    obs: "in_tissue", "array_row", "array_col", "library_id", "visium_spot_id", "library"'
```

##### Aggregating spatial information from an element into a set of regions

```python
sdata = from_zarr("data.zarr")
table = spatialdata.aggregate(
    source="/images/image", regions="/circles/visium_spots", method=["mean", "std"]
)
```

##### Subsetting/querying by coordinate system, bounding box, spatial region, table rows

```python
"""
SpatialData object with:
├── images
│     ├── '/images/point16': DataArray (3, 1024, 1024), with axes: c, y, x
│     ├── '/images/point23': DataArray (3, 1024, 1024), with axes: c, y, x
│     └── 'point8': DataArray (3, 1024, 1024), with axes: c, y, x
├── labels
│     ├── '/labels/point16': DataArray (1024, 1024), with axes: y, x
│     ├── 'point23': DataArray (1024, 1024), with axes: y, x
│     └── 'point8': DataArray (1024, 1024), with axes: y, x
├── polygons
│     └── 'Shapes_point16_1': AnnData with obs.spatial describing 2 polygons, with axes x, y
└── table
      └── 'AnnData object with n_obs × n_vars = 3309 × 36
    obs: 'row_num', 'point', 'cell_id', 'X1', 'center_rowcoord', 'center_colcoord', 'cell_size', 'category', 'donor', 'Cluster', 'batch', 'library_id'
    uns: 'mapping_info'
    obsm: 'X_scanorama', 'X_umap', 'spatial'': AnnData (3309, 36)
with coordinate systems:
▸ point16
    with axes: c, y, x
    with elements: /images/point16, /labels/point16, /polygons/Shapes_point16_1
▸ point23
    with axes: c, y, x
    with elements: /images/point23, /labels/point23
▸ point8
    with axes: c, y, x
    with elements: /images/point8, /labels/point8
"""
sdata0 = sdata.query.coordinate_system("point23", filter_rows=False)
sdata1 = sdata.query.bounding_box((0, 20, 0, 300))
sdata1 = sdata.query.polygon("/polygons/annotations")
# TODO: syntax discussed in https://github.com/scverse/spatialdata/issues/43
sdata1 = sdata.query.table(...)
```

##### Related notes/issues/PRs

-   [Issue discussing SpatialData layout](https://github.com/scverse/spatialdata/issues/12)
-   [Notes from Basel Hackathon](https://hackmd.io/MPeMr2mbSRmeIzOCgwKbxw)
