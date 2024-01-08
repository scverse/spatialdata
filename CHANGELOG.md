# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [0.1.0] - tbd

### Added

#### Major

-   Implemented support in SpatialData for storing multiple tables. These tables
    can annotate a SpatialElement but not necessarily so.
-   Increased in-memory vs on-disk control: changes performed in-memory (e.g. adding a new image) are not automatically performed on-disk.

#### Minor

-   Added public helper function get_table_keys in spatialdata.models to retrieve annotation information of a given table.
-   Added public helper function check_target_region_column_symmetry in spatialdata.models to check whether annotation
    metadata in table.uns['spatialdata_attrs'] corresponds with respective columns in table.obs.

## [0.0.x] - tbd

### Minor

-   improved usability and robustness of sdata.write() when overwrite=True @aeisenbarth

### Added

### Fixed

#### Minor

-   refactored data loader for deep learning

## [0.0.14] - 2023-10-11

### Added

#### Minor

-   new API: sdata.rename_coordinate_systems()

#### Technical

-   decompose affine transformation into simpler transformations
-   remove padding for blobs()

#### Major

-   get_extent() function to compute bounding box of the data

#### Minor

-   testing against pre-release packages

### Fixed

-   Fixed bug with get_values(): ignoring background channel in labels

## [0.0.13] - 2023-10-02

### Added

-   polygon_query() support for images #358

### Fixed

-   Fix missing c_coords argument in blobs multiscale #342
-   Replaced hardcoded string with instance_key #346

## [0.0.12] - 2023-06-24

### Added

-   Add multichannel blobs sample data (by @melonora)

## [0.0.11] - 2023-06-21

### Improved

-   Aggregation APIs.

## [0.0.10] - 2023-06-06

### Fixed

-   Fix blobs (#282)

## [0.0.9] - 2023-05-23

### Updated

-   Update napari-spatialdata pin (#279)
-   pin typing-extensions

## [0.0.8] - 2023-05-22

### Merged

-   Merge pull request #271 from scverse/fix/aggregation

## [0.0.7] - 2023-05-20

### Updated

-   Update readme

## [0.0.6] - 2023-05-10

### Added

-   This release adds polygon spatial query.

## [0.0.5] - 2023-05-05

### Fixed

-   fix tests badge (#242)

## [0.0.4] - 2023-05-04

### Tested

-   This release tests distribution via pypi

## [0.0.3] - 2023-05-02

### Added

-   This is an alpha release to test the release process.

## [0.0.2] - 2023-05-02

### Added

-   make version dynamic

## [0.0.1.dev1] - 2023-03-25

### Added

-   Dev version, not official release yet
