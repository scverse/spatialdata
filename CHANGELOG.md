# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [0.1.1] - xxxx-xx-xx

### Added

-   Added method `update_annotated_regions_metadata() which updates the `region`value automatically from the `region_key` columns

### Changed

-   Renamed `join_sdata_spatialelement_table` to `join_spatialelement_table`, and made it work also without `SpatialData` objects.

## [0.1.0] - 2024-03-24

### Added

#### Major

-   Implemented support in SpatialData for storing multiple tables. These tables can annotate a SpatialElement but not
    necessarily so.
-   Added SQL like joins that can be executed by calling one public function `join_sdata_spatialelement_table`. The
    following joins are supported: `left`, `left_exclusive`, `right`, `right_exclusive` and `inner`. The function has
    an option to match rows. For `left` only matching `left` is supported and for `right` join only `right` matching of
    rows is supported. Not all joins are supported for `Labels` elements. The elements and table can either exist within
    a `SpatialData` object or outside.
-   Added function `match_element_to_table` which allows the user to perform a right join of `SpatialElement`(s) with a
    table with rows matching the row order in the table.
-   Increased in-memory vs on-disk control: changes performed in-memory (e.g. adding a new image) are not automatically
    performed on-disk.

#### Minor

-   Added public helper function get_table_keys in spatialdata.models to retrieve annotation information of a given
    table.
-   Added public helper function check_target_region_column_symmetry in spatialdata.models to check whether annotation
    metadata in table.uns['spatialdata_attrs'] corresponds with respective columns in table.obs.
-   Added function validate_table_in_spatialdata in SpatialData to validate the annotation target of a table being
    present in the SpatialData object.
-   Added function get_annotated_regions in SpatialData to get the regions annotated by a given table.
-   Added function get_region_key_column in SpatialData to get the region_key column in table.obs.
-   Added function get_instance_key_column in SpatialData to get the instance_key column in table.obs.
-   Added function set_table_annotates_spatialelement in SpatialData to either set or change the annotation metadata of
    a table in a given SpatialData object.
-   Added table_name parameter to the aggregate function to allow users to give a custom table name to table resulting
    from aggregation.
-   Added table_name parameter to the get_values function.
-   Added tables property in SpatialData.
-   Added tables setter in SpatialData.
-   Added gen_spatial_elements generator in SpatialData to generate the SpatialElements in a given SpatialData object.
-   Added gen_elements generator in SpatialData to generate elements of a SpatialData object including tables.
-   added SpatialData.subset() API
-   added SpatialData.locate_element() API
-   added utils function: transform_to_data_extent()
-   added utils function: are_extents_equal()
-   added utils function: postpone_transformation()
-   added utils function: remove_transformations_to_coordinate_system()
-   added utils function: get_centroids()
-   added utils function: deepcopy()
-   added operation: to_circles()
-   added testing utilities: assert_spatial_data_objects_are_identical(), assert_elements_are_identical(),
    assert_elements_dict_are_identical()

### Changed

#### Major

-   refactored data loader for deep learning

#### Minor

-   Changed the string representation of SpatialData to reflect the changes in regard to multiple tables.

### Fixed

#### Major

-   improved usability and robustness of sdata.write() when overwrite=True @aeisenbarth
-   generalized queries to any combination of 2D/3D data and 2D/3D query region #409
-   fixed warnings for categorical dtypes in tables in TableModel and PointsModel

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
