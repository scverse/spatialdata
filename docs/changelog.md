# Changelog

Please refer directly to the [Releases](https://github.com/scverse/spatialdata/releases) section on GitHub, where you can find curated release notes for each release.
For developers, please consult the [contributing guide](https://github.com/scverse/spatialdata/blob/main/docs/contributing.md), which explains how to keep release notes are up-to-date at each release.

## Unreleased

### Changed

- `SpatialData.read()` now only accepts a path or URL (`str`, `Path` or `UPath`) for `file_path`; passing an already-open `zarr.Group` raises a `TypeError`. Use `spatialdata.read_zarr()` to read from a `zarr.Group`.
- `SpatialData.attrs` is now typed as `dict[str, JSONValue]` instead of `dict[Any, Any]`, making explicit the pre-existing invariant that the attrs must be JSON-serializable (they are stored as Zarr attributes). Accordingly, `SpatialData.get_attrs()` is now typed as returning `JSONValue | pd.DataFrame`: the previous annotation did not cover attrs values such as lists, numbers, booleans and `None`, which have always been valid and are returned as-is when `return_as=None`. This is a typing-only change, the runtime behaviour is unchanged.
- `SpatialData.locate_element()` now accepts a table (`AnnData`) in addition to a `SpatialElement`; tables were already located correctly, but the annotation did not allow passing them. This is a typing-only change, the runtime behaviour is unchanged.

### Fixed

- `rasterize(..., return_regions_as_labels=True)` now supports more than 65,535 labels: the output dtype is `uint16`, `uint32` or `uint64` depending on the maximum label. Labels of 2**53 or higher (also from summing overlapping labels) raise a `ValueError` instead of being silently rounded (datashader aggregates in float64).
- Querying a `SpatialData` object with multiple (batched) bounding boxes now raises an explicit `NotImplementedError` instead of failing with an opaque `AssertionError`. Batched queries are supported when querying a `SpatialElement` directly, but not when querying a `SpatialData` object, since that would require returning one `SpatialData` object per bounding box.
