# Changelog

Please refer directly to the [Releases](https://github.com/scverse/spatialdata/releases) section on GitHub, where you can find curated release notes for each release.
For developers, please consult the [contributing guide](https://github.com/scverse/spatialdata/blob/main/docs/contributing.md), which explains how to keep release notes are up-to-date at each release.

## Unreleased

### Changed

- `SpatialData.read()` now only accepts a path or URL (`str`, `Path` or `UPath`) for `file_path`; passing an already-open `zarr.Group` raises a `TypeError`. Use `spatialdata.read_zarr()` to read from a `zarr.Group`.
- `SpatialData.attrs` is now typed as `dict[str, JSONValue]` instead of `dict[Any, Any]`, making explicit the pre-existing invariant that the attrs must be JSON-serializable (they are stored as Zarr attributes). Accordingly, `SpatialData.get_attrs()` is now typed as returning `JSONValue | pd.DataFrame`: the previous annotation did not cover attrs values such as lists, numbers, booleans and `None`, which have always been valid and are returned as-is when `return_as=None`. This is a typing-only change, the runtime behaviour is unchanged.
- `SpatialData.locate_element()` now accepts a table (`AnnData`) in addition to a `SpatialElement`; tables were already located correctly, but the annotation did not allow passing them. This is a typing-only change, the runtime behaviour is unchanged.
