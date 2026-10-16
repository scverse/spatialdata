# Changelog

Please refer directly to the [Releases](https://github.com/scverse/spatialdata/releases) section on GitHub, where you can find curated release notes for each release.
For developers, please consult the [contributing guide](https://github.com/scverse/spatialdata/blob/main/docs/contributing.md), which explains how to keep release notes are up-to-date at each release.

## Unreleased

### Added

- Tables can now be `mudata.MuData` objects, to store multimodal measurements of the same instances (e.g. RNA and protein of the same cells) in a single table. The annotation metadata lives in the global `mdata.uns` and the `region_key`/`instance_key` columns in the global `mdata.obs`; modalities may cover a subset of the instances. MuData tables are supported by `TableModel.parse()`/`validate()`, reading and writing (zarr v2 and v3), `sanitize_table()`, the joins and `match_table_to_element()`, spatial queries, `filter_by_table_query()` (`obs_expr` and `obs_names_expr` only), `get_values()` (use `"<modality>:<key>"` to take an `obs` column or a variable from a modality), `concatenate()` and `deepcopy()`. `rasterize_bins()` and `ImageTilesDataset` raise an explicit error for MuData tables. Following the MuData convention, the column names of the global `obs`/`var` of a MuData table may contain `":"` (e.g. `"rna:leiden"` from `pull_obs()`, or the modality weights of muon's WNN); it is stored as `"%3A"` on disk, since `":"` is not allowed in Windows file names. `mudata>=0.4.1` is now a dependency.

### Changed

- `SpatialData.read()` now only accepts a path or URL (`str`, `Path` or `UPath`) for `file_path`; passing an already-open `zarr.Group` raises a `TypeError`. Use `spatialdata.read_zarr()` to read from a `zarr.Group`.
- `SpatialData.attrs` is now typed as `dict[str, JSONValue]` instead of `dict[Any, Any]`, making explicit the pre-existing invariant that the attrs must be JSON-serializable (they are stored as Zarr attributes). Accordingly, `SpatialData.get_attrs()` is now typed as returning `JSONValue | pd.DataFrame`: the previous annotation did not cover attrs values such as lists, numbers, booleans and `None`, which have always been valid and are returned as-is when `return_as=None`. This is a typing-only change, the runtime behaviour is unchanged.
- `SpatialData.locate_element()` now accepts a table (`AnnData`) in addition to a `SpatialElement`; tables were already located correctly, but the annotation did not allow passing them. This is a typing-only change, the runtime behaviour is unchanged.

### Fixed

- Querying a `SpatialData` object with multiple (batched) bounding boxes now raises an explicit `NotImplementedError` instead of failing with an opaque `AssertionError`. Batched queries are supported when querying a `SpatialElement` directly, but not when querying a `SpatialData` object, since that would require returning one `SpatialData` object per bounding box.
