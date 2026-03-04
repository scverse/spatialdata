# Context for SpatialData PR #1055: Lazy Table Loading

## PR Link
https://github.com/scverse/spatialdata/pull/1055

## Background

This PR adds `lazy: bool = False` to `SpatialData.read()` and `read_zarr()` so that AnnData tables are loaded lazily via dask, keeping large matrices out of memory. This matters for Mass Spectrometry Imaging (MSI) datasets where tables can have millions of pixels and hundreds of thousands of m/z bins (e.g. 179,389 x 460,517 = ~40GB dense, though stored as sparse CSC).

### What the PR already does (6 commits):
- `io_table.py`: Added lazy parameter, uses `anndata.experimental.read_lazy()` when `lazy=True`
- `io_zarr.py`: Passes `lazy` through to `_read_table()`
- `spatialdata.py`: Passes `lazy` through to `read_zarr()`; skips eager validation for lazy tables
- `relational_query.py`: Fixed query operations to handle `Dataset2D` obs from lazy tables
- `_utils.py`: Fixed `_inplace_fix_subset_categorical_obs()` for lazy tables
- `models.py`: Modified validation to skip eager checks for lazy tables
- `tests/io/test_readwrite.py`: Added lazy loading tests

### Benchmark (from PR):
100,000 pixels x 100,000 m/z bins, 3,000 peaks/pixel (~296M non-zeros):
- Memory: 15.4 MB vs 2,270.7 MB (99% savings)
- Load time: 0.13s vs 1.57s (12x faster)

---

## Investigation Results

We tested the PR against real-world MSI datasets and found three important things.

### Finding 1: Chunking is already correct -- no fix needed

The original concern was that `read_lazy()` wraps the entire sparse matrix as a single dask chunk, defeating lazy loading. **This is NOT true** with anndata 0.13.

`read_lazy()` internally calls `read_elem_lazy(elem)` without passing `chunks`, which triggers these defaults in `read_elem_lazy`:
- CSC sparse: `(n_rows, 1000)` -- 1000-column chunks
- CSR sparse: `(1000, n_cols)` -- 1000-row chunks
- Dense: uses on-disk zarr chunk layout

Verified on real data (179,389 x 460,517 CSC sparse):

```python
from anndata.experimental import read_elem_lazy
import zarr

x_group = zarr.open_group("dataset.zarr/tables/my_table/X", mode="r")
lazy_X = read_elem_lazy(x_group)
print(lazy_X.chunks)
# ((179389,), (1000, 1000, 1000, ... 1000, 517))
# 461 column chunks -- correct!
```

Column slicing only reads the relevant zarr chunks from disk. No rechunking needed.

### Finding 2: `read_lazy()` crashes on some real datasets -- but it's a writer problem

`read_lazy()` crashes on some datasets with:

```
IORegistryError: No read method registered for IOSpec(encoding_type='', encoding_version='')
from <class 'zarr.core.array.Array'>.
Error raised while reading key 'y' of <class 'zarr.core.array.Array'> from /obs
```

**Root cause**: anndata has two separate IO registries:

| Registry | Readers for zarr.Array | Purpose |
|---|---|---|
| **Eager** (`_REGISTRY`) | 8 readers | Used by `read_elem()`, `read_zarr()` |
| **Lazy** (`_LAZY_REGISTRY`) | 2 readers | Used by `read_elem_lazy()`, `read_lazy()` |

The eager registry has a catch-all reader for `IOSpec('', '')` (plain arrays with no encoding metadata). The lazy registry does NOT. So when obs columns are stored without `encoding-type` attributes, eager reading works but lazy reading crashes.

**The key question was: is this a reader bug or a writer bug?**

We tested what anndata itself writes:

```python
# When anndata writes an int32 obs column via write_zarr():
#   encoding-type='array', encoding-version='0.2.0'    <-- STAMPED
#
# When anndata writes a string obs column via write_zarr():
#   encoding-type='string-array', encoding-version='0.2.0'    <-- STAMPED
#
# When anndata writes a categorical:
#   encoding-type='categorical', encoding-version='0.2.0'    <-- STAMPED
```

Then we checked all real datasets:

| Dataset | obs encoding metadata | Created by |
|---|---|---|
| Hippocampus.zarr | **MISSING** on all non-categorical columns | Thyra streaming COO converter |
| mouse_brain.zarr | **MISSING** on all non-categorical columns | Thyra streaming COO converter |
| sample_A.zarr | `'array'` / `'string-array'` on all columns | Standard anndata `write_zarr()` |
| sample_B.zarr | `'array'` / `'string-array'` on all columns | Standard anndata `write_zarr()` |
| xenium.zarr | `'array'` / `'string-array'` on all columns | Standard anndata `write_zarr()` |

**Conclusion**: The datasets created through anndata's standard `write_zarr()` have proper encoding metadata and `read_lazy()` would work on them. The datasets created by Thyra's streaming COO converter write raw zarr arrays without the encoding attributes. The fix belongs in the writer.

### Finding 3: `table.uns` values become dask arrays after `read_lazy()`

All values in `table.uns` (unstructured metadata like mean spectra, peak lists, parameter dicts) become dask arrays:

```python
lazy = read_lazy(zarr_path)
type(lazy.uns['mean_spectra']['global_mean']['mz'])   # dask.array.core.Array
type(lazy.uns['peak_lists']['auto_peaks']['indices'])  # dask.array.core.Array
```

`np.array()` on a dask array silently materializes it (no crash), but direct JSON/Pydantic serialization fails. Downstream code needs `.compute()` or `np.asarray()` before serialization.

This is not a bug -- it's how `read_lazy()` works. But it's a gotcha for downstream consumers. If the writer is fixed and `read_lazy()` is used, this needs to be documented or handled.

---

## What Needs to Be Done

### Option A: Fix the writer (recommended)

Fix Thyra's streaming COO converter to write proper anndata encoding attributes on obs columns. When writing a zarr array for an obs column, add:

- For numeric arrays (int32, float64, etc.):
  ```python
  arr = zarr.open_array(path, ...)
  arr.attrs['encoding-type'] = 'array'
  arr.attrs['encoding-version'] = '0.2.0'
  ```

- For string arrays:
  ```python
  arr = zarr.open_array(path, ...)
  arr.attrs['encoding-type'] = 'string-array'
  arr.attrs['encoding-version'] = '0.2.0'
  ```

With this fix, `read_lazy()` works on all datasets, the PR's current implementation using `read_lazy()` is correct, and the chunking is already optimal for sparse data.

You would also want a migration script to stamp the encoding metadata on existing datasets (Hippocampus.zarr, mouse_brain.zarr) so they work with lazy loading too.

### Option B: Piecewise loading in SpatialData (workaround)

If the writer can't be fixed (or for backwards compatibility with existing datasets), change `_read_table()` in SpatialData to build the AnnData piecewise:

1. Read obs, var, uns eagerly via `read_elem()` (handles missing encoding metadata; obs/var/uns are small)
2. Read X lazily via `read_elem_lazy()` (proper chunking automatic for sparse)
3. Read layers lazily if they exist
4. Assemble into AnnData

This sidesteps the crash AND the uns dask-wrapping issue, but it's working around data that wasn't written to anndata spec.

### The PR as-is

If Option A is done, the PR's current approach (`read_lazy()` on the whole table) is already correct:
- Chunking works for sparse data
- Query API fixes are in place
- Validation skipping is correct
- The uns dask-wrapping is the only remaining gotcha to document

The main remaining work would be improving test coverage (currently 84% patch, 7 uncovered lines).

---

## Known Downstream Limitations (not SpatialData bugs)

### dask + scipy.sparse aggregate bug

`.mean(axis=0)` and `.sum(axis=0)` fail with a `keepdims` error when dask wraps scipy.sparse chunks. Workaround: iterate over column chunks manually.

### sklearn/UMAP require materialized arrays

Pattern is "deferred materialization" -- subset features BEFORE `.compute()`:

```python
# Materializes only ~200 columns instead of 460,517
data = table.X[:, feature_indices].compute()
```

---

## Key Files

| File | Purpose |
|------|---------|
| `src/spatialdata/_io/io_table.py` | Core lazy loading logic |
| `src/spatialdata/_io/io_zarr.py` | Passes lazy parameter through |
| `src/spatialdata/_core/spatialdata.py` | Entry point for `SpatialData.read()` |
| `src/spatialdata/_core/query/relational_query.py` | Query API fixes for Dataset2D obs |
| `src/spatialdata/_utils.py` | Helper for lazy AnnData detection |
| `src/spatialdata/models/models.py` | Validation skip for lazy tables |
