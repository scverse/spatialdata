# Interoperability

The on-disk representation of SpatialData can be read from other languages. Here we list interfaces for working with SpatialData from your language of choice:

## R

- [spatialdataR](https://helenalc.github.io/spatialdataR/) provides an R implementation of the `SpatialData` object, with out-of-memory images and labels, `duckdb`-backed points and shapes, and tables represented as `SingleCellExperiment` objects.

## JavaScript and TypeScript

- [SpatialData.js](https://github.com/Taylor-CCB-Group/SpatialData.js) provides a TypeScript and JavaScript library for interfacing with SpatialData stores.
- [Vitessce](https://vitessce.io/docs/data-file-types/#spatialdatazarr) reads `spatialdata.zarr` stores directly and uses them for interactive visualization.

## File format

The SpatialData on-disk format builds on [OME-NGFF](https://ngff.openmicroscopy.org/latest/). See the [design document](design_doc.md) for details of the current on-disk layout.
