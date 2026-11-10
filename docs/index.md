```{eval-rst}
.. image:: _static/img/spatialdata_horizontal.svg
  :class: dark-light p-2
  :alt: SpatialData banner
```

# An open and universal framework for processing spatial omics data.

SpatialData is a data framework, a schema and a serialization format for uni- and multi-modal spatial omics datasets. It is supported by a growing suite of software libraries for performant access, manipulation and analysis in multiple languages, including Python, R and JavaScript.

The Python `spatialdata` library lets you:

- hold images, segmentation masks (labels), points (e.g. transcript locations), shapes (e.g. cell boundaries or Visium spots) and annotation tables (e.g. gene expression per cell) together in one {class}`~spatialdata.SpatialData` object;
- align elements and whole datasets through coordinate transformations and shared coordinate systems;
- query and crop data spatially, and aggregate measurements across elements, e.g. transcript counts or image intensities per cell;
- read and write the SpatialData Zarr format, with lazy access to large data through dask;
- exchange data with AnnData, scanpy, squidpy, GeoPandas, xarray and other tools ({doc}`interoperability`).

This site documents how to install, use and extend it. See the links below to learn more about the specifications and other packages in the SpatialData ecosystem.

- [SpatialData specifications](https://spatialdata-spec.scverse.org/): language-independent definitions of SpatialData datasets and their on-disk representation.
- `spatialdata-io`: load data from common spatial omics technologies into `spatialdata` ([repository][spatialdata-io-repo], [documentation][spatialdata-io-docs]).
- `spatialdata-plot`: Static plotting library for `spatialdata` ([repository][spatialdata-plot-repo], [documentation][spatialdata-plot-docs]).
- `napari-spatialdata`: napari plugin for interactive exploration and annotation of `spatialdata` ([repository][napari-spatialdata-repo], [documentation][napari-spatialdata-docs]).
- `spatialdataR`: R implementation of the SpatialData framework ([repository][spatialdata-r-repo]).
- `SpatialData.js`: JavaScript/TypeScript implementation of the SpatialData framework ([repository][spatialdata-js-repo]).

Please see our publication {cite}`marconatoSpatialDataOpenUniversal2024` for citation and to learn more.

```{eval-rst}
.. note::
   This library is currently under active development. We may make changes to the API between versions as the community provides feedback. To ensure reproducibility, please make note of the version you are developing against.
```

:::{note}
With dask's partition fusion enabled (the default), user code that operates on multi-partition `Points` dataframes can fail as described in [#1064](https://github.com/scverse/spatialdata/issues/1064) and [dask#12193](https://github.com/dask/dask/issues/12193).
`spatialdata` avoids this internally; for your own operations, wrap them in `spatialdata.disable_dask_tune_optimization`:

```python
import contextlib

from spatialdata import disable_dask_tune_optimization

with disable_dask_tune_optimization() if data.npartitions > 1 else contextlib.nullcontext():
    ...  # your operation on the points dask dataframe
```
:::

[//]: # "numfocus-fiscal-sponsor-attribution"

spatialdata is part of the scverse® project ([website](https://scverse.org), [governance](https://scverse.org/about/roles)) and is fiscally sponsored by [NumFOCUS](https://numfocus.org/).
If you like scverse® and want to support our mission, please consider making a tax-deductible [donation](https://numfocus.org/donate-to-scverse) to help the project pay for developer time, professional services, travel, workshops, and a variety of other needs.

<div align="center">
<a href="https://numfocus.org/project/scverse">
  <img
    src="https://raw.githubusercontent.com/numfocus/templates/master/images/numfocus-logo.png"
    width="200"
  >
</a>
</div>

```{toctree}
:caption: Getting started
:hidden: true
:maxdepth: 1

installation
quickstart
```

```{toctree}
:caption: Guides
:hidden: true
:maxdepth: 1

user_guide
tutorials/notebooks/notebooks
```

```{toctree}
:caption: Reference
:hidden: true
:maxdepth: 1

api
tutorials/notebooks/datasets/README
glossary
SpatialData specifications <https://spatialdata-spec.scverse.org/>
design_doc
interoperability
```

```{toctree}
:caption: Project
:hidden: true
:maxdepth: 1

changelog
contributing
citation
references
GitHub <https://github.com/scverse/spatialdata>
Discourse <https://discourse.scverse.org/c/help/spatialdata/44>
```

<!-- Links -->

[napari-spatialdata-repo]: https://github.com/scverse/napari-spatialdata
[spatialdata-io-repo]: https://github.com/scverse/spatialdata-io
[spatialdata-plot-repo]: https://github.com/scverse/spatialdata-plot
[napari-spatialdata-docs]: https://spatialdata.scverse.org/projects/napari/en/stable/notebooks/spatialdata.html
[spatialdata-io-docs]: https://spatialdata.scverse.org/projects/io/en/stable/
[spatialdata-plot-docs]: https://spatialdata.scverse.org/projects/plot/en/stable/
[spatialdata-r-repo]: https://github.com/HelenaLC/spatialdataR
[spatialdata-js-repo]: https://github.com/Taylor-CCB-Group/SpatialData.js
