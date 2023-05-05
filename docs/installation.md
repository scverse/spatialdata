# Installation

`spatialdata` requires Python version >= 3.9 to run.

## PyPI

Install `spatialdata` by running::

```bash
    pip install spatialdata
```

## Visualization and IO

The SpatialData ecosystem is designed to work with the following packages:

-   [spatialdata-io][]: `spatialdata` IO for common spatial omics technologies.
-   [spatialdata-plot][]: Static plotting library for `spatialdata`.
-   [napari-spatialdata][]: napari plugin for `spatialdata`.

They can be installed with:

```bash
pip install "spatialdata[extra]"
```

## Additional dependencies

To use the dataloader in `spatialdata`, torch needs to be installed. This can be done with:

```bash
pip install "spatialdata[torch]"
```

## Development version

To install `spatialdata` from GitHub, run::

```bash
pip install git+https://github.com/scverse/spatialdata
```

<!-- Links -->

[napari-spatialdata]: https://github.com/scverse/napari-spatialdata
[spatialdata-io]: https://github.com/scverse/spatialdata-io
[spatialdata-plot]: https://github.com/scverse/spatialdata-plot
