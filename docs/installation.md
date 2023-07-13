# Installation

`spatialdata` requires Python version >= 3.9 to run and the installation time requires a few minutes on a standard desktop computer.

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

To install `spatialdata` from GitHub, run:

```bash
pip install git+https://github.com/scverse/spatialdata
```

Alternative you can clone the repository and do an editable install with:

```bash
pip install -e .
```

This is the reccommended way to install the package in case in which you want to contribute to the code. To update the package you can use `git pull`.

Important: if you perform an editable install of `spatialdata` and then install `spatialdata-plot`, `spatialdata-io` or `napari-spatialdata`, they may automatically override the installation of `spatialdata` with the version from PyPI. To check if this happened you can run `python -c "import spatialdata; print(spatialdata.__path__)"`: if you get a path that contains `site-packages`, then your editable installation has been overridden and you need to reinstall the package by rerunning `pip install -e .` in the cloned `spatialdata` repo.

<!-- Links -->

[napari-spatialdata]: https://github.com/scverse/napari-spatialdata
[spatialdata-io]: https://github.com/scverse/spatialdata-io
[spatialdata-plot]: https://github.com/scverse/spatialdata-plot
