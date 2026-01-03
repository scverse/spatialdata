# Installation

`spatialdata` requires Python (the minimum required version is specified in PyPI) to run and the installation time requires a few minutes on a standard desktop computer.

## PyPI

Install `spatialdata` by running:

```bash
    pip install spatialdata
```

## Visualization and readers

The SpatialData ecosystem is designed to work with the following packages:

- [spatialdata-io][]: `spatialdata` readers and converters for common spatial omics technologies.
- [spatialdata-plot][]: Static plotting library for `spatialdata`.
- [napari-spatialdata][]: napari plugin for `spatialdata`.

They can be installed with:

```bash
pip install "spatialdata[extra]"
```

## Additional dependencies

To use the `PyTorch` dataloader in `spatialdata`, `torch` needs to be installed. This can be done with:

```bash
pip install "spatialdata[torch]"
```

## Development version

To install `spatialdata` from GitHub, run:

```bash
pip install git+https://github.com/scverse/spatialdata
```

Alternative you can clone the repository (or a fork of it if you are contributing) and do an editable install with:

```bash
pip install -e .
```

This is the reccommended way to install the package in case in which you want to contribute to the code. In this case, to subsequently update the package you can use `git pull`.

### A note on editable install

If you perform an editable install of `spatialdata` and then install `spatialdata-plot`, `spatialdata-io` or `napari-spatialdata`, they may automatically override the installation of `spatialdata` with the version from PyPI.

To check if this happened you can run

```
python -c "import spatialdata; print(spatialdata.__path__)"
```

if you get a path that contains `site-packages`, then your editable installation has been overridden and you need to reinstall the package by rerunning `pip install -e .` in the cloned `spatialdata` repo.

## Conda

You can install the `spatialdata`, `spatialdata-io`, `spatialdata-plot` and `napari-spatialdata` packages from the `conda-forge` channel using

```bash
mamba install -c conda-forge spatialdata spatialdata-io spatialdata-plot napari-spatialdata
```

Update: currently (Feb 2025), due to particular versions being unavailable on `conda-forge` for some (dependencies of our) dependencies, the latest versions of the packages of the `SpatialData` ecosystem are not available on `conda-forge`. We are waiting for the availability to be unlocked. The latest versions are always available via PyPI.

## Docker

## Docker

A `Dockerfile` is available in the repository; the image that can be built from it contains `spatialdata` (with `torch`), `spatialdata-io` and `spatialdata-plot` (not `napari-spatialdata`)'; the libaries are installed from PyPI.

To build the image, run:

```bash
# this is for Apple Silicon machines, if you are not using such machine you can omit the --build-arg
docker build --build-arg TARGETPLATFORM=linux/arm64 --tag spatialdata .
docker run -it spatialdata
```

We also publish images automatically via GitHub Actions; you can see the [list of available images here](https://github.com/scverse/spatialdata/pkgs/container/spatialdata/versions).

Once you have the image name, you can pull and run it with:

```bash
docker pull ghcr.io/scverse/spatialdata:spatialdata0.3.0_spatialdata-io0.1.7_spatialdata-plot0.2.9
docker run -it ghcr.io/scverse/spatialdata:spatialdata0.3.0_spatialdata-io0.1.7_spatialdata-plot0.2.9
```

<!-- Links -->

[napari-spatialdata]: https://github.com/scverse/napari-spatialdata
[spatialdata-io]: https://github.com/scverse/spatialdata-io
[spatialdata-plot]: https://github.com/scverse/spatialdata-plot
