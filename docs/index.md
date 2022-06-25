# spatialdata

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/workflow/status/giovp/spatialdata/Test/main
[link-tests]: https://github.com/scverse/spatialdata.git/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/spatialdata

Spatial data format.

## Getting started

-   Check out the [API documentation](api.md)
-   Read about `SpatialData`'s sister classes: [AnnData](https://anndata.readthedocs.io) and [MuData](https://mudata.readthedocs.io).
-   Read about the on disk format: [OME-NGFF](https://github.com/ome/ngff)

## Installation

You need to have Python 3.8 or newer installed on your system. If you don't have
Python installed, we recommend installing `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`\_.

There are several alternative options to install spatialdata:

<!--
1) Install the latest release of `spatialdata` from `PyPI <https://pypi.org/project/spatialdata/>`_:

```bash
pip install spatialdata
```
-->

To install the latest development version:

```bash
pip install git+https://github.com/scverse/spatialdata.git@main
```

## Release notes

See the [changelog](changelog.md).

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

> t.b.a

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/scverse/spatialdata/issues
[link-docs]: https://spatialdata.readthedocs.io/latest/
[changelog]: https://spatialdata.readthedocs.io/latest/changelog.html

```{toctree}
:hidden: true
:maxdepth: 1

api.md
changelog.md
developer_docs.md
references.md

notebooks/example
```
