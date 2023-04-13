# Work in progress ⚠

-   **The library is not ready.** We aim at a beta release in the following months. If interested in a demo/early beta, please reach out to us.
-   To get involved in the discussion you are welcome to join our Zulip workspace and/or our community meetings every second week; [more info here](https://imagesc.zulipchat.com/#narrow/stream/329057-scverse/topic/SpatialData.20meetings).
-   Links to the OME-NGFF specification: [0.4](https://ngff.openmicroscopy.org/latest/), [0.5-dev (tables)](https://github.com/ome/ngff/pull/64), [0.5-dev (transformations and coordinate systems)](https://github.com/ome/ngff/pull/138)

# spatialdata

[![Tests][badge-tests]][link-tests]
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/scverse/spatialdata/main.svg)](https://results.pre-commit.ci/latest/github/scverse/spatialdata/main)
[![codecov](https://codecov.io/gh/scverse/spatialdata/branch/main/graph/badge.svg?token=X19DRSIMCU)](https://codecov.io/gh/scverse/spatialdata)
[![DOI](https://zenodo.org/badge/487366481.svg)](https://zenodo.org/badge/latestdoi/487366481)

[badge-tests]: https://github.com/scverse/spatialdata/actions/workflows/test.yaml/badge.svg
[link-tests]: https://github.com/scverse/spatialdata/actions/workflows/test.yaml

## Getting started

Please refer to the [documentation][link-docs]. In particular:

-   [API documentation][link-api].
-   [Design doc][link-design-doc].
-   [Example notebooks][link-notebooks].

## Installation

Check out the docs for more complete installation instructions. For now you can install `spatialdata` with:

```bash
pip install git+https://github.com/scverse/spatialdata.git@main
```

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

> t.b.a

<!-- Links -->

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/scverse/spatialdata/issues
[changelog]: https://spatialdata.readthedocs.io/latest/changelog.html
[design doc]: https://scverse-spatialdata.readthedocs.io/en/latest/design_doc.html
[link-docs]: https://spatialdata.scverse.org/en/latest/
[link-api]: https://spatialdata.scverse.org/en/latest/api.html
[link-design-doc]: https://spatialdata.scverse.org/en/latest/design_doc.html
[link-notebooks]: https://spatialdata.scverse.org/en/latest/tutorials/notebooks/notebooks.html

<img src='https://github.com/giovp/spatialdata-sandbox/raw/main/graphics/overview.png'/>

### Remove once contributing docs are in place

link repo to notebok repo:

-   git submodule add https://github.com/scverse/spatialdata-notebooks notebooks
-   fetch and pull to update from main directly in the submodule
