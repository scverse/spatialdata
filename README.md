![SpatialData banner](https://github.com/scverse/spatialdata/blob/main/docs/_static/img/spatialdata_horizontal.png?raw=true)

# SpatialData: an open and universal framework for processing spatial omics data.

[![Tests][badge-tests]][link-tests]
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/scverse/spatialdata/main.svg)](https://results.pre-commit.ci/latest/github/scverse/spatialdata/main)
[![codecov](https://codecov.io/gh/scverse/spatialdata/branch/main/graph/badge.svg?token=X19DRSIMCU)](https://codecov.io/gh/scverse/spatialdata)
[![documentation badge](https://readthedocs.org/projects/scverse-spatialdata/badge/?version=latest)](https://spatialdata.scverse.org/en/latest/)

SpatialData is a data framework that comprises a FAIR storage format and a collection of python libraries for performant access, alignment, and processing of uni- and multi-modal spatial omics datasets. This repository contains the core spatialdata library. See the links below to learn more about other packages in the SpatialData ecosystem.

-   [spatialdata-io](https://github.com/scverse/spatialdata-io): load data from common spatial omics technologies into spatialdata.
-   [spatialdata-plot](https://github.com/scverse/spatialdata-plot): Static plotting library for spatialdata.
-   [napari-spatialdata](https://github.com/scverse/napari-spatialdata): napari plugin for interactive exploration and annotation of spatialdata.

![SpatialDataOverview](https://github.com/scverse/spatialdata/assets/1120672/cb91071f-12a7-4b8e-9430-2b3a0f65e52f)

-   **The library is currently under review.** We expect there to be changes as the community provides feedback.
-   To get involved in the discussion, or if you need help to get started, you are welcome to join our [`scverse` Zulip chat](https://imagesc.zulipchat.com/#narrow/stream/329057-scverse/topic/segmentation) and our [scverse discourse forum](https://discourse.scverse.org/).
-   The SpatialData storage format is built on top of the [OME-NGFF](https://ngff.openmicroscopy.org/latest/) specification.

## Getting started

Please refer to the [documentation][link-docs]. In particular:

-   [API documentation][link-api].
-   [Design doc][link-design-doc].
-   [Example notebooks][link-notebooks].

## Installation

Check out the docs for more complete [installation instructions](https://spatialdata.scverse.org/en/latest/installation.html). To get started with the "batteries included" installation, you can install via pip:

```bash
pip install "spatialdata[extra]"
```

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

[L Marconato*, G Palla*, KA Yamauchi*, I Virshup*, E Heidari, T Treis, M Toth, R Shrestha, H Vöhringer, W Huber, M Gerstung, J Moore, FJ Theis, O Stegle, bioRxiv, 2023](https://www.biorxiv.org/content/10.1101/2023.05.05.539647v1). \* = equal contribution

<!-- Links -->

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/scverse/spatialdata/issues
[changelog]: https://spatialdata.readthedocs.io/latest/changelog.html
[design doc]: https://scverse-spatialdata.readthedocs.io/en/latest/design_doc.html
[link-docs]: https://spatialdata.scverse.org/en/latest/
[link-api]: https://spatialdata.scverse.org/en/latest/api.html
[link-design-doc]: https://spatialdata.scverse.org/en/latest/design_doc.html
[link-notebooks]: https://spatialdata.scverse.org/en/latest/tutorials/notebooks/notebooks.html
[badge-tests]: https://github.com/scverse/spatialdata/actions/workflows/test_and_deploy.yaml/badge.svg
[link-tests]: https://github.com/scverse/spatialdata/actions/workflows/test_and_deplot.yaml
