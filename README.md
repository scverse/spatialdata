![SpatialData banner](https://github.com/scverse/spatialdata/blob/main/docs/_static/img/spatialdata_horizontal.png?raw=true)

# SpatialData: an open and universal framework for processing spatial omics data.

[![Tests][badge-tests]][link-tests]
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/scverse/spatialdata/main.svg)](https://results.pre-commit.ci/latest/github/scverse/spatialdata/main)
[![codecov](https://codecov.io/gh/scverse/spatialdata/branch/main/graph/badge.svg?token=X19DRSIMCU)](https://codecov.io/gh/scverse/spatialdata)
[![documentation badge](https://readthedocs.org/projects/scverse-spatialdata/badge/?version=latest)](https://spatialdata.scverse.org/en/latest/)
[![DOI](https://zenodo.org/badge/487366481.svg)](https://zenodo.org/badge/latestdoi/487366481)

SpatialData is a data framework that comprises a FAIR storage format and a collection of python libraries for performant access, alignment, and processing of uni- and multi-modal spatial omics datasets. This repository contains the core spatialdata library. See the links below to learn more about other packages in the SpatialData ecosystem.

-   [spatialdata-io](https://github.com/scverse/spatialdata-io): load data from common spatial omics technologies into spatialdata.
-   [spatialdata-plot](https://github.com/scverse/spatialdata-plot): Static plotting library for spatialdata.
-   [napari-spatialdata](https://github.com/scverse/napari-spatialdata): napari plugin for interactive exploration and annotation of spatial data.

[//]: # "numfocus-fiscal-sponsor-attribution"

The spatialdata project uses a [consensus based governance model](https://scverse.org/about/roles/) and is fiscally sponsored by [NumFOCUS](https://numfocus.org/). Consider making a [tax-deductible donation](https://numfocus.org/donate-to-scverse) to help the project pay for developer time, professional services, travel, workshops, and a variety of other needs.

The spatialdata project also received support by the Chan Zuckerberg Initiative.

<div align="center">
  <a href="https://numfocus.org/project/scverse">
    <img height="60px" 
         src="https://raw.githubusercontent.com/numfocus/templates/master/images/numfocus-logo.png" 
         align="center">
  </a>
</div>
<br>

![SpatialDataOverview](https://github.com/scverse/spatialdata/assets/1120672/cb91071f-12a7-4b8e-9430-2b3a0f65e52f)

-   **The library is currently under review.** We expect there to be changes as the community provides feedback. We have an announcement channel for communicating these changes, please see the contact section below.
-   The SpatialData storage format is built on top of the [OME-NGFF](https://ngff.openmicroscopy.org/latest/) specification.

## Getting started

Please refer to the [documentation][link-docs]. In particular:

-   [API documentation][link-api].
-   [Design doc][link-design-doc].
-   [Example notebooks][link-notebooks].

Another useful resource to get started is the source code of the [`spatialdata-io`](https://github.com/scverse/spatialdata-io) package, which shows example of how to read data from common technologies.

## Installation

Check out the docs for more complete [installation instructions](https://spatialdata.scverse.org/en/latest/installation.html). To get started with the "batteries included" installation, you can install via pip:

```bash
pip install "spatialdata[extra]"
```

Note: if you are using a Mac with an M1/M2 chip, please follow the installation instructions.

## Limitations

-   Windows support. Currently the framework is tested on Linux and macOS machines, not Windows machines. Users have reported bugs in read/write operations (27 March 2024).

## Contact

To get involved in the discussion, or if you need help to get started, you are welcome to use the following options.

-   <ins>Chat</ins> via [`scverse` Zulip](https://imagesc.zulipchat.com/#narrow/stream/329057-scverse/topic/segmentation) (public or 1 to 1).
-   <ins>Forum post</ins> in the [scverse discourse forum](https://discourse.scverse.org/).
-   <ins>Bug report/feature request</ins> via the [GitHub issue tracker][issue-tracker].
-   <ins>Zoom call</ins> as part of the SpatialData Community Meetings, held every 2 weeks on Thursday, [schedule here](https://hackmd.io/enWU826vRai-JYaL7TZaSw).

Finally, especially relevant for for developers that are building a library upon `spatialdata`, please follow this channel for:

-   <ins>Announcements</ins> on new features and important changes [Zulip](https://imagesc.zulipchat.com/#narrow/stream/329057-scverse/topic/spatialdata.20announcements).

## Citation

Marconato, L., Palla, G., Yamauchi, K.A. et al. SpatialData: an open and universal data framework for spatial omics. Nat Methods (2024). https://doi.org/10.1038/s41592-024-02212-x

<!-- Links -->

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/scverse/spatialdata/issues
[changelog]: https://spatialdata.readthedocs.io/latest/changelog.html
[design doc]: https://scverse-spatialdata.readthedocs.io/en/latest/design_doc.html
[link-docs]: https://spatialdata.scverse.org/en/latest/
[link-api]: https://spatialdata.scverse.org/en/latest/api.html
[link-design-doc]: https://spatialdata.scverse.org/en/latest/design_doc.html
[link-notebooks]: https://spatialdata.scverse.org/en/latest/tutorials/notebooks/notebooks.html
[badge-tests]: https://github.com/scverse/spatialdata/actions/workflows/test.yaml/badge.svg
[link-tests]: https://github.com/scverse/spatialdata/actions/workflows/test.yaml
