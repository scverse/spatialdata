```{eval-rst}
.. image:: _static/img/spatialdata_horizontal.png
  :class: dark-light p-2
  :alt: SpatialData banner
```

# An open and universal framework for processing spatial omics data.

SpatialData is a data framework that comprises a FAIR storage format and a collection of python libraries for performant access, alignment, and processing of uni- and multi-modal spatial omics datasets. This page provides documentation on how to install, use, and extend the core `spatialdata` library. See the links below to learn more about other packages in the SpatialData ecosystem.

- `spatialdata-io`: load data from common spatial omics technologies into `spatialdata` ([repository][spatialdata-io-repo], [documentation][spatialdata-io-docs]).
- `spatialdata-plot`: Static plotting library for `spatialdata` ([repository][spatialdata-plot-repo], [documentation][spatialdata-plot-docs]).
- `napari-spatialdata-repo`: napari plugin for interactive exploration and annotation of `spatialdata` ([repository][napari-spatialdata-repo], [documentation][napari-spatialdata-docs]).

Please see our publication {cite}`marconatoSpatialDataOpenUniversal2024` for citation and to learn more.

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

```{eval-rst}
.. note::
   This library is currently under active development. We may make changes to the API between versions as the community provides feedback. To ensure reproducibility, please make note of the version you are developing against.
```

```{eval-rst}
.. card:: Installation
    :link: installation
    :link-type: doc

    Learn how to install ``spatialdata``.

.. card:: User Guide
    :link: user_guide
    :link-type: doc

    Navigate your way through ``spatialdata`` tutorials.

.. card:: Tutorials
    :link: tutorials/notebooks/notebooks
    :link-type: doc

    Learn how to use ``spatialdata`` with hands-on examples.

.. card:: API
    :link: api
    :link-type: doc

    Find a detailed documentation of ``spatialdata``.

.. card:: Datasets
    :link: tutorials/notebooks/datasets/README
    :link-type: doc

    Example datasets from 8 different technologies.

.. card:: Design document
    :link: design_doc
    :link-type: doc

    Learn about the design approach behind ``spatialdata``.

.. card:: Contributing
    :link: contributing
    :link-type: doc

    Learn how to contribute to ``spatialdata``.

```

```{toctree}
:hidden: true
:maxdepth: 1

installation.md
user_guide.md
api.md
tutorials/notebooks/notebooks.md
tutorials/notebooks/datasets/README.md
glossary.md
design_doc.md
contributing.md
changelog.md
references.md
```

<!-- Links -->

[napari-spatialdata-repo]: https://github.com/scverse/napari-spatialdata
[spatialdata-io-repo]: https://github.com/scverse/spatialdata-io
[spatialdata-plot-repo]: https://github.com/scverse/spatialdata-plot
[napari-spatialdata-docs]: https://spatialdata.scverse.org/projects/napari/en/stable/notebooks/spatialdata.html
[spatialdata-io-docs]: https://spatialdata.scverse.org/projects/io/en/stable/
[spatialdata-plot-docs]: https://spatialdata.scverse.org/projects/plot/en/stable/api.html
