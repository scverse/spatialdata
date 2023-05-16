```{eval-rst}
.. image:: _static/img/spatialdata_horizontal.png
  :class: dark-light p-2
  :alt: SpatialData banner
```

# An open and universal framework for processing spatial omics data.

SpatialData is a data framework that comprises a FAIR storage format and a collection of python libraries for performant access, alignment, and processing of uni- and multi-modal spatial omics datasets. This page provides documentation on how to install, use, and extend the core `spatialdata` library. See the links below to learn more about other packages in the SpatialData ecosystem.

-   [spatialdata-io][]: load data from common spatial omics technologies into `spatialdata`.
-   [spatialdata-plot][]: Static plotting library for `spatialdata`.
-   [napari-spatialdata][]: napari plugin for interactive exploration and annotation of `spatialdata`.

```{eval-rst}
.. note::
   This library is currently under active development. We may make changes to the API between versions as the community provides feedback. To ensure reproducibility, please make note of the version you are developing against.
```

```{eval-rst}
.. card:: Installation
    :link: installation
    :link-type: doc

    Learn how to install ``spatialdata``.

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
api.md
tutorials/notebooks/notebooks.md
tutorials/notebooks/datasets/README.md
design_doc.md
contributing.md
changelog.md
references.md
```

<!-- Links -->

[napari-spatialdata]: https://github.com/scverse/napari-spatialdata
[spatialdata-io]: https://github.com/scverse/spatialdata-io
[spatialdata-plot]: https://github.com/scverse/spatialdata-plot
