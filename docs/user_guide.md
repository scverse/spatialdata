# How-to guides

::::{card}
:class-card: sd-mb-4 sd-shadow-sm

Authors: {bdg-secondary}`Lotte Pollaris` {bdg-secondary}`Bishoy Wadie` {bdg-secondary}`Friedrich Preusser`
::::

Looking to do something specific with SpatialData? Check out the FAQs below for quick guides, code snippets and links to in-depth tutorials.

If you want to read more about the framework, please have a look at our publication: [Marconato et al.](https://www.nature.com/articles/s41592-024-02212-x)

## Getting started

:::{dropdown} How can I install SpatialData?
:open:
```
pip install spatialdata
```

This command installs barebone SpatialData.
For a more detailed description of the installation process including all bells and whistles, see the [installation guide](https://spatialdata.scverse.org/en/stable/installation.html).

:::

:::{dropdown} I have a SpatialData object, how do I open it and see what's inside?
```
sdata = sd.read_zarr("your-dataset.zarr")
```

For more details, see [the tutorial section on reading zarr data](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/models1.html#reading-spatialdata-zarr-data). Additionaly, see [the introductory tutorial on SpatialData](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/intro.html) to understand how to explore the different element types within a SpatialData object.

:::

:::::{dropdown} How do I create a SpatialData object?


::::{tab-set}
:::{tab-item} If you have an existing Squidpy object.

Checkout [the tutorial on creating a SpatialData object from scratch](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/sdata_from_scratch.html).

:::

:::{tab-item} If you want to start from raw files

If you don’t have a SpatialData object or corresponding zarr data on disk, you will have to create a SpatialData object to make use of the SpatialData framework.
You can create a SpatialData object directly from your raw files (e.g. CSVs, cell-gene matrix/cell-proteins matrix, images etc.) by using the [spatialdata-io](https://github.com/scverse/spatialdata-io) library that has reader functions for most spatial omics techniques. See the [list of technologies currently supported by spatialdata-io](https://spatialdata.scverse.org/projects/io/en/latest/).

For example, if you have data coming from a MERSCOPE®, just use:

```
sdata = spatialdata_io.merscope(
    'merscope_ex/' # path to the folder with MERSCOPE files
    )
```

For more technical details on creating a SpatialData object from raw data, see [the tutorial section on reader functions from spatialdata-io](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/models1.html#reader-functions-from-spatialdata-io).

If there is no reader implemented for your data type, see [the tutorial section on constructing a SpatialData object from scratch](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/models1.html#construct-a-spatialdata-object-from-scratch).


:::

::::

:::::

:::{dropdown} If you want to play around with existing data.

Browse our [collection of demo datasets](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/datasets/README.html) (already available as .zarr), generated with different spatial omics technologies.

:::


## Preparing your data

:::{dropdown} How do I align my images?
:open:

Sometimes, you have multiple images containing information, but they have different pixels size/are rotated/are moved. For instance, you may have a multi-omics dataset with two images, each representing a different modality, but you need to align them within a single object while preserving their respective molecular signals. Here, we explain how you can spatially align(=register) these images. NOTE: SpatialData only allows for affine(=linear) transformations to map images onto each other.

Do you know how to align your images (i.e. Do you have a transformation matrix)?

No → Check out our [Landmark annotation tutorial](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/alignment_using_landmarks.html)

Yes → Check out our [Transformation/coordinate system tutorial](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/transformations.html)

:::

:::{dropdown} How do I annotate my data?

Annotation can have a lot of meanings. You might want to annotate specific regions in your tissue as tumor, add more details about your cell shapes, group together specific transcripts, include celltype annotations or cell sizes for your cells etc.

:::

:::{dropdown} How can I interactively annotate regions in my data?

This is possible within the SpatialData framework, making use of napari, as explained in the [tutorial on ROI annotation with Napari](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/napari_rois.html).

:::

:::{dropdown} I have annotated regions in an external tool, how do I add them to SpatialData?

If the annotated regions are saved in a geojson, you can add them as follows:

```
from spatialdata.models import ShapesModel
sdata['very_interesting_regions']=ShapesModel.parse('path_to_geojson')
```

For more details, including information on how to add annotations for these regions, see the [tables tutorial](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/tables.html).

:::

:::{dropdown} I have cells in my dataset, how do I annotate them? (usage of AnnData)

One of the most obvious things to do for spatial omics data is to annotate cells using the [AnnData](https://anndata.readthedocs.io/en/stable/) format (called tables in SpatialData). These tables can contain count/intensity data, all types of annotations, and make it possible to make use of [scanpy](https://scanpy.readthedocs.io/en/stable/) functionality (normalization/clustering/DE calculation).
For more technical details on how to create a table from scratch to annotate your shapes/labels/points, see [tutorial section on how tables are modeled in SpatialData](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/models2.html#tables).

:::


:::{dropdown} How do I analyze a spatial subset of my data?


Eventually, you might want to know which cells are present in which region (e.g. 'Are immune cells infiltrating the tumor core?').
Therefore, subsampling your data based on spatial coordinates can be helpful. Other examples would be for instance if you want to target your analysis to a specific subregion of the tissue (e.g. tumor core), if you have multiple tissues on one slide, if you want to start small, or if you want to only focus on regions with good tissue quality.
:::

:::{dropdown} I want to annotate a specific region myself.

Please see the previous section ('How do I annotate my data?') and then specifically the section about interactively annotating regions. This will guide you through the process of extracting coordinates, which will be used for spatially subsetting the data. Next, see below.
:::

:::{dropdown} I know the coordinates of the region I want to subsample.

You will need to perform a spatial query, which filters the data based on spatial coordinates. See the [spatial query tutorial](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/spatial_query.html).

:::

:::{dropdown} How do I summarize/aggregate my data?

You have annotated your object with regions or labels and want to determine which cells or transcripts are present in each area. For example, you may be interested in analyzing the composition of cell types within each region or generating a pseudobulk profile per region or label. Or maybe you just want to know your gene counts per cell. In SpatialData terminology, this corresponds to the **aggregate** function.

The method of aggregation depends on the type of elements in your SpatialData object. Learn how to do it in our [Integrate/aggregate signals across spatial layers tutorial](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/aggregation.html).

:::

## Visualization and Analysis

:::{dropdown} How do I visualize my data?
:open:

To make data visualization easier for each layer in your SpatialData object, we offer [spatialdata-plot](https://github.com/scverse/spatialdata-plot), a package that enhances SpatialData’s functionality for intuitive and detailed plotting.

The standard format for plotting a specific layer follows this syntax:

```
sdata.pl.render_<element_type>("<element_name>").pl.show()
```

You can also chain different plots. For example, if you want to plot regions of interest (ROIs) and/or cell outlines (i.e. shapes) on top of your image:

```
(
    sdata
    .pl.render_image(
        "image_layer_name"
    )
    .pl.render_shapes(
        "shape_layer_name"
    )
    .pl.show()
)
```

For more details and examples on visualizing technology-specific datasets, explore our [technology-specific tutorials](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks.html#technology-specific), which includes examples for various technologies (e.g., Visium-HD, Xenium, etc.). Keep in mind that the visualization methods demonstrated in these tutorials are broadly applicable and not limited to a specific technology.

If you run into scalability issues with your plotting, or everything feels very slow, [the tutorial on speeding up plotting](https://spatialdata.scverse.org/en/latest/tutorials/notebooks/notebooks/examples/speed_up_illustration.html) might help you out!

:::

:::{dropdown} How can I use SpatialData to analyze my data?

SpatialData is **not** intended to analyze data itself, but to provide data infrastructure to efficiently manipulate spatial omics data. If you want to learn more about analyzing spatial omics data, please browse the [documentation section on extenal tools](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks.html#external-tutorials) for demos of how various analysis packages work with SpatialData.

To see how SpatialData and Squidpy interact, see the [Squidpy integration tutorial](https://spatialdata.scverse.org/en/latest/tutorials/notebooks/notebooks/examples/squidpy_integration.html).

:::


## Going further

:::{dropdown} SpatialData looks amazing, I want to learn more. Give me all the technical details!
:open:

Sure, no problem! There is much more to learn about SpatialData.

You want to learn how to combine SpatialData and deep learning? Checkout the [tutorial on deep learning with DenseNet](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/densenet.html).

If you want to create a labels layer from your shapes (rasterize), a shapes layer from your labels (vectorize), or convert your spot-based data to a labels layer, see the [labels/shapes interchangeability tutorial](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/labels_shapes_interchangeability.html). This is mainly useful to unlock specific functionalities, or speed up calculations.

Finally, here is a more [advanced technical tutorial on transformations](https://spatialdata.scverse.org/en/latest/tutorials/notebooks/notebooks/examples/transformations_advanced.html#other-technical-topics).

:::

:::{dropdown} I am sold, how can I get in touch and contribute to SpatialData?
:open:


Thank you for your interest in contributing to SpatialData! The [contributing guide](https://spatialdata.scverse.org/en/stable/contributing.html) explains how to contribute to SpatialData as a developer. If you would like to contribute to satellite projects, such as `spatialdata-io`, `spatialdata-plot` etc, please consult the contribution guides also in those repositories.
For larger contributions, and especially regarding new APIs or new design choices, please before writing any code get in touch with us! We are happy to talk via GitHub, via chat or with a call. The [contact section of the README](https://github.com/scverse/spatialdata/#contact) lists multiple ways of reaching out.

:::
