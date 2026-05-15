# SpatialData User Guide

Lotte Pollaris, Bishoy Wadie, Friedrich Preusser

## Introduction

The SpatialData user guide gives an overview of the SpatialData data framework and its documentation. Based on questions you might have, this guide points you to the correct tutorials.

If you want to read more about the framework, please have a look at our publication: [Marconato et al.](https://www.nature.com/articles/s41592-024-02212-x)

<details open>
<summary><h2>How can I install SpatialData?</summary>

```
pip install spatialdata
```

This command installs barebone SpatialData.  
For a more detailed description on the installation process including all bells and whistles, see [here](https://spatialdata.scverse.org/en/stable/installation.html).

</details>

<details open>
<summary><h2>I have a SpatialData object—how do I open it and see what's inside?</summary>

```
sdata = sd.read_zarr(“your-dataset.zarr")
```

Please see [this section](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/models1.html#reading-spatialdata-zarr-data) that gives more context about how to read in a new zarr file. For a more detailed tutorial that also explains how to explore the different element types within the SpatialData object, see [here](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/intro.html).

</details>

<details open>
<summary><h2>How do I create a SpatialData object?
</summary>

<details open>
<summary><h3>If you have an existing Squidpy object.</summary>
    
Please have a look at [this tutorial](https://github.com/scverse/spatialdata-notebooks/blob/main/notebooks%2Fexamples%2Fsdata_from_scratch.ipynb).
  
</details>
    
<details open>
<summary><h3>If you want to start from raw files</summary>
    
If you don’t have a SpatialData object or corresponding zarr file, you will have to create a SpatialData object to make use of the SpatialData framework.
You can create a SpatialData object directly from your raw files (e.g. CSVs, cell-gene matrix/cell-proteins matrix, images etc.) by using the [spatialdata-io](https://github.com/scverse/spatialdata-io) library that has reader functions for most spatial omics techniques. See [here](https://spatialdata.scverse.org/projects/io/en/latest/) for a list of currently supported technologies. 
    
For example, if you have data coming from a MERSCOPE®, just use:

```
sdata = spatialdata_io.merscope(
    'merscope_ex/' # path to the folder with MERSCOPE files
    )
```

Please also see [this section](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/models1.html#reader-functions-from-spatialdata-io) for a more technical tutorial with more details on how to create a SpatialData object from raw data using io-readers.

If there is no reader implemented for your data type, please refer to [this section](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/models1.html#construct-a-spatialdata-object-from-scratch) to learn about building SpatialData objects from scratch.

</details>
    
<details open>
<summary><h3>If you want to play around with existing data.</summary>
    
Please see [here](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/datasets/README.html) for a repository of demo datasets (already available as .zarr) generated with different spatial omics technologies.
  
</details>
</details>

<details open>
<summary><h2>How can I use SpatialData to analyze my data?</summary>

SpatialData is **not** intended to analyze data itself, but to provide data infrastucture to efficiently manipulate spatial omics data. If you want to learn more about analyzing spatial omics data, please browse [these other tutorials](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks.html#external-tutorials) for a demo of how various analysis packages work with SpatialData.

If you want to see how SpatialData and Squidpy can interact, please have a look [here](https://spatialdata.scverse.org/en/latest/tutorials/notebooks/notebooks/examples/squidpy_integration.html).

</details>

<details open>
<summary><h2>How do I align my images?</summary>
Sometimes, you have multiple images containing information, but they have different pixels size/are rotated/are moved. For instance, you may have a multi-omics dataset with two images, each representing a different modality, but you need to align them within a single object while preserving their respective molecular signals. Here, we explain how you can spatially align(=register) these images. NOTE: SpatialData only allows for affine(=linear) transformations to map images onto each other.

Do you know how to align your images (i.e. Do you have a transformation matrix)?

No → Check out our [Landmark annotation tutorial](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/alignment_using_landmarks.html)

Yes → Check out our [Transformation/coordinate system tutorial](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/transformations.html)

</details>

<details open>
<summary><h2>How do I annotate my data?</summary> 
    
 Annotation can have a lot of meanings. You might want to annotate specific regions in your tissue as tumor, add more details about your cell shapes, group together specific transcripts, include celltype annotations or cell sizes for your cells etc.   
<details open>
<summary> <h3> How can I spatially annotate regions in my data? </summary>
    
This is possible within the SpatialData framework, making use of napari, like explained [in this tutorial](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/napari_rois.html).

</details>
<details open>
<summary> <h3> I have annotated regions in an external tool, how do I add them to SpatialData?</summary>
    
If the annotated regions are saved in a geojson, you can add them as follows:      
```    
from spatialdata.models import ShapesModel
sdata['very_interesting_regions']=ShapesModel.parse('path_to_geojson')
```
For more details, including information on how to add annotations for these regions, please have a look at [this tutorial](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/tables.html).

</details>
    
<details open>
    
    
<summary> <h3> I have cells in my dataset, how do I annotate them? (usage of AnnData)</summary>

One of the most obvious things to do for spatial omics data is to annotate cells using the [AnnData](https://anndata.readthedocs.io/en/stable/) format (called tables in SpatialData). These tables can contain count/intensity data, all types of annotations, and make it possible to make use of [scanpy](https://scanpy.readthedocs.io/en/stable/) functionality (normalization/clustering/DE calculation).  
If you want more technical details on how to create a table from scratch to annotate your shapes/labels/points, you can have a look [here](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/models2.html#tables).

</details>
    
    
    
</details>
    
<details open>
<summary><h2>How do I analyze a spatial subset of my data?
    </summary>

Eventually, you might want to know which cells are present in which region (e.g. 'Are immune cells infiltrating the tumor core?').
Therefore, subsampling your data based on spatial coordinates can be helpful. Other examples would be for instance if you want to target your analysis to a specific subregion of the tissue (e.g. tumor core), if you have multiple tissues on one slide, if you want to start small, or if you want to only focus on regions with good tissue quality.

<details open>
<summary><h3>I want to annotate a specific region myself.</summary>
    
Please see the previous section ('How do I annotate my data?') and then specifically the section about spatially annotating regions. This will guide you through the process of extracting coordinates, which will be used for spatially subsetting the data. Next, see below.
</details>

<details open>
<summary><h3>I know the coordinates of the region I want to subsample.</summary>
    
You will need to perform a spatial query, which filters the data based on spatial coordinates. Please see [this tutorial](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/spatial_query.html).
    
</details>

</details>
    
    
<details open>
<summary><h2>How do I visualize my data?</summary>

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

If you run into scalability issues with your plotting, or everything feels very slow, [this tutorial](https://spatialdata.scverse.org/en/latest/tutorials/notebooks/notebooks/examples/speed_up_illustration.html) might help you out!

</details>

<details open>
<summary><h2>How do I summarize/aggregate my data?</summary>

You have annotated your object with regions or labels and want to determine which cells or transcripts are present in each area. For example, you may be interested in analyzing the composition of cell types within each region or generating a pseudobulk profile per region or label. Or maybe you just want to know your gene counts per cell. In SpatialData terminology, this corresponds to the **aggregate** function.

The method of aggregation depends on the type of elements in your SpatialData object. Learn how to do it in our [Integrate/aggregate signals across spatial layers tutorial](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/aggregation.html).

</details>
<details open>
    
<summary> <h2> SpatialData looks amazing, I want to learn more. Give me all the technical details!</summary>
    
Sure, no problem! There is much more to learn about SpatialData.

You want to learn how to combine SpatialData and deep learning? We got you covered [here](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/densenet.html).

If you want to create a labels layer from your shapes (rasterize), a shapes layer from you labels (vectorize, or convert your spot-based data to a labels layer [this tutorial](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/labels_shapes_interchangeability.html) might help you out). This is mainly useful to unlock specific functionalities, or speed up calculations.

Finally here is a more [advanced technical tutorial on transformations](https://spatialdata.scverse.org/en/latest/tutorials/notebooks/notebooks/examples/transformations_advanced.html#other-technical-topics).

</details>
    
<details open>
    
<summary> <h2> I am sold, how can I get in touch and contribute to SpatialData?

</summary>

[Here](https://spatialdata.scverse.org/en/stable/contributing.html) is a guide that explains how to contribute to SpatialData as a developer. If you would like to contribute to satellite projects, such as `spatialdata-io`, `spatialdata-plot` etc, please consult the contribution guides also in those repositories.
For larger contributions, and especially regarding new APIs or new design choices, please before writing any code get in touch with us! We are happy to talk via GitHub, via chat or with a call. [In the README](https://github.com/scverse/spatialdata/#contact) you can find multiple ways for reaching out.

</details>
