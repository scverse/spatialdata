import sys
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path

from scverse_doc.config import DEFAULTS
from scverse_doc.registry import intersphinx

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "extensions"))
sys.path.insert(0, str(HERE / "tutorials" / "notebooks" / "extensions"))

info = metadata("spatialdata")
project = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, {author}"
version = info["Version"]
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
source_repository = urls["Source"]
release = info["Version"]

bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
templates_path = ["_templates"]

extensions = [
    "scverse_doc",
    "sphinx.ext.linkcode",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "IPython.sphinxext.ipython_console_highlighting",
    "scverse_misc.sphinx_ext",
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
    *[p.stem for p in (HERE / "tutorials" / "notebooks" / "extensions").glob("*.py")],
]

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
}
autodoc_process_signature = True
myst_url_schemes = ("http", "https", "mailto")
typehints_defaults = "braces"
always_use_bars_union = True  # use `|` instead of `Union` in types even when building with Python ≤3.14

intersphinx_mapping = intersphinx() | {
    "annsel": ("https://annsel.readthedocs.io/en/latest/", None),
    "dask": ("https://docs.dask.org/en/latest/", None),
    "datashader": ("https://datashader.org/", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "ome_zarr": ("https://ome-zarr.readthedocs.io/en/stable/", None),
    "pooch": ("https://www.fatiando.org/pooch/latest/", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "upath": ("https://universal-pathlib.readthedocs.io/en/latest/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "xrspatial": ("https://xarray-spatial.readthedocs.io/en/stable/", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable/", None),
}

xref_overrides = {
    # private paths of public objects, as they appear in type annotations
    "numpy._typing._array_like.NDArray": ("py:data", "numpy.typing.NDArray"),
    "zarr.storage._fsspec.FsspecStore": "zarr.storage.FsspecStore",
    "zarr.storage._local.LocalStore": "zarr.storage.LocalStore",
    # wrong names in the tutorials notebooks (scverse/spatialdata-tutorials); remove once fixed there
    "spatialdata.ops.get_transformation_between_coordinate_systems": (
        "py:func",
        "spatialdata.transformations.get_transformation_between_coordinate_systems",
    ),
    "spatialdata.transformations.Transformation.to_affine_matrix": (
        "py:meth",
        "spatialdata.transformations.BaseTransformation.to_affine_matrix",
    ),
}

exclude_patterns = [
    *DEFAULTS.exclude_patterns,
    "tutorials/notebooks/index.md",
    "tutorials/notebooks/README.md",
    "tutorials/notebooks/references.md",
    "tutorials/notebooks/notebooks/paper_reproducibility/*",
    "tutorials/notebooks/notebooks/developers_resources/storage_format/*.ipynb",
    "tutorials/notebooks/notebooks/developers_resources/storage_format/Readme.md",
    "tutorials/notebooks/notebooks/examples/technology_stereoseq.ipynb",
    "tutorials/notebooks/notebooks/examples/technology_curio.ipynb",
    "tutorials/notebooks/notebooks/examples/technology_cosmx.ipynb",
    "tutorials/notebooks/notebooks/examples/stereoseq_data/*",
]

# no solution yet (7.4.7); using the workaround shown here: https://github.com/sphinx-doc/sphinx/issues/12589

ogp_image = "_static/img/spatialdata_horizontal.png"

suppress_warnings = [
    "autosummary.import_cycle",
]

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_title = project
html_logo = "_static/img/spatialdata_horizontal.svg"

html_theme_options = {"show_toc_level": 4}

nitpick_ignore = [
    # Add an entry here when a missing link is outside our control.
    ("py:class", "igraph.Graph"),
    # upstream packages without an intersphinx inventory
    ("py:class", "multiscale_spatial_image.to_multiscale.to_multiscale.Methods"),
    # not in datashader's API docs (only the concrete reductions are)
    ("py:class", "datashader.reductions.Reduction"),
    # private classes that appear in public signatures and base classes, but are not part of the API docs
    ("py:class", "spatialdata._core._elements.Images"),
    ("py:class", "spatialdata._core._elements.Labels"),
    ("py:class", "spatialdata._core._elements.Points"),
    ("py:class", "spatialdata._core._elements.Shapes"),
    ("py:class", "spatialdata._core._elements.Tables"),
    ("py:class", "spatialdata._core.spatialdata.QueryManager"),
    ("py:class", "spatialdata._io.format.CoordinateMixinV01"),
    ("py:class", "spatialdata._io.format.PointsAttrsMixinV01"),
    ("py:attr", "spatialdata._io._utils.BadFileHandleMethod.ERROR"),
    ("py:attr", "spatialdata._io._utils.BadFileHandleMethod.WARN"),
    ("py:class", "spatialdata.models.models.RasterSchema"),
    ("py:type", "spatialdata._types.JSONValue"),
]
