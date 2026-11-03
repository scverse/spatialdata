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
    "datatree": ("https://datatree.readthedocs.io/en/latest/", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
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

nitpicky = False  # TODO: solve upstream, then set back to True to warn about broken links.
# no solution yet (7.4.7); using the workaround shown here: https://github.com/sphinx-doc/sphinx/issues/12589
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
]
