import shutil
import sys
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path, PurePosixPath

from sphinxcontrib import katex

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "extensions"))
sys.path.insert(0, str(HERE / "tutorials" / "notebooks" / "extensions"))


# -- Project information -----------------------------------------------------

info = metadata("spatialdata")
project = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, {author}"
version = info["Version"]
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
repository_url = urls["Source"]

release = info["Version"]

bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
templates_path = ["_templates"]
needs_sphinx = "4.0"

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "scverse",
    "github_repo": project,
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# -- General configuration ---------------------------------------------------

extensions = [
    "git_ref",  # needs to be before scanpydoc.rtd_github_links
    "scanpydoc.rtd_github_links",  # needs to be before sphinx.ext.linkcode
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.katex",
    "sphinx_autodoc_typehints",
    "sphinx_design",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinxext.opengraph",
    "scverse_misc.sphinx_ext",
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
    *[p.stem for p in (HERE / "tutorials" / "notebooks" / "extensions").glob("*.py")],
]

rtd_links_prefix = PurePosixPath("src")

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
}

autosummary_generate = True
autodoc_process_signature = True
autodoc_member_order = "groupwise"
default_role = "literal"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
myst_heading_anchors = 6  # create anchors for h1-h6
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_url_schemes = ("http", "https", "mailto")
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True
typehints_defaults = "braces"
always_use_bars_union = True  # use `|` instead of `Union` in types even when building with Python ≤3.14

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "annsel": ("https://annsel.readthedocs.io/en/latest/", None),
    "dask": ("https://docs.dask.org/en/latest/", None),
    "datatree": ("https://datatree.readthedocs.io/en/latest/", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
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


# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_title = project
html_logo = "_static/img/spatialdata_horizontal.png"

html_theme_options = {
    "repository_url": repository_url,
    "use_repository_button": True,
    "path_to_docs": "docs/",
    "navigation_with_keys": False,
    "show_toc_level": 4,
}

pygments_style = "default"
katex_prerender = shutil.which(katex.NODEJS_BINARY) is not None

nitpick_ignore = [
    # Add an entry here when a missing link is outside our control.
    ("py:class", "igraph.Graph"),
]
