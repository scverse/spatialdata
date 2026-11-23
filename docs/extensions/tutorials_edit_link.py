"""Point "Edit on GitHub" on tutorial pages at scverse/spatialdata-tutorials.

The tutorials are a git submodule at ``docs/tutorials/notebooks``, so the default
edit link (``spatialdata/edit/<ref>/docs/tutorials/notebooks/...``) is a 404.
Edits go to the tutorials' ``main``, not to the pinned commit.
Notebooks open in github.dev, which edits them as cells; GitHub's own editor shows raw JSON.
"""

from sphinx.application import Sphinx

PREFIX = "tutorials/notebooks/"
REPO = "scverse/spatialdata-tutorials"


def _edit_link(app: Sphinx, pagename: str, templatename: str, context: dict, doctree: object) -> None:
    if not pagename.startswith(PREFIX):
        return
    if context.get("page_source_suffix") == ".ipynb":
        base = f"https://github.dev/{REPO}/blob/main/"
    else:
        base = f"https://github.com/{REPO}/edit/main/"
    # pydata-sphinx-theme renders this template lazily; it must contain `file_name`.
    context["edit_page_provider_name"] = "GitHub"
    context["edit_page_url_template"] = base + "{{ file_name | replace('" + PREFIX + "', '', 1) }}"


def setup(app: Sphinx) -> dict[str, bool]:
    app.connect("html-page-context", _edit_link)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
