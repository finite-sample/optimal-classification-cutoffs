"""Sphinx configuration — fleet standard via py-canon, plus this repo's extras."""

from py_canon.sphinx import configure

configure(globals())

# myst-nb renders the notebooks under docs/examples/ and subsumes myst-parser,
# which must not be loaded alongside it. nbsphinx was the previous renderer and
# needed a pandoc binary that the shared docs workflow does not install.
extensions = [e for e in extensions if e != "myst_parser"] + ["myst_nb"]  # noqa: F821

# The notebooks are committed with their outputs; re-running them on every docs
# build costs minutes and makes the build depend on matplotlib rendering.
nb_execution_mode = "off"

# autosummary_generate writes stubs for every autosummary directive; the API
# pages here list members explicitly instead, and the unwritten stubs otherwise
# surface as warnings under the workflow's -W.
autosummary_generate = False

intersphinx_mapping.update(  # noqa: F821
    {
        "numpy": ("https://numpy.org/doc/stable/", None),
        "scipy": ("https://docs.scipy.org/doc/scipy/", None),
        "sklearn": ("https://scikit-learn.org/stable/", None),
    }
)

html_title = "Optimal Classification Cutoffs"
html_theme_options = {
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/finite-sample/optimal-classification-cutoffs/",
    "source_branch": "master",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#2563eb",
        "color-brand-content": "#1e40af",
    },
    "dark_css_variables": {
        "color-brand-primary": "#60a5fa",
        "color-brand-content": "#93c5fd",
    },
}
# No static assets: docs/_static/ is gitignored, so the runner has no such
# directory and Sphinx warns -- fatal under the docs workflow's -W.
html_static_path = []

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
