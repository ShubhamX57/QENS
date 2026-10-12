"""Sphinx configuration for the uqens documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

import uqens  # noqa: E402

project = "uqens"
copyright = "2026, Shubham Shinde and uqens contributors"
author = "Shubham Shinde"
release = uqens.__version__
version = uqens.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
]

autodoc_default_options = {"members": True, "undoc-members": False, "member-order": "bysource"}
autodoc_typehints = "description"
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "fieldlist", "html_image"]
myst_heading_anchors = 3

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "uqens"
html_theme_options = {
    "light_logo": "qens_logo_light.png",
    "dark_logo": "qens_logo_dark.png",
    "light_css_variables": {
        "color-brand-primary": "#0a7abf",
        "color-brand-content": "#0a7abf",
        "font-stack": "'DM Sans', -apple-system, sans-serif",
        "font-stack--monospace": "'IBM Plex Mono', monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#38bdf8",
        "color-brand-content": "#38bdf8",
        "color-background-primary": "#0b0f1a",
        "color-background-secondary": "#111827",
        "font-stack": "'DM Sans', -apple-system, sans-serif",
        "font-stack--monospace": "'IBM Plex Mono', monospace",
    },
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/ShubhamX57/QENS/",
    "source_branch": "v2",
    "source_directory": "docs/",
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sphinx = False

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
