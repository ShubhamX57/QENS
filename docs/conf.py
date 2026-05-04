""" 
Sphinx configuration for the qens documentation.

"""


import os, sys
sys.path.insert(0, os.path.abspath(".."))

project   = "qens"
copyright = "2026, QENS Analysis Contributors"
author    = "QENS Analysis Contributors"
release   = "2.0.0"
version   = "2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser"
]

autodoc_default_options = {
    "members": True, "undoc-members": False,
    "show-inheritance": True, "member-order": "bysource"
}
autodoc_typehints        = "description"
autosummary_generate     = True
napoleon_numpy_docstring = True
napoleon_use_param       = True
napoleon_use_rtype       = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy":  ("https://numpy.org/doc/stable", None),
    "scipy":  ("https://docs.scipy.org/doc/scipy", None)
}

myst_enable_extensions = [
    "colon_fence", "deflist", "dollarmath",
    "fieldlist", "html_image", "tasklist"
]
myst_dmath_double_inline = True

source_suffix  = {".rst": "restructuredtext", ".md": "markdown"}
master_doc     = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "qens"
html_theme_options = {
    "light_logo": "qens_logo_light.png",
    "dark_logo":  "qens_logo_dark.png",
    "light_css_variables": {
        "color-brand-primary":   "#0a7abf",
        "color-brand-content":   "#0a7abf",
        "font-stack":            "'DM Sans', -apple-system, sans-serif",
        "font-stack--monospace": "'IBM Plex Mono', monospace"
    },
    "dark_css_variables": {
        "color-brand-primary":        "#38bdf8",
        "color-brand-content":        "#38bdf8",
        "color-background-primary":   "#0b0f1a",
        "color-background-secondary": "#111827",
        "font-stack":                 "'DM Sans', -apple-system, sans-serif",
        "font-stack--monospace":      "'IBM Plex Mono', monospace"
    },
    "navigation_with_keys": True,
    "top_of_page_button":   "edit",
    "source_repository":    "https://github.com/ShubhamX57/QENS/",
    "source_branch":        "main",
    "source_directory":     "docs/"
}

html_static_path    = ["_static"]
html_css_files      = ["custom.css"]
html_show_sphinx    = False

copybutton_prompt_text      = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
