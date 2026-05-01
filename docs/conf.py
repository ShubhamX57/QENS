import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "QENS"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

html_theme = "sphinx_rtd_theme"

html_logo = "../logo/qens_logo_dark.png"

html_theme_options = {
    "logo_only": True,
    "display_version": False,
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
]

.wy-side-nav-search {
    display: block;
    width: 300px;
    padding: .809em;
    margin-bottom: .809em;
    z-index: 200;
    background-color: #000000;
    text-align: center;
    color: #fcfcfc
}
