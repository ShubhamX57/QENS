import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "QENS"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

html_theme = 'python_docs_theme'

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

html_static_path = ["_static"]

def setup(app):
    app.add_css_file("custom.css")
