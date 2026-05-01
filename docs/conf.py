import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = 'QENS'

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc"
]

html_theme = 'sphinx_rtd_theme'

html_logo = "../logo/qens_logo_dark.png"




extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon"
]

html_theme_options = {
    "logo_only": True,
    "display_version": False,
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
}
