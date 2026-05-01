import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "QENS"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]


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

html_theme_path = ['_themes']             
html_theme = 'my_scientific_theme'        

html_static_path = ['_static']            
