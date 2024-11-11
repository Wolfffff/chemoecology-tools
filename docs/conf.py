"""Sphinx configuration."""

project = "Chemoecology Tools"
author = "Scott Wolf"
copyright = "2024, Scott Wolf"

extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx_design",
]

# MyST settings
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

# Theme settings
html_theme = "furo"
html_favicon = "_static/favicon.png"
html_logo = "_static/logo.png"
html_static_path = ["_static"]

autodoc_typehints = "description"

# Notebook execution
nb_execution_mode = "force"
nb_execution_timeout = 300

exclude_patterns = ["jupyter_execute"]
