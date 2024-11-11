"""Sphinx configuration."""

project = "Chemoecology Tools"
author = "Scott Wolf"
copyright = "2024, Scott Wolf"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
