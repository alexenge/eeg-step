# Configuration file for the Sphinx documentation builder.

import os
import sys

# -- Path setup --------------------------------------------------------------

sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------

project = "eeg-step"
author = "Alexander Enge"
release = "0.1.0"  # optional placeholder, can be automated later

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx_copybutton",
]

autosummary_generate = True
myst_enable_extensions = ["colon_fence"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
