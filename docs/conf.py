# Configuration file for the Sphinx documentation builder.
# Full docs: https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import subprocess
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------
# Add your package (src/step) to sys.path so autodoc can find it
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "eeg-step"
author = "Alexander Enge"
copyright = f"{datetime.now().year}, {author}"


# Get version automatically from Git tags (e.g. v0.1.0 → 0.1.0)
def get_version() -> str:
    raw = (
        subprocess.check_output(["git", "describe", "--tags"], text=True)
        .strip()
        .lstrip("v")
    )
    parts = raw.split("-")
    if len(parts) == 3:
        base, commits, _ = parts
        return f"{base}.dev{commits}"
    return raw


version = release = get_version()

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",  # automatically include docstrings
    "sphinx.ext.autosummary",  # create summary tables
    "sphinx.ext.napoleon",  # Google/NumPy docstring style
    "sphinx.ext.viewcode",  # add source code links
    "sphinx.ext.intersphinx",  # link to external docs
    "sphinx.ext.todo",  # support TODOs in docs
    "sphinx_autodoc_typehints",  # show type hints in function signatures
    "myst_parser",  # Markdown support
]

# Autosummary generation
autosummary_generate = True
autodoc_typehints = "description"
autodoc_preserve_defaults = True

# Markdown (MyST) configuration
myst_enable_extensions = [
    "deflist",
    "colon_fence",
    "tasklist",
    "attrs_block",
]

# Intersphinx links (optional, useful for scientific stack)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "mne": ("https://mne.tools/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_title = f"{project} {version}"
html_show_sourcelink = True

# -- Options for TODOs -------------------------------------------------------
todo_include_todos = True

# -- Miscellaneous settings --------------------------------------------------
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Suppress warnings about missing type hints
suppress_warnings = ["autodoc.import_object"]
