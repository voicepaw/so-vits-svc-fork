# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from pathlib import Path
from typing import Any

from sphinx.application import Sphinx
from sphinx.ext import apidoc

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "SoftVC VITS Singing Voice Conversion Fork"
copyright = "2023, 34j"
author = "34j"
release = "0.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
]
napoleon_google_docstring = False

# The suffix of source filenames.
source_suffix = [".rst", ".md"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: list[str] = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# -- Automatically run sphinx-apidoc -----------------------------------------


def run_apidoc(_: Any) -> None:
    docs_path = Path(__file__).parent
    module_path = docs_path.parent / "src" / "so_vits_svc_fork"

    apidoc.main(
        [
            "--force",
            "--module-first",
            "-o",
            docs_path.as_posix(),
            module_path.as_posix(),
        ]
    )


def setup(app: Sphinx) -> None:
    app.connect("builder-inited", run_apidoc)
