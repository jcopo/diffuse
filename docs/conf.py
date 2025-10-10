# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
# Configuration file for the Sphinx documentation builder.
# Based on Flax documentation configuration

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "Diffuse"
copyright = "2025, Diffuse Team"
author = "Diffuse Team"
release = "0.1.0"
version = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "sphinxcontrib.tikz",
    "sphinxcontrib.bibtex",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The suffix(es) of source filenames.
source_suffix = [".rst", ".md", ".ipynb"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = f"{project}"
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"

html_theme_options = {
    "repository_url": "https://github.com/jcopo/diffuse",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": True,
    "path_to_docs": "docs/",
    "repository_branch": "main",
    "launch_buttons": {
        "colab_url": "https://colab.research.google.com",
    },
    "show_navbar_depth": 2,
}

html_context = {
    "display_github": True,
    "github_user": "jcopo",
    "github_repo": "diffuse",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Extension configuration -------------------------------------------------

# Autosectionlabel settings
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 3

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Autodoc settings
autodoc_typehints = "signature"
autosummary_generate = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "flax": ("https://flax.readthedocs.io/en/latest/", None),
}

# MyST settings
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# Enable footnote references for citations
myst_footnote_transition = True

# Notebook execution
nb_execution_mode = "off"
nb_execution_timeout = 100

# MyST-NB configuration
nb_render_image_options = {"width": "100%"}

# Configure toggleable code cells
togglebutton_hint = "Click to show/hide code"
togglebutton_hint_hide = "Click to hide code"

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# TikZ configuration
tikz_proc_suite = "GhostScript"  # Works well for most systems
tikz_transparent = True

# BibTeX configuration
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "plain"
bibtex_reference_style = "author_year"
