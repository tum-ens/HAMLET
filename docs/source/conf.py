import os
import sys

# Adjust the path as needed
sys.path.insert(0, os.path.abspath('../../hamlet'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'HAMLET'
copyright = '2024, Markus Doepfert'
author = 'Markus Doepfert'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',    # Extracts docstrings for API docs
    'sphinx.ext.napoleon',   # Supports Google and NumPy docstrings
    'sphinx.ext.viewcode',   # Adds source code links
    'sphinx.ext.autosummary' # Summarizes modules and classes
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Generate autosummary files
autosummary_generate = True
