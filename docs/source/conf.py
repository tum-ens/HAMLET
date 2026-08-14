from pathlib import Path

# The `sys.path.insert(0, '../../hamlet')` that used to be here was dead: it put the *inside* of
# the package on the path, so `import hamlet` would never have worked through it -- and nothing
# needs it to. There is not one `automodule`/`autoclass`/`autosummary` directive under
# `docs/source/`; the documentation is hand-written and imports nothing. That is also why the
# docs build installs Sphinx alone rather than HAMLET and its whole dependency tree.

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'HAMLET'
copyright = '2024, Markus Doepfert'
author = 'Markus Doepfert'
# Read from the same `VERSION` file that `pyproject.toml` reads, rather than restated here. It
# said 1.0.0 while the repository was on 1.2.0.
release = Path(__file__).resolve().parents[2].joinpath('VERSION').read_text(encoding='utf-8').strip()

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
