# Configuration file for the Sphinx documentation builder.

import sys
from pathlib import Path

sys.path.insert(0, Path('../../src').resolve().as_posix())

project = 'pygama'
copyright = '2020, the LEGEND Collaboration'

extensions = [
    'sphinx.ext.githubpages',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
    'sphinx_multiversion',
    'sphinx_copybutton',
    'myst_parser'
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
master_doc = 'index'
language = 'python'
# in _templates/ we have a custom layout.html to include the version menu
# (adapted from sphinx-multiversion docs)
templates_path = ['_templates']
pygments_style = 'sphinx'

# readthedocs.io Sphinx theme
html_theme = 'sphinx_rtd_theme'

# list here pygama dependencies that are not required for building docs and
# could be unmet at build time
autodoc_mock_imports = [
    'pygama._version',
    'pandas',
    # 'numpy',
    'matplotlib',
    'mplhep',
    'scipy',
    'numba',
    'pytest',
    'pyhf',
    'awkward',
    'iminuit',
    'boost-histogram',
    'hepunits',
    'hepstats',
    'uproot',
    'h5py',
    'pint',
    'pyfftw',
    'tqdm',
    'tinydb',
    'parse'
]

# sphinx-napoleon
# enforce consistent usage of NumPy-style docstrings
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'numba': ('https://numba.readthedocs.io/en/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'iminuit': ('https://iminuit.readthedocs.io/en/stable', None),
    'h5py': ('https://docs.h5py.org/en/stable', None)
}

# sphinx-autodoc
# Include __init__() docstring in class docstring
autoclass_content = 'both'

# sphinx-multiversion

# For now, we include only (certain) branches when building docs.
# To add a specific release to the list of versions for which docs should be build,
# one must create a new branch named `releases/...`
smv_branch_whitelist = r'^(main|refactor|releases/.*)$'
smv_tag_whitelist = '^$'
smv_released_pattern = '^$'
smv_outputdir_format = '{ref.name}'
smv_prefer_remote_refs = False

# HACK: we need to regenerate the API documentation before the actual build,
# but it's not possible with the current sphinx-multiversion. Changes have been
# proposed in this PR: https://github.com/Holzhaus/sphinx-multiversion/pull/62
# but there's no timeline for merging yet. For the following option to be considered,
# one needs to install the sphinx-multiversion-pre-post-build fork from PyPI
smv_prebuild_command = 'make -ik apidoc'

# The right way to find all docs versions is to look for matching branches on
# the default remote
smv_remote_whitelist = r'^origin$'
