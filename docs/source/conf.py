# -*- coding: utf-8 -*-

# -- Path setup --------------------------------------------------------------
import sys, os
print("i'm here!")
print(os.listdir( os.path.abspath('../../pygama')))
sys.path.insert(0, os.path.abspath('../../pygama'))
# exit()

import decoders.dataloading

import module, waveform
import pygama

# -- Project information -----------------------------------------------------
project = 'pygama'
copyright = '2018, Shanks, Meijer, Wiseman'
author = 'Shanks, Meijer, Wiseman'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx_gallery.gen_gallery',
]

sphinx_gallery_conf = {
    'examples_dirs': '../../examples',
    'gallery_dirs': 'gallery',
    'doc_module': ('module','waveform'),
    'backreferences_dir': '_as_gen',
}

plot_gallery = 'True'

autosummary_generate = True
autodoc_default_flags = ['members', 'undoc-members']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = 'bizstyle' # 'sphinxdoc'
html_static_path = ['_static']
htmlhelp_basename = 'pygama_doc'


