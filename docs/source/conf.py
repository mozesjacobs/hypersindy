import os
import sys

# -- Project information

project = 'HyperSINDy'
copyright = '2023, Mozes Jacobs'
author = 'Mozes Jacobs'

release = '0.0.1'
version = '0.0.3'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output
#html_theme = 'sphinx_rtd_theme'
html_theme = 'classic'

# -- Options for EPUB output
epub_show_urls = 'footnote'

# HyperSINDy path
#sys.path.append(os.path.abspath('../src/hypersindy/'))
#sys.path.append(os.path.abspath('../../src/'))
sys.path.append(os.path.abspath('src/'))

#try:
#  import hypersindy
  #from .hypersindy import HyperSINDy
#  print(hypersindy.HyperSINDy())
  #print(help(hypersindy))
  #from .hypersindy import HyperSINDy
  #print(dir(hypersindy))
  #print(hypersindy['__package__'])
  #print(hypersindy)
  #print(HyperSINDy())
  #from hypersindy import HyperSINDy
  #import hypersindy.HyperSINDy
#except ImportError:
#  print("oh no 2")